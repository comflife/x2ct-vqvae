import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class NuScenesRadarDataset(Dataset):
    """
    NuScenes radar 데이터를 로드하는 데이터셋
    UltraLiDAR에서 생성된 radar 데이터 형식에 맞춤
    """
    def __init__(self, 
                 data_root,
                 ann_file,
                 train=True,
                 radar_channels=6,  # x, y, z, rcs, vx, vy
                 bev_size=640,
                 point_cloud_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]):
        """
        Args:
            data_root: NuScenes 데이터 루트 경로
            ann_file: annotation 파일 경로 (.pkl)
            train: 훈련용/테스트용 구분
            radar_channels: radar 점군 채널 수
            bev_size: BEV 이미지 크기
            point_cloud_range: 점군 범위 [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        self.data_root = data_root
        self.ann_file = ann_file
        self.train = train
        self.radar_channels = radar_channels
        self.bev_size = bev_size
        self.point_cloud_range = point_cloud_range
        
        # Annotation 파일 로드
        with open(ann_file, 'rb') as f:
            self.data_infos = pickle.load(f)
        
        if isinstance(self.data_infos, dict):
            self.data_infos = self.data_infos.get('infos', self.data_infos)
        
        print(f"Loaded NuScenes radar dataset: {len(self.data_infos)} samples")
        
    def __len__(self):
        return len(self.data_infos)
    
    def points_to_bev(self, points):
        """
        점군 데이터를 BEV 이미지로 변환
        Args:
            points: numpy array [N, 6] (x, y, z, rcs, vx, vy)
        Returns:
            bev: numpy array [H, W] BEV 이미지
        """
        if len(points) == 0:
            bev_h = self.bev_size[0] if isinstance(self.bev_size, (list, tuple)) else self.bev_size
            bev_w = self.bev_size[1] if isinstance(self.bev_size, (list, tuple)) else self.bev_size
            return np.zeros((bev_h, bev_w), dtype=np.float32)
        
        x_min, y_min = self.point_cloud_range[0], self.point_cloud_range[1]
        x_max, y_max = self.point_cloud_range[3], self.point_cloud_range[4]
        
        # BEV 크기 설정
        bev_h = self.bev_size[0] if isinstance(self.bev_size, (list, tuple)) else self.bev_size
        bev_w = self.bev_size[1] if isinstance(self.bev_size, (list, tuple)) else self.bev_size
        
        # 점들을 BEV 그리드에 매핑
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        intensities = points[:, 3]  # RCS 값
        
        # 범위 내 점들만 선택
        valid_mask = (x_coords >= x_min) & (x_coords <= x_max) & \
                    (y_coords >= y_min) & (y_coords <= y_max)
        
        if not np.any(valid_mask):
            return np.zeros((bev_h, bev_w), dtype=np.float32)
        
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]
        intensities = intensities[valid_mask]
        
        # 픽셀 좌표로 변환
        x_img = ((x_coords - x_min) / (x_max - x_min) * (bev_w - 1)).astype(int)
        y_img = ((y_coords - y_min) / (y_max - y_min) * (bev_h - 1)).astype(int)
        
        # 범위 체크
        valid_pixels = (x_img >= 0) & (x_img < bev_w) & \
                      (y_img >= 0) & (y_img < bev_h)
        
        x_img = x_img[valid_pixels]
        y_img = y_img[valid_pixels]
        intensities = intensities[valid_pixels]
        
        # BEV 이미지 생성
        bev = np.zeros((bev_h, bev_w), dtype=np.float32)
        
        if len(x_img) > 0:
            # 동일한 픽셀에 여러 점이 있는 경우 최대값 사용
            for i in range(len(x_img)):
                bev[y_img[i], x_img[i]] = max(bev[y_img[i], x_img[i]], intensities[i])
        
        return bev
    
    def load_radar_points(self, info):
        """
        radar 점군 데이터 로드
        """
        # NuScenes radar 파일 경로 구성
        if 'radar_path' in info:
            radar_path = info['radar_path']
        elif 'sweeps' in info and len(info['sweeps']) > 0:
            # sweeps에서 radar 데이터 찾기
            radar_sweeps = [s for s in info['sweeps'] if 'RADAR' in s.get('sensor', '')]
            if radar_sweeps:
                radar_path = radar_sweeps[0]['data_path']
            else:
                return np.zeros((0, self.radar_channels))
        else:
            return np.zeros((0, self.radar_channels))
        
        # 절대 경로 구성
        if not os.path.isabs(radar_path):
            radar_path = os.path.join(self.data_root, radar_path)
        
        try:
            # .pcd 파일 또는 .bin 파일 로드
            if radar_path.endswith('.pcd'):
                # PCD 파일 로드 (간단한 구현)
                points = self.load_pcd_file(radar_path)
            elif radar_path.endswith('.bin'):
                # Binary 파일 로드
                points = np.fromfile(radar_path, dtype=np.float32).reshape(-1, self.radar_channels)
            else:
                # Numpy 파일 시도
                points = np.load(radar_path)
                if points.ndim == 1:
                    points = points.reshape(-1, self.radar_channels)
            
            return points.astype(np.float32)
            
        except Exception as e:
            print(f"Warning: Failed to load radar data from {radar_path}: {e}")
            return np.zeros((0, self.radar_channels))
    
    def load_pcd_file(self, pcd_path):
        """
        간단한 PCD 파일 로더
        """
        try:
            with open(pcd_path, 'r') as f:
                lines = f.readlines()
            
            # 헤더 파싱
            data_start = 0
            for i, line in enumerate(lines):
                if line.startswith('DATA'):
                    data_start = i + 1
                    break
            
            # 데이터 파싱
            points = []
            for line in lines[data_start:]:
                if line.strip():
                    values = list(map(float, line.strip().split()))
                    if len(values) >= self.radar_channels:
                        points.append(values[:self.radar_channels])
            
            return np.array(points, dtype=np.float32)
            
        except Exception as e:
            print(f"Error loading PCD file {pcd_path}: {e}")
            return np.zeros((0, self.radar_channels))
    
    def __getitem__(self, idx):
        info = self.data_infos[idx]
        
        # Radar 점군 로드
        radar_points = self.load_radar_points(info)
        
        # BEV 이미지 변환
        radar_bev = self.points_to_bev(radar_points)
        
        # 정규화 (0-1 범위)
        if radar_bev.max() > 0:
            radar_bev = radar_bev / radar_bev.max()
        
        # 데이터 반환
        sample = {
            'radar_points': torch.tensor(radar_points, dtype=torch.float32),
            'radar_bev': torch.tensor(radar_bev, dtype=torch.float32).unsqueeze(0),  # [1, H, W]
            'sample_idx': idx,
            'token': info.get('token', f'sample_{idx}')
        }
        
        return sample


class NuScenesRadarLidarDataset(NuScenesRadarDataset):
    """
    NuScenes radar와 lidar를 함께 로드하는 데이터셋
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def load_lidar_points(self, info):
        """
        lidar 점군 데이터 로드
        """
        if 'lidar_path' in info:
            lidar_path = info['lidar_path']
        elif 'pts_filename' in info:
            lidar_path = info['pts_filename']
        else:
            return np.zeros((0, 4))  # x, y, z, intensity
        
        # 절대 경로 구성
        if not os.path.isabs(lidar_path):
            lidar_path = os.path.join(self.data_root, lidar_path)
        
        try:
            if lidar_path.endswith('.bin'):
                # NuScenes lidar는 보통 .bin 형식
                points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)  # x,y,z,intensity,ring
                points = points[:, :4]  # ring 제거
            else:
                points = np.load(lidar_path)
                if points.shape[1] > 4:
                    points = points[:, :4]
            
            return points.astype(np.float32)
            
        except Exception as e:
            print(f"Warning: Failed to load lidar data from {lidar_path}: {e}")
            return np.zeros((0, 4))
    
    def __getitem__(self, idx):
        info = self.data_infos[idx]
        
        # Radar 데이터
        radar_points = self.load_radar_points(info)
        radar_bev = self.points_to_bev(radar_points)
        
        # Lidar 데이터
        lidar_points = self.load_lidar_points(info)
        lidar_bev = self.points_to_bev(lidar_points)  # lidar도 BEV로 변환
        
        # 정규화
        if radar_bev.max() > 0:
            radar_bev = radar_bev / radar_bev.max()
        if lidar_bev.max() > 0:
            lidar_bev = lidar_bev / lidar_bev.max()
        
        sample = {
            'radar_points': torch.tensor(radar_points, dtype=torch.float32),
            'radar_bev': torch.tensor(radar_bev, dtype=torch.float32).unsqueeze(0),
            'lidar_points': torch.tensor(lidar_points, dtype=torch.float32),
            'lidar_bev': torch.tensor(lidar_bev, dtype=torch.float32).unsqueeze(0),
            'sample_idx': idx,
            'token': info.get('token', f'sample_{idx}')
        }
        
        return sample
