import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class RadarLidarBEV_dataset(Dataset):
    """
    Class for loading radar BEV images and paired lidar BEV images
    """
    def __init__(self,
                 data_dir,
                 train,
                 radar_scale=256,
                 lidar_scale=256,
                 num_radar_views=4,
                 radar_channels=3,
                 lidar_channels=3):
        """
        :param data_dir: (str) path to data folder
        :param train: (bool) are we training or testing
        :param radar_scale: (int) resize radar BEV images to this value
        :param lidar_scale: (int) resize lidar BEV images to this value
        :param num_radar_views: (int) number of radar views to use
        :param radar_channels: (int) number of channels in radar BEV
        :param lidar_channels: (int) number of channels in lidar BEV
        
        :return data: (dict)
        'radar_bev': stacked radar BEV images torch.Tensor(num_views, channels, scale, scale)
        'lidar_bev': lidar BEV image torch.Tensor(channels, scale, scale)
        'scene_name': scene identifier (str)
        """
        
        self.data_dir = data_dir
        self.train = train
        self.radar_scale = radar_scale
        self.lidar_scale = lidar_scale
        self.num_radar_views = num_radar_views
        self.radar_channels = radar_channels
        self.lidar_channels = lidar_channels
        
        # 데이터 split 파일 읽기
        split_file = 'train_split.txt' if train else 'test_split.txt'
        split_path = os.path.join(data_dir, split_file)
        
        if os.path.exists(split_path):
            with open(split_path, 'r') as f:
                self.scene_list = [line.strip() for line in f.readlines()]
        else:
            # split 파일이 없으면 전체 디렉토리 스캔
            self.scene_list = self._scan_data_directory()
        
        print(f"Loaded {'train' if train else 'test'} dataset with {len(self.scene_list)} scenes")
    
    def _scan_data_directory(self):
        """데이터 디렉토리를 스캔하여 scene 리스트 생성"""
        scene_dirs = []
        for scene_dir in sorted(glob.glob(os.path.join(self.data_dir, "*"))):
            if os.path.isdir(scene_dir):
                # radar와 lidar 파일이 모두 존재하는지 확인
                radar_files = glob.glob(os.path.join(scene_dir, "radar_*.npy"))
                lidar_file = os.path.join(scene_dir, "lidar_bev.npy")
                
                if len(radar_files) >= self.num_radar_views and os.path.exists(lidar_file):
                    scene_dirs.append(os.path.basename(scene_dir))
        
        return scene_dirs
    
    def __len__(self):
        return len(self.scene_list)
    
    def __getitem__(self, idx):
        scene_name = self.scene_list[idx]
        scene_dir = os.path.join(self.data_dir, scene_name)
        
        # Radar BEV 데이터 로드
        radar_bevs = []
        radar_files = sorted(glob.glob(os.path.join(scene_dir, "radar_*.npy")))[:self.num_radar_views]
        
        for radar_file in radar_files:
            radar_bev = np.load(radar_file).astype(np.float32)
            # 정규화 (0-1 범위로)
            if radar_bev.max() > radar_bev.min():
                radar_bev = (radar_bev - radar_bev.min()) / (radar_bev.max() - radar_bev.min())
            
            # 채널 차원이 없으면 추가
            if len(radar_bev.shape) == 2:
                radar_bev = radar_bev[np.newaxis, ...]  # (H, W) -> (1, H, W)
            
            # 리사이즈 (필요한 경우)
            if radar_bev.shape[-1] != self.radar_scale or radar_bev.shape[-2] != self.radar_scale:
                radar_bev = torch.nn.functional.interpolate(
                    torch.tensor(radar_bev).unsqueeze(0), 
                    size=(self.radar_scale, self.radar_scale), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0).numpy()
            
            radar_bevs.append(radar_bev)
        
        radar_bevs = np.stack(radar_bevs, axis=0)  # (num_views, C, H, W)
        
        # Lidar BEV 데이터 로드
        lidar_file = os.path.join(scene_dir, "lidar_bev.npy")
        lidar_bev = np.load(lidar_file).astype(np.float32)
        
        # 정규화
        if lidar_bev.max() > lidar_bev.min():
            lidar_bev = (lidar_bev - lidar_bev.min()) / (lidar_bev.max() - lidar_bev.min())
        
        # 채널 차원이 없으면 추가
        if len(lidar_bev.shape) == 2:
            lidar_bev = lidar_bev[np.newaxis, ...]  # (H, W) -> (1, H, W)
        
        # 리사이즈 (필요한 경우)
        if lidar_bev.shape[-1] != self.lidar_scale or lidar_bev.shape[-2] != self.lidar_scale:
            lidar_bev = torch.nn.functional.interpolate(
                torch.tensor(lidar_bev).unsqueeze(0), 
                size=(self.lidar_scale, self.lidar_scale), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0).numpy()
        
        return {
            'radar_bevs': torch.tensor(radar_bevs),
            'lidar_bev': torch.tensor(lidar_bev),
            'scene_name': scene_name
        }


class RadarLidarBEV_dataset_simple(Dataset):
    """
    간단한 버전: 단일 파일에서 radar와 lidar BEV 데이터를 로드
    """
    def __init__(self, data_file, train=True, radar_scale=256, lidar_scale=256):
        """
        :param data_file: (str) .npz 파일 경로 (radar_bevs, lidar_bevs 키를 포함)
        """
        self.data = np.load(data_file)
        self.radar_bevs = self.data['radar_bevs']
        self.lidar_bevs = self.data['lidar_bevs']
        self.radar_scale = radar_scale
        self.lidar_scale = lidar_scale
        
        assert len(self.radar_bevs) == len(self.lidar_bevs), "Radar and Lidar data length mismatch"
        
        # Train/test split (80/20)
        total_len = len(self.radar_bevs)
        split_idx = int(0.8 * total_len)
        
        if train:
            self.radar_bevs = self.radar_bevs[:split_idx]
            self.lidar_bevs = self.lidar_bevs[:split_idx]
        else:
            self.radar_bevs = self.radar_bevs[split_idx:]
            self.lidar_bevs = self.lidar_bevs[split_idx:]
            
        print(f"Loaded {'train' if train else 'test'} dataset with {len(self.radar_bevs)} samples")
    
    def __len__(self):
        return len(self.radar_bevs)
    
    def __getitem__(self, idx):
        radar_bev = self.radar_bevs[idx].astype(np.float32)
        lidar_bev = self.lidar_bevs[idx].astype(np.float32)
        
        # 정규화
        radar_bev = (radar_bev - radar_bev.min()) / (radar_bev.max() - radar_bev.min() + 1e-8)
        lidar_bev = (lidar_bev - lidar_bev.min()) / (lidar_bev.max() - lidar_bev.min() + 1e-8)
        
        return {
            'radar_bevs': torch.tensor(radar_bev),
            'lidar_bev': torch.tensor(lidar_bev),
            'scene_name': f"scene_{idx}"
        }
