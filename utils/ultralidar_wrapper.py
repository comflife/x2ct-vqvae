import torch
import torch.nn as nn
import numpy as np
import sys
import os
from pathlib import Path

class UltraLiDARRadarWrapper(nn.Module):
    """
    UltraLiDAR 모델을 VQGAN 인터페이스로 래핑하는 클래스
    """
    def __init__(self, config_path, checkpoint_path):
        super().__init__()
        
        # UltraLiDAR 경로 추가
        ultralidar_path = Path(checkpoint_path).parent.parent  # /home/byounggun/r2l/UltraLiDAR_nusc_waymo
        sys.path.insert(0, str(ultralidar_path))
        
        try:
            # UltraLiDAR 플러그인 임포트
            import plugin
            from mmcv import Config
            from mmdet3d.models import build_model
            
            # 설정 로드
            cfg = Config.fromfile(config_path)
            cfg.model.pretrained = None
            cfg.model.model_type = "codebook_training"
            cfg.model.train_cfg = None
            
            # 모델 빌드
            self.model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
            
            # 체크포인트 로드
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict'))
            
            # 모듈 prefix 제거
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_key = k[7:]
                else:
                    new_key = k
                new_state_dict[new_key] = v
            
            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.eval()
            
            # 설정 저장
            self.cfg = cfg
            self.voxel_size = cfg.model.voxelizer.step
            self.point_cloud_range = [
                cfg.model.voxelizer.x_min, cfg.model.voxelizer.y_min, cfg.model.voxelizer.z_min,
                cfg.model.voxelizer.x_max, cfg.model.voxelizer.y_max, cfg.model.voxelizer.z_max
            ]
            
            print(f"✅ UltraLiDAR model loaded successfully from {checkpoint_path}")
            
        except Exception as e:
            print(f"❌ Failed to load UltraLiDAR model: {e}")
            raise e
    
    def points_to_bev(self, points):
        """
        점군 데이터를 BEV 이미지로 변환
        Args:
            points: torch.Tensor [N, 6] (x, y, z, intensity, vx, vy)
        Returns:
            bev: torch.Tensor [1, H, W] BEV 이미지
        """
        if points.dim() == 3:
            points = points.squeeze(0)
        
        # BEV 파라미터
        x_min, y_min = self.point_cloud_range[0], self.point_cloud_range[1]
        x_max, y_max = self.point_cloud_range[3], self.point_cloud_range[4]
        bev_size = 640  # UltraLiDAR 사용 크기
        
        # 점들을 BEV 그리드에 매핑
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        intensities = points[:, 3]
        
        # 범위 내 점들만 선택
        valid_mask = (x_coords >= x_min) & (x_coords <= x_max) & \
                    (y_coords >= y_min) & (y_coords <= y_max)
        
        if valid_mask.sum() == 0:
            return torch.zeros(1, bev_size, bev_size)
        
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]
        intensities = intensities[valid_mask]
        
        # 픽셀 좌표로 변환
        x_img = ((x_coords - x_min) / (x_max - x_min) * (bev_size - 1)).long()
        y_img = ((y_coords - y_min) / (y_max - y_min) * (bev_size - 1)).long()
        
        # BEV 이미지 생성
        bev = torch.zeros(bev_size, bev_size, device=points.device)
        bev[y_img, x_img] = intensities
        
        return bev.unsqueeze(0)  # [1, H, W]
    
    def bev_to_points(self, bev_image, intensity_threshold=0.1):
        """
        BEV 이미지를 점군으로 변환
        """
        if bev_image.dim() == 4:
            bev_image = bev_image.squeeze(0)  # [C, H, W]
        if bev_image.dim() == 3:
            bev_image = bev_image.squeeze(0)  # [H, W]
        
        y_indices, x_indices = torch.where(bev_image > intensity_threshold)
        
        if len(x_indices) == 0:
            return torch.zeros((0, 6), device=bev_image.device)
        
        # BEV 좌표를 실제 좌표로 변환
        x_min, y_min = self.point_cloud_range[0], self.point_cloud_range[1]
        x_max, y_max = self.point_cloud_range[3], self.point_cloud_range[4]
        bev_size = bev_image.shape[-1]
        
        x_coords = (x_indices.float() / (bev_size - 1)) * (x_max - x_min) + x_min
        y_coords = (y_indices.float() / (bev_size - 1)) * (y_max - y_min) + y_min
        z_coords = torch.zeros_like(x_coords)
        
        intensities = bev_image[y_indices, x_indices]
        vx_comp = torch.zeros_like(x_coords)
        vy_comp = torch.zeros_like(x_coords)
        
        points = torch.stack([x_coords, y_coords, z_coords, intensities, vx_comp, vy_comp], dim=1)
        
        return points
    
    def encoder(self, x):
        """
        VQGAN encoder 인터페이스
        Args:
            x: BEV 이미지 [B, C, H, W] 또는 점군 리스트
        Returns:
            latents: 인코딩된 특징
        """
        # 점군이 입력으로 들어온 경우 BEV로 변환
        if isinstance(x, list):
            bev_images = []
            for points in x:
                bev = self.points_to_bev(points)
                bev_images.append(bev)
            x = torch.stack(bev_images, dim=0)
        
        # UltraLiDAR 모델에서는 점군을 voxel로 변환 후 인코딩
        # 여기서는 BEV를 점군으로 변환 후 UltraLiDAR 파이프라인 사용
        batch_size = x.shape[0]
        all_latents = []
        
        for i in range(batch_size):
            bev = x[i]  # [C, H, W]
            
            # BEV를 점군으로 변환
            points = self.bev_to_points(bev)
            
            if len(points) == 0:
                # 빈 점군인 경우 zero latent 반환
                latents = torch.zeros(1, 1024, 80, 80, device=x.device)
            else:
                # UltraLiDAR 인코딩 파이프라인
                with torch.no_grad():
                    try:
                        # Voxelization
                        voxels = self.model.voxelizer([points])
                        
                        # Encoding
                        latents = self.model.lidar_encoder(voxels)
                        
                        # Pre-quantization (있는 경우)
                        if hasattr(self.model, 'pre_quant'):
                            latents = self.model.pre_quant(latents)
                            
                    except Exception as e:
                        print(f"Warning: Encoding failed for sample {i}: {e}")
                        latents = torch.zeros(1, 1024, 80, 80, device=x.device)
            
            all_latents.append(latents)
        
        return torch.cat(all_latents, dim=0)
    
    def quantize(self, latents):
        """
        VQGAN quantize 인터페이스
        """
        with torch.no_grad():
            try:
                quant_latents, emb_loss, info = self.model.vector_quantizer(latents)
                
                # info에서 min_encoding_indices 추출
                if isinstance(info, dict) and 'min_encoding_indices' in info:
                    min_encoding_indices = info['min_encoding_indices']
                elif hasattr(info, 'unsqueeze'):  # info가 직접 indices인 경우
                    min_encoding_indices = info
                else:
                    # fallback: latents에서 코드북 인덱스 계산
                    min_encoding_indices = torch.argmax(quant_latents.view(-1, quant_latents.shape[-1]), dim=-1)
                
                quant_stats = {"min_encoding_indices": min_encoding_indices}
                
                return quant_latents, emb_loss, quant_stats
                
            except Exception as e:
                print(f"Warning: Quantization failed: {e}")
                batch_size = latents.shape[0]
                fake_quant = torch.zeros_like(latents)
                fake_loss = torch.tensor(0.0, device=latents.device)
                fake_indices = torch.zeros(batch_size * 80 * 80, dtype=torch.long, device=latents.device)
                return fake_quant, fake_loss, {"min_encoding_indices": fake_indices}
    
    def generator(self, quantized_latents):
        """
        VQGAN generator/decoder 인터페이스
        """
        with torch.no_grad():
            try:
                # UltraLiDAR decoder 사용
                reconstructed = self.model.lidar_decoder(quantized_latents)
                
                # Sigmoid 적용하여 확률로 변환
                reconstructed = torch.sigmoid(reconstructed)
                
                return reconstructed
                
            except Exception as e:
                print(f"Warning: Generation failed: {e}")
                # Fallback: 빈 BEV 반환
                batch_size = quantized_latents.shape[0]
                return torch.zeros(batch_size, 1, 640, 640, device=quantized_latents.device)
    
    def forward(self, x):
        """
        전체 순전파 (encoding -> quantization -> decoding)
        """
        latents = self.encoder(x)
        quant_latents, emb_loss, quant_stats = self.quantize(latents)
        reconstructed = self.generator(quant_latents)
        
        return {
            'reconstructed': reconstructed,
            'latents': latents,
            'quant_latents': quant_latents,
            'emb_loss': emb_loss,
            'quant_stats': quant_stats
        }


def load_ultralidar_radar_model(checkpoint_path, config_path=None):
    """
    UltraLiDAR radar 모델을 로드하는 헬퍼 함수
    """
    if config_path is None:
        # 기본 config 경로 추정
        ultralidar_path = Path(checkpoint_path).parent.parent
        config_path = ultralidar_path / "configs" / "ultralidar_nusc_radar_debug.py"
    
    try:
        model = UltraLiDARRadarWrapper(str(config_path), checkpoint_path)
        return model
    except Exception as e:
        print(f"❌ Failed to load UltraLiDAR radar model: {e}")
        raise e


def load_ultralidar_lidar_model(checkpoint_path, config_path=None):
    """
    UltraLiDAR lidar 모델을 로드하는 헬퍼 함수
    """
    if config_path is None:
        # 기본 config 경로 추정 - lidar 모델용
        ultralidar_path = Path(checkpoint_path).parent.parent
        possible_configs = [
            ultralidar_path / "configs" / "ultralidar_nusc.py",
            ultralidar_path / "configs" / "ultralidar_nusc_stage2.py",
            ultralidar_path / "configs" / "ultralidar_lidar.py"
        ]
        for pconfig in possible_configs:
            if pconfig.exists():
                config_path = str(pconfig)
                break
        else:
            config_path = None
    
    try:
        # lidar 모델도 동일한 wrapper 클래스 사용 가능
        model = UltraLiDARRadarWrapper(str(config_path), checkpoint_path)
        return model
    except Exception as e:
        print(f"❌ Failed to load UltraLiDAR lidar model: {e}")
        raise e
