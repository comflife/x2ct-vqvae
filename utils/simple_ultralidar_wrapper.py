import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path


class SimpleUltraLiDARWrapper(nn.Module):
    """
    UltraLiDAR 체크포인트에서 가중치만 추출하는 간단한 래퍼
    mmcv 의존성 없이 동작
    """
    def __init__(self, checkpoint_path, model_type='radar'):
        super().__init__()
        
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        
        # 체크포인트 로드
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # state_dict 추출
        if 'state_dict' in checkpoint:
            self.state_dict_full = checkpoint['state_dict']
        elif 'model' in checkpoint:
            self.state_dict_full = checkpoint['model']
        else:
            self.state_dict_full = checkpoint
        
        print(f"✅ Checkpoint loaded successfully")
        print(f"📊 Available keys (first 10): {list(self.state_dict_full.keys())[:10]}")
        
        # 간단한 더미 모델 생성 (VQGAN 인터페이스 호환)
        self.encoder_weights = self._extract_encoder_weights()
        self.decoder_weights = self._extract_decoder_weights()
        self.quantizer_weights = self._extract_quantizer_weights()
        
    def _extract_encoder_weights(self):
        """인코더 가중치 추출"""
        encoder_weights = {}
        for key, value in self.state_dict_full.items():
            if 'lidar_encoder' in key or 'encoder' in key:
                encoder_weights[key] = value
        return encoder_weights
    
    def _extract_decoder_weights(self):
        """디코더 가중치 추출"""
        decoder_weights = {}
        for key, value in self.state_dict_full.items():
            if 'lidar_decoder' in key or 'decoder' in key or 'generator' in key:
                decoder_weights[key] = value
        return decoder_weights
    
    def _extract_quantizer_weights(self):
        """양자화기 가중치 추출"""
        quantizer_weights = {}
        for key, value in self.state_dict_full.items():
            if 'vector_quantizer' in key or 'quantize' in key:
                quantizer_weights[key] = value
        return quantizer_weights
    
    def points_to_bev(self, points, bev_size=640, point_cloud_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]):
        """점군을 BEV로 변환 (간단한 구현)"""
        if isinstance(points, list):
            if len(points) == 0:
                return torch.zeros(1, bev_size, bev_size)
            points = points[0]
        
        if len(points) == 0:
            return torch.zeros(1, bev_size, bev_size)
        
        if points.dim() == 3:
            points = points.squeeze(0)
        
        x_min, y_min = point_cloud_range[0], point_cloud_range[1]
        x_max, y_max = point_cloud_range[3], point_cloud_range[4]
        
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        intensities = points[:, 3] if points.shape[1] > 3 else torch.ones_like(x_coords)
        
        # 범위 내 점들만 선택
        valid_mask = (x_coords >= x_min) & (x_coords <= x_max) & \
                    (y_coords >= y_min) & (y_coords <= y_max)
        
        if not valid_mask.any():
            return torch.zeros(1, bev_size, bev_size)
        
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]
        intensities = intensities[valid_mask]
        
        # 픽셀 좌표로 변환
        x_img = ((x_coords - x_min) / (x_max - x_min) * (bev_size - 1)).long()
        y_img = ((y_coords - y_min) / (y_max - y_min) * (bev_size - 1)).long()
        
        # 범위 체크
        valid_pixels = (x_img >= 0) & (x_img < bev_size) & \
                      (y_img >= 0) & (y_img < bev_size)
        
        x_img = x_img[valid_pixels]
        y_img = y_img[valid_pixels]
        intensities = intensities[valid_pixels]
        
        # BEV 생성
        bev = torch.zeros(bev_size, bev_size, device=points.device)
        if len(x_img) > 0:
            bev[y_img, x_img] = intensities
        
        return bev.unsqueeze(0)  # [1, H, W]
    
    def encoder(self, x):
        """더미 인코더 (실제로는 간단한 변환만 수행)"""
        batch_size = x.shape[0]
        
        # 더미 latent 생성 (실제 UltraLiDAR 구조에 맞춤)
        # 640 -> 80 (8x downsampling)
        latent_h, latent_w = 80, 80
        latent_dim = 1024  # UltraLiDAR codebook_dim
        
        # 간단한 downsampling (실제로는 학습된 인코더를 사용해야 함)
        x_downsampled = nn.functional.avg_pool2d(x, kernel_size=8, stride=8)  # [B, C, 80, 80]
        
        # 채널 차원을 1024로 확장
        if x_downsampled.shape[1] != latent_dim:
            # 간단한 linear projection (실제로는 학습된 가중치 사용)
            latents = x_downsampled.repeat(1, latent_dim, 1, 1)[:, :latent_dim]
        else:
            latents = x_downsampled
        
        return latents
    
    def quantize(self, latents):
        """더미 양자화 (코드북 인덱스 생성)"""
        batch_size, channels, h, w = latents.shape
        
        # 더미 양자화 - 실제로는 코드북과의 거리 계산 필요
        # 여기서는 간단히 argmax 사용
        latents_flat = latents.view(batch_size, channels, -1)  # [B, C, H*W]
        
        # 더미 코드북 인덱스 (0~1023 범위)
        code_indices = torch.randint(0, 1024, (batch_size, h * w), device=latents.device)
        
        # 더미 양자화된 latents
        quant_latents = latents  # 실제로는 코드북에서 임베딩 조회
        
        # 더미 임베딩 손실
        emb_loss = torch.tensor(0.0, device=latents.device)
        
        stats = {"min_encoding_indices": code_indices}
        
        return quant_latents, emb_loss, stats
    
    def generator(self, quantized_latents):
        """더미 생성기 (실제로는 학습된 디코더 사용)"""
        batch_size = quantized_latents.shape[0]
        
        # 간단한 upsampling (80 -> 640)
        upsampled = nn.functional.interpolate(
            quantized_latents, 
            size=(640, 640), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 단일 채널로 변환
        if upsampled.shape[1] > 1:
            generated = upsampled.mean(dim=1, keepdim=True)  # [B, 1, 640, 640]
        else:
            generated = upsampled
        
        # Sigmoid 적용
        generated = torch.sigmoid(generated)
        
        return generated
    
    def get_embedding_weight(self):
        """임베딩 가중치 추출"""
        # UltraLiDAR 체크포인트에서 임베딩 가중치 찾기
        for key, value in self.state_dict_full.items():
            if 'embedding.weight' in key and 'vector_quantizer' in key:
                print(f"Found embedding weight: {key}, shape: {value.shape}")
                return value
        
        # 찾지 못한 경우 더미 가중치 생성
        print("Warning: Could not find embedding weight. Using random initialization.")
        return torch.randn(1024, 1024)  # [codebook_size, emb_dim]
    
    def forward(self, x):
        """전체 순전파"""
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


def load_simple_ultralidar_model(checkpoint_path, model_type='radar'):
    """간단한 UltraLiDAR 모델 로더"""
    try:
        model = SimpleUltraLiDARWrapper(checkpoint_path, model_type)
        print(f"✅ Simple UltraLiDAR {model_type} model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Failed to load simple UltraLiDAR {model_type} model: {e}")
        raise e


# 호환성을 위한 별칭
UltraLiDARRadarWrapper = SimpleUltraLiDARWrapper
load_ultralidar_radar_model = load_simple_ultralidar_model
load_ultralidar_lidar_model = lambda path, config=None: load_simple_ultralidar_model(path, 'lidar')
