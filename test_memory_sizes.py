#!/usr/bin/env python3
"""
간단한 해결책 테스트: 더 작은 latent space 사용
"""

import torch
import sys
sys.path.append('/home/byounggun/r2l/x2ct-vqvae')

from utils.simple_ultralidar_wrapper import load_simple_ultralidar_model

def test_smaller_latent_space():
    """더 작은 latent space로 메모리 사용량 테스트"""
    print("Testing smaller latent space...")
    
    device = torch.device('cuda:1')
    
    # UltraLiDAR 모델 로드
    radar_path = "/home/byounggun/r2l/UltraLiDAR_nusc_waymo/work_dirs/nusc_radar_debug/epoch_121.pth"
    lidar_path = "/home/byounggun/r2l/UltraLiDAR_nusc_waymo/work_dirs/nusc_stage2/epoch_200.pth"
    
    ae_radar = load_simple_ultralidar_model(radar_path, 'radar').to(device)
    ae_lidar = load_simple_ultralidar_model(lidar_path, 'lidar').to(device)
    
    # 다양한 입력 크기 테스트
    test_sizes = [
        (320, 320),  # 1/2 크기
        (256, 256),  # 더 작은 크기  
        (160, 160),  # 1/4 크기
    ]
    
    for h, w in test_sizes:
        print(f"\n=== Testing {h}x{w} input ===")
        dummy_bev = torch.randn(1, 1, h, w).to(device)
        
        try:
            with torch.no_grad():
                # Radar 인코딩
                radar_latents = ae_radar.encoder(dummy_bev)
                radar_quant, _, _ = ae_radar.quantize(radar_latents)
                
                # 올바른 reshape
                B, C, H, W = radar_quant.shape
                radar_context = radar_quant.permute(0, 2, 3, 1).contiguous().view(B, H*W, C)
                
                # Lidar 인코딩
                lidar_latents = ae_lidar.encoder(dummy_bev)
                _, _, lidar_info = ae_lidar.quantize(lidar_latents)
                lidar_codes = lidar_info['min_encoding_indices'].view(B, -1)
                
                print(f"Input: {dummy_bev.shape}")
                print(f"Radar context: {radar_context.shape}")
                print(f"Lidar codes: {lidar_codes.shape}")
                print(f"Total sequence length: {radar_context.shape[1] + lidar_codes.shape[1]}")
                
                # 메모리 사용량 확인
                allocated = torch.cuda.memory_allocated(device) / 1024**3
                print(f"GPU memory: {allocated:.2f} GB")
                
        except Exception as e:
            print(f"Error with {h}x{w}: {e}")
        
        # 메모리 정리
        torch.cuda.empty_cache()

if __name__ == "__main__":
    test_smaller_latent_space()
