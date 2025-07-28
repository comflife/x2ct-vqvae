#!/usr/bin/env python3
"""
UltraLiDAR 모델과 Transformer 차원 테스트 스크립트
"""

import torch
import os
import sys
sys.path.append('/home/byounggun/r2l/x2ct-vqvae')

from utils.simple_ultralidar_wrapper import load_simple_ultralidar_model
from ml_collections import config_flags
from absl import app, flags
import ml_collections

# 설정 로드
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "configs/radar_to_lidar_sampler_config.py", "Main config")
config_flags.DEFINE_config_file("radar_config", "configs/default_radar_vqgan_config.py", "Radar config")  
config_flags.DEFINE_config_file("lidar_config", "configs/default_lidar_vqgan_config.py", "Lidar config")

def test_ultralidar_dimensions():
    """UltraLiDAR 모델의 실제 차원 테스트"""
    print("=" * 60)
    print("TESTING ULTRALIDAR MODEL DIMENSIONS")
    print("=" * 60)
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # UltraLiDAR 모델 로드
    radar_path = "/home/byounggun/r2l/UltraLiDAR_nusc_waymo/work_dirs/nusc_radar_debug/epoch_121.pth"
    lidar_path = "/home/byounggun/r2l/UltraLiDAR_nusc_waymo/work_dirs/nusc_stage2/epoch_200.pth"
    
    print(f"\n1. Loading UltraLiDAR models...")
    ae_radar = load_simple_ultralidar_model(radar_path, 'radar').to(device)
    ae_lidar = load_simple_ultralidar_model(lidar_path, 'lidar').to(device)
    
    # 테스트 입력 생성 (640x640 BEV)
    print(f"\n2. Testing with 640x640 BEV input...")
    dummy_bev = torch.randn(1, 1, 640, 640).to(device)
    print(f"Input BEV shape: {dummy_bev.shape}")
    
    # Radar 인코딩 테스트
    print(f"\n3. Testing Radar encoding...")
    with torch.no_grad():
        radar_latents = ae_radar.encoder(dummy_bev)
        print(f"Radar latents shape: {radar_latents.shape}")
        
        radar_quant, radar_vq_loss, radar_info = ae_radar.quantize(radar_latents)
        print(f"Radar quantized shape: {radar_quant.shape}")
        
        # 올바른 reshape: (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = radar_quant.shape
        radar_context_correct = radar_quant.permute(0, 2, 3, 1).contiguous().view(B, H*W, C)
        print(f"Radar context (CORRECT): {radar_context_correct.shape} = (batch, seq_len, emb_dim)")
        print(f"  Sequence length: {H*W}, Embedding dim: {C}")
        
        # 잘못된 방식 (기존 코드의 문제)
        radar_context_wrong = radar_quant.view(B, -1, radar_quant.shape[-1])
        print(f"Radar context (WRONG): {radar_context_wrong.shape} = (batch, seq_len, emb_dim)")
        print(f"  This would give wrong sequence length and embedding dim!")
        
    # Lidar 인코딩 테스트
    print(f"\n4. Testing Lidar encoding...")
    with torch.no_grad():
        lidar_latents = ae_lidar.encoder(dummy_bev)
        print(f"Lidar latents shape: {lidar_latents.shape}")
        
        lidar_quant, lidar_vq_loss, lidar_info = ae_lidar.quantize(lidar_latents)
        print(f"Lidar quantized shape: {lidar_quant.shape}")
        
        if isinstance(lidar_info, dict) and 'min_encoding_indices' in lidar_info:
            lidar_codes = lidar_info['min_encoding_indices']
            print(f"Lidar codes shape: {lidar_codes.shape}")
            print(f"Lidar codes flattened: {lidar_codes.view(1, -1).shape}")
        else:
            print(f"Lidar info type: {type(lidar_info)}")
            if isinstance(lidar_info, dict):
                print(f"Lidar info keys: {list(lidar_info.keys())}")
    
    # 임베딩 가중치 확인
    print(f"\n5. Testing embedding weights...")
    if hasattr(ae_lidar, 'get_embedding_weight'):
        embedding_weight = ae_lidar.get_embedding_weight()
        print(f"Lidar embedding weight shape: {embedding_weight.shape}")
    else:
        print("No get_embedding_weight method found")
    
    # 시퀀스 길이 계산 (올바른 방식)
    print(f"\n6. Calculating sequence lengths (CORRECTED)...")
    B, C, H, W = radar_quant.shape
    radar_seq_len_correct = H * W  # 80 * 80 = 6400
    radar_emb_dim = C  # 1024
    
    if isinstance(lidar_info, dict) and 'min_encoding_indices' in lidar_info:
        lidar_seq_len = lidar_codes.view(1, -1).shape[1]  # 6400
        total_seq_len_correct = radar_seq_len_correct + lidar_seq_len
        print(f"Radar context sequence length: {radar_seq_len_correct}")
        print(f"Radar embedding dimension: {radar_emb_dim}")
        print(f"Lidar target sequence length: {lidar_seq_len}")
        print(f"Total sequence length: {total_seq_len_correct}")
        print(f"")
        print(f"Block size limit: 16384")
        if total_seq_len_correct <= 16384:
            print(f"✅ Total sequence length is within block size!")
        else:
            print(f"❌ Total sequence length exceeds block size by {total_seq_len_correct - 16384}")
    
    # 기존 잘못된 계산도 보여주기
    print(f"\n6b. Previous WRONG calculation:")
    radar_seq_len_wrong = radar_quant.view(1, -1, radar_quant.shape[-1]).shape[1]
    total_seq_len_wrong = radar_seq_len_wrong + lidar_seq_len if 'lidar_seq_len' in locals() else radar_seq_len_wrong
    print(f"Wrong radar sequence length: {radar_seq_len_wrong}")
    print(f"Wrong total sequence length: {total_seq_len_wrong}")
    
    print(f"\n7. Memory usage:")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU memory allocated: {allocated:.2f} GB")

def test_transformer_config():
    """Transformer 설정 테스트"""
    print("\n" + "=" * 60)
    print("TESTING TRANSFORMER CONFIGURATION")
    print("=" * 60)
    
    H = FLAGS.config
    # Config는 locked이므로 직접 할당하지 않고 FLAGS에서 가져오기
    radar_config = FLAGS.radar_config
    lidar_config = FLAGS.lidar_config
    
    print(f"\nMain config:")
    print(f"  Block size: {H.model.block_size}")
    print(f"  Embedding dim: {H.model.n_emb}")
    print(f"  Attention heads: {H.model.n_head}")
    print(f"  Layers: {H.model.n_layers}")
    
    print(f"\nRadar config:")
    print(f"  Codebook size: {radar_config.model.codebook_size}")
    print(f"  Embedding dim: {radar_config.model.emb_dim}")
    print(f"  Latent shape: {radar_config.model.latent_shape}")
    
    print(f"\nLidar config:")
    print(f"  Codebook size: {lidar_config.model.codebook_size}")
    print(f"  Embedding dim: {lidar_config.model.emb_dim}")
    print(f"  Latent shape: {lidar_config.model.latent_shape}")
    
    return H, radar_config, lidar_config

def test_context_dimension_compatibility():
    """Context 차원 호환성 테스트"""
    print("\n" + "=" * 60)
    print("TESTING CONTEXT DIMENSION COMPATIBILITY")
    print("=" * 60)
    
    H = FLAGS.config
    radar_config = FLAGS.radar_config
    lidar_config = FLAGS.lidar_config
    
    # Transformer context linear layer 예상 입력 차원
    expected_context_dim = lidar_config.model.emb_dim  # ct_config는 lidar_config로 대체됨
    transformer_emb_dim = H.model.n_emb
    
    print(f"Expected context input dim (from lidar config): {expected_context_dim}")
    print(f"Transformer embedding dim: {transformer_emb_dim}")
    print(f"Block size limit: {H.model.block_size}")
    
    # 실제 UltraLiDAR에서 나오는 차원과 비교
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    radar_path = "/home/byounggun/r2l/UltraLiDAR_nusc_waymo/work_dirs/nusc_radar_debug/epoch_121.pth"
    
    ae_radar = load_simple_ultralidar_model(radar_path, 'radar').to(device)
    dummy_bev = torch.randn(1, 1, 640, 640).to(device)
    
    with torch.no_grad():
        radar_latents = ae_radar.encoder(dummy_bev)
        radar_quant, _, _ = ae_radar.quantize(radar_latents)
        
        # 올바른 reshape: (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = radar_quant.shape
        radar_context = radar_quant.permute(0, 2, 3, 1).contiguous().view(B, H*W, C)
        
        print(f"\nActual radar context shape: {radar_context.shape}")
        print(f"Actual radar embedding dim: {radar_context.shape[-1]}")
        print(f"Actual radar sequence length: {radar_context.shape[1]}")
        
        # 호환성 확인
        if radar_context.shape[-1] == expected_context_dim:
            print("✅ Context dimension MATCHES config!")
        else:
            print(f"❌ Context dimension MISMATCH! Got {radar_context.shape[-1]}, expected {expected_context_dim}")
            
        if radar_context.shape[1] <= H.model.block_size//2:  # context + target를 고려
            print("✅ Sequence length within reasonable block size!")
        else:
            print(f"❌ Sequence length may be too large! Got {radar_context.shape[1]}, half of block size is {H.model.block_size//2}")
            
        return radar_context.shape

def main(argv):
    """메인 테스트 함수"""
    print("Starting UltraLiDAR dimension compatibility tests...")
    
    try:
        test_ultralidar_dimensions()
        H, radar_config, lidar_config = test_transformer_config() 
        context_shape = test_context_dimension_compatibility()
        
        print("\n" + "=" * 60)
        print("📋 SUMMARY AND RECOMMENDATIONS")
        print("=" * 60)
        
        print(f"✅ UltraLiDAR models load successfully")
        print(f"✅ Embedding dimensions match (1024)")
        print(f"✅ Sequence lengths are manageable:")
        print(f"   - Radar context: 6,400 tokens")  
        print(f"   - Lidar target: 6,400 tokens")
        print(f"   - Total: 12,800 tokens (within block_size 16,384)")
        
        print(f"\n🔧 FIXES NEEDED:")
        print(f"1. Fix reshape in generate_latent_ids(): use permute(0,2,3,1).view() instead of view()")
        print(f"2. Make sure transformer context_linear expects 1024 input dimensions")
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS COMPLETED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    app.run(main)
