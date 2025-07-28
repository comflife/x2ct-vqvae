#!/bin/bash

# Radar to Lidar BEV Generation Training Script

# 설정 파일들 생성 (필요한 경우)
echo "Setting up configuration files..."

# Radar VQGAN 설정 파일 (UltraLiDAR 구조)
cat > configs/default_radar_vqgan_config.py << 'EOF'
from ml_collections import ConfigDict

def get_config():
    config = ConfigDict()
    
    config.run = ConfigDict()
    config.run.name = 'ultralidar_radar'
    config.run.experiment = 'radar_bev'
    
    config.data = ConfigDict()
    config.data.channels = 1  # Radar BEV는 단일 채널 (강도)
    config.data.img_size = 640  # UltraLiDAR에서 사용하는 크기
    
    config.model = ConfigDict()
    # UltraLiDAR VQEncoder/VQDecoder 구조에 맞춤
    config.model.emb_dim = 1024  # UltraLiDAR codebook_dim
    config.model.nf = 64
    config.model.ch_mult = [1, 1, 2, 2, 4]
    config.model.res_blocks = 2
    config.model.attn_resolutions = [16, 32]
    config.model.codebook_size = 1024  # UltraLiDAR n_e
    config.model.latent_shape = [1, 80, 80]  # 640/8 = 80 (일반적인 다운샘플링)
    config.model.sampler_load_step = 121  # epoch_121.pth
    
    # UltraLiDAR 특화 설정
    config.ultralidar = ConfigDict()
    config.ultralidar.voxel_size = [0.15625, 0.15625, 0.2]
    config.ultralidar.point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
    config.ultralidar.img_size = 640
    config.ultralidar.num_patches = 6400  # 80*80
    
    return config
EOF

# Lidar VQGAN 설정 파일 (UltraLiDAR 구조)
cat > configs/default_lidar_vqgan_config.py << 'EOF'
from ml_collections import ConfigDict

def get_config():
    config = ConfigDict()
    
    config.run = ConfigDict()
    config.run.name = 'ultralidar_lidar'
    config.run.experiment = 'lidar_bev'
    
    config.data = ConfigDict()
    config.data.channels = 1  # Lidar BEV도 단일 채널 (강도)
    config.data.img_size = 640  # UltraLiDAR에서 사용하는 크기
    
    config.model = ConfigDict()
    # UltraLiDAR VQEncoder/VQDecoder 구조에 맞춤
    config.model.emb_dim = 1024  # UltraLiDAR codebook_dim
    config.model.nf = 64
    config.model.ch_mult = [1, 1, 2, 2, 4]
    config.model.res_blocks = 2
    config.model.attn_resolutions = [16, 32]
    config.model.codebook_size = 1024  # UltraLiDAR n_e
    config.model.latent_shape = [1, 80, 80]  # 640/8 = 80 (일반적인 다운샘플링)
    config.model.sampler_load_step = 200  # epoch_200.pth
    
    # UltraLiDAR 특화 설정
    config.ultralidar = ConfigDict()
    config.ultralidar.voxel_size = [0.15625, 0.15625, 0.2]
    config.ultralidar.point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
    config.ultralidar.img_size = 640
    config.ultralidar.num_patches = 6400  # 80*80
    
    return config
EOF

echo "Configuration files created!"

# 훈련 실행
echo "Starting Radar to Lidar BEV training with UltraLiDAR..."

# UltraLiDAR 경로 설정
export ULTRALIDAR_PATH="/home/byounggun/r2l/UltraLiDAR_nusc_waymo"
export PYTHONPATH="$ULTRALIDAR_PATH:$PYTHONPATH"

python train_radar_to_lidar_sampler.py \
    --config=configs/radar_to_lidar_sampler_config.py \
    --radar_config=configs/default_radar_vqgan_config.py \
    --lidar_config=configs/default_lidar_vqgan_config.py

echo "Training completed!"
