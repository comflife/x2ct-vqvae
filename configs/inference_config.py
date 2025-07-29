import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    
    # Run configuration
    config.run = ml_collections.ConfigDict()
    config.run.name = 'radar_to_lidar_inference'
    config.run.experiment = 'inference'
    config.run.use_cuda = True
    config.run.gpu_id = 1
    
    # Model configuration (학습 시 사용된 설정과 일치시킴)
    config.model = ml_collections.ConfigDict()
    config.model.name = 'absorbing'  # 학습에서 사용한 sampler 타입
    config.model.latent_shape = [40, 40]  # lidar config와 동일
    config.model.codebook_size = 1024  # lidar config와 동일
    config.model.emb_dim = 1024  # lidar config와 동일
    config.model.n_emb = 512  # 학습 시 사용된 값 (512)
    config.model.block_size = 4096  # 학습 시 사용된 값 (4096)
    
    # Transformer 설정들을 학습 시 사용된 값으로 수정
    config.model.n_layers = 12  # 학습 시 사용된 값 (12)
    config.model.n_head = 8   # 학습 시 사용된 값 (8)
    config.model.n_embd = 512  # 학습 시 사용된 값 (512)
    config.model.attn_pdrop = 0.1
    config.model.resid_pdrop = 0.1
    config.model.embd_pdrop = 0.1
    
    # Model paths
    config.model_paths = ml_collections.ConfigDict()
    config.model_paths.radar_vqgan_path = '/home/byounggun/r2l/UltraLiDAR_nusc_waymo/work_dirs/nusc_radar_debug/epoch_121.pth'
    config.model_paths.lidar_vqgan_path = '/home/byounggun/r2l/UltraLiDAR_nusc_waymo/work_dirs/nusc_stage2/epoch_200.pth'
    
    # Data configuration
    config.data = ml_collections.ConfigDict()
    config.data.data_root = '/data1/nuScenes/'
    config.data.val_ann_file = 'nuscenes_infos_train_radar.pkl'  # data_root 기준 상대경로로 수정
    
    # Diffusion sampling configuration
    config.diffusion = ml_collections.ConfigDict()
    config.diffusion.sampling_steps = 100
    config.diffusion.sampling_temp = 1.0
    config.diffusion.sampling_batch_size = 1  # inference에서는 1로 설정
    config.diffusion.loss_type = 'reweighted_elbo'  # 학습에서 사용한 loss type
    config.diffusion.time_sampling = 'uniform'  # 누락된 필드 추가
    config.diffusion.mask_schedule = 'random'  # 누락된 필드 추가
    config.diffusion.flash_attn = True
    
    # 추가로 필요할 수 있는 absorbing diffusion 설정들
    config.diffusion.num_timesteps = 100  # diffusion timesteps
    config.diffusion.schedule = 'linear'  # noise schedule
    config.diffusion.beta_start = 0.0001
    config.diffusion.beta_end = 0.02
    config.diffusion.mask_token_id = 1024  # mask token ID (보통 codebook_size와 같음)
    config.diffusion.absorb_type = 'uniform'  # absorbing type
    
    return config