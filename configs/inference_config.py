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
    config.model.name = "absorbing"
    config.model.n_emb = 256
    config.model.n_head = 4
    config.model.n_layers = 6
    config.model.block_size = 16384
    config.model.attn_pdrop = 0.1
    config.model.embd_pdrop = 0.1
    config.model.resid_pdrop = 0.1
    
    # Model paths
    config.model_paths = ml_collections.ConfigDict()
    config.model_paths.radar_vqgan_path = '/home/byounggun/r2l/UltraLiDAR_nusc_waymo/work_dirs/nusc_radar_debug/epoch_121.pth'
    config.model_paths.lidar_vqgan_path = '/home/byounggun/r2l/UltraLiDAR_nusc_waymo/work_dirs/nusc_stage2/epoch_200.pth'
    
    # Data configuration
    config.data = ml_collections.ConfigDict()
    config.data.data_root = '/data1/nuScenes/'
    config.data.radar_val_ann_file = 'nuscenes_infos_val_radar.pkl'  # val로 수정
    config.data.lidar_val_ann_file = 'nuscenes_infos_lidar_val.pkl'  # val로 수정
    config.data.img_size = 640
    
    # Diffusion sampling configuration
    config.diffusion = ml_collections.ConfigDict()
    config.diffusion.loss_type = "reweighted_elbo"
    config.diffusion.mask_schedule = "random"
    config.diffusion.time_sampling = "uniform"
    config.diffusion.sampling_steps = 1024
    config.diffusion.sampling_temp = 0.8
    config.diffusion.sampling_batch_size = 2
    config.diffusion.flash_attn = True

    return config