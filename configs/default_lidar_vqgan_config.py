import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    
    # Model configuration for UltraLiDAR lidar VQGAN
    config.model = ml_collections.ConfigDict()
    config.model.name = 'ultralidar_lidar_vqgan'
    config.model.codebook_size = 1024  # UltraLiDAR uses 1024 codebook size
    config.model.emb_dim = 1024  # UltraLiDAR embedding dimension - IMPORTANT!
    config.model.n_hiddens = 240
    config.model.n_res_layers = 4
    config.model.downsample = [4, 4]  # Downsample factor for BEV
    config.model.latent_shape = [40, 40]  # 320/8 = 40 for each dimension
    
    # Data configuration
    config.data = ml_collections.ConfigDict()
    config.data.img_size = [320, 320]  # 메모리 절약을 위해 320x320으로 축소
    config.data.channels = 1  # Single channel for lidar BEV
    
    # Training configuration (not used for inference)
    config.train = ml_collections.ConfigDict()
    config.train.learning_rate = 4.5e-06
    config.train.beta1 = 0.5
    config.train.beta2 = 0.9
    config.train.epochs = 100
    config.train.batch_size = 8
    
    # Loss configuration
    config.loss = ml_collections.ConfigDict()
    config.loss.commitment_cost = 0.25
    config.loss.perceptual_weight = 1.0
    config.loss.gan_weight = 1.0
    
    return config
