from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference

def get_config():
    config = ConfigDict()

    #######################################################################
    ############################# RUN CONFIG ##############################
    #######################################################################
    config.run = run = ConfigDict()
    # Name for set of experiments in wandb
    run.name = 'radar_to_lidar_sampler'
    # Creates a separate log subfolder for each experiment
    run.experiment = 'radar_lidar'
    run.wandb_dir = 'online'
    # Set this to 'disabled' to disable wandb logging
    run.wandb_mode = 'online'
    # Enables logging to visdom
    run.enable_visdom = False
    run.visdom_server = 'http://localhost'
    run.visdom_port = 8097
    run.log_to_file = False
    # GPU 설정
    run.gpu_id = 0  # 사용할 GPU 번호 (0,1,2,3 중 선택)
    run.use_cuda = True

    #######################################################################
    ############################# DATA CONFIG #############################
    #######################################################################
    config.data = data = ConfigDict()
    # NuScenes 데이터 경로
    data.data_root = "/data1/nuScenes/"  # NuScenes 데이터 루트 경로
    data.data_dir = "/data1/nuScenes/"  # 호환성을 위한 별칭
    data.img_size = FieldReference(320)  # 메모리와 성능의 균형을 위해 320x320 사용
    data.num_radar_views = 1  # radar는 단일 뷰
    data.channels = 1  # Radar BEV 채널 수
    data.load_res = None
    data.dataset = 'nuscenes_radar_lidar'
    data.cupy = False
    data.use_synthetic = False
    data.loader = "nuscenes_radar_lidar"  # NuScenes radar+lidar 데이터로더 사용
    # 실제 존재하는 pkl 파일들 (tiny 버전 사용)
    data.train_ann_file = "nuscenes_infos_val_radar_tiny.pkl"  # 우선 작은 데이터로 테스트
    data.val_ann_file = "nuscenes_infos_val_radar_tiny.pkl"

    #######################################################################
    ########################### TRAINING CONFIG ###########################
    #######################################################################
    config.train = train = ConfigDict()
    train.amp = True
    train.batch_size = FieldReference(1)  # 메모리 절약을 위해 배치 크기 줄임
    # How often to plot new loss values to graphs
    train.plot_graph_steps = 100
    # How often to plot reconstruction images
    train.plot_recon_steps = 500
    # How often to evaluate on test set
    train.eval_steps = 5000
    # How often to save checkpoints
    train.checkpoint_steps = 5000
    # How often to update ema model params
    train.ema_update_every = 10
    train.ema_decay = 0.995
    # What model step to load
    train.load_step = 0
    # Number of times to repeat evaluation
    train.eval_repeats = 10
    train.use_context = True
    train.total_steps = 100000

    #######################################################################
    ############################# MODEL CONFIG ############################
    #######################################################################
    config.model = model = ConfigDict()
    # Name of architecture. Currently in ['absorbing', 'autoregressive'].
    model.name = "absorbing"
    # Network width
    model.n_emb = 512
    # Number of attention heads
    model.n_head = 8
    # Number of layers
    model.n_layers = 12
    # Max input size to initialise positional embeddings etc at
    # 320x320 BEV: Radar context (1600) + Lidar target (1600) = 3200 토큰
    model.block_size = 4096  # 여유를 위해 4096으로 설정
    # Dropout params
    model.attn_pdrop = 0.1
    model.embd_pdrop = 0.1
    model.resid_pdrop = 0.1
    model.load_step = 0

    config.diffusion = diffusion = ConfigDict()
    diffusion.loss_type = "reweighted_elbo"
    diffusion.mask_schedule = "random"
    diffusion.time_sampling = "uniform"
    # Number of steps to sample with
    diffusion.sampling_steps = 1024  # lidar BEV 해상도에 맞게 조정
    # Temperature to sample diffusion with
    diffusion.sampling_temp = 0.8
    # Batch size for sampling
    diffusion.sampling_batch_size = 2
    diffusion.flash_attn = True

    #######################################################################
    ########################### OPTIMIZER CONFIG ##########################
    #######################################################################
    config.optimizer = optimizer = ConfigDict()
    optimizer.learning_rate = 1e-4
    optimizer.warmup_steps = 5000
    optimizer.weight_decay = 0.1

    #######################################################################
    ########################### MODEL PATHS CONFIG ########################
    #######################################################################
    config.model_paths = model_paths = ConfigDict()
    # Radar VQGAN 모델 경로 (UltraLiDAR radar model)
    model_paths.radar_vqgan_path = "/home/byounggun/r2l/UltraLiDAR_nusc_waymo/work_dirs/nusc_radar_debug/epoch_121.pth"
    # Lidar VQGAN 모델 경로 (UltraLiDAR lidar model)
    model_paths.lidar_vqgan_path = "/home/byounggun/r2l/UltraLiDAR_nusc_waymo/work_dirs/nusc_stage2/epoch_200.pth"

    return config
