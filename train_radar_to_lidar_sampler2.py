import torch
from torch.utils.data import DataLoader
from einops import rearrange, repeat
import numpy as np

import wandb
import visdom
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import time
import copy
import os
from collections import defaultdict
from tqdm import tqdm

# 기존 모델들
from models.vqgan_2d import VQAutoEncoder as VQAutoEncoder2D, Generator as Generator2D
from models.vqgan_3d import VQAutoEncoder as VQAutoEncoder3D, Generator as Generator3D
# 새로운 데이터로더
from utils.radar_lidar_dataloader import RadarLidarBEV_dataset, RadarLidarBEV_dataset_simple
from utils.nuscenes_radar_dataloader import NuScenesRadarDataset, NuScenesRadarLidarDataset
# UltraLiDAR wrapper (간단한 버전 - mmcv 의존성 없음)
from utils.simple_ultralidar_wrapper import SimpleUltraLiDARWrapper, load_simple_ultralidar_model
from utils.sampler_utils import get_latent_loaders, get_sampler, latent_ids_to_onehot
from utils.log_utils import log, flatten_collection, track_variables, log_stats, plot_images, save_model, config_log, load_model
from utils.train_utils import optim_warmup, update_ema

# mmcv 및 mmdet3d 임포트 추가
try:
    from mmcv.utils import Config
except ImportError:
    from mmcv.utils.config import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmcv.runner import load_checkpoint

# UltraLiDAR 플러그인 임포트 (모델 등록을 위해)
import sys
sys.path.append('/home/byounggun/r2l/UltraLiDAR_nusc_waymo')
import plugin

# Commandline arguments
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
config_flags.DEFINE_config_file("radar_config", "configs/default_radar_vqgan_config.py", "Radar VQGAN training configuration.", lock_config=True)
config_flags.DEFINE_config_file("lidar_config", "configs/default_lidar_vqgan_config.py", "Lidar VQGAN training configuration.", lock_config=True)
flags.DEFINE_string("radar_ultralidar_config", "configs/ultralidar_nusc_radar_debug.py", "UltraLiDAR radar config path")
flags.DEFINE_string("lidar_ultralidar_config", "configs/ultralidar_nusc.py", "UltraLiDAR lidar config path")
flags.mark_flags_as_required(["config"])

# Torch options
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def setup_device(config):
    """GPU 설정 및 디바이스 선택"""
    if hasattr(config.run, 'use_cuda') and config.run.use_cuda and torch.cuda.is_available():
        if hasattr(config.run, 'gpu_id'):
            gpu_id = config.run.gpu_id
            if gpu_id < torch.cuda.device_count():
                torch.cuda.set_device(gpu_id)
                device = torch.device(f'cuda:{gpu_id}')
                log(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
                log(f"GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f} GB")
            else:
                device = torch.device('cuda')
                log(f"GPU {gpu_id} not available, using default GPU")
        else:
            device = torch.device('cuda')
            log("Using default CUDA device")
    else:
        device = torch.device('cpu')
        log("Using CPU")
    
    return device

def log_gpu_memory():
    """GPU 메모리 사용량 로그"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        total_memory = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1024**3
        usage_percent = (allocated / total_memory) * 100
        log(f"GPU Memory - Allocated: {allocated:.2f}GB ({usage_percent:.1f}%), Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB, Total: {total_memory:.1f}GB")

# Device will be set in main()
device = None

def update_model_weights(optim, loss, amp=False, scaler=None):
    optim.zero_grad()
    if amp:
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
    else:
        loss.backward()
        optim.step()

def reconstruct_from_codes(H, sampler, x, generator):
    """코드로부터 이미지 재구성"""
    latents_one_hot = latent_ids_to_onehot(x, H.lidar_config.model.latent_shape, H.lidar_config.model.codebook_size)
    q = sampler.embed(latents_one_hot)
    images = generator(q.float())
    return images

def load_pretrained_vqgan(model_path, config_path, device, model_type='radar'):
    """사전 훈련된 UltraLiDAR 모델 로드 (full mmdet3d loading)"""
    cfg = Config.fromfile(config_path)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if os.path.exists(model_path):
        log(f"Loading checkpoint from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    log(f"Loaded full UltraLiDAR {model_type} model from {model_path} using config {config_path}")
    return model

def extract_points_from_data(data, H=None):
    """mmdet3d 데이터에서 points 추출"""
    import os
    
    # 이미 로드된 points 데이터가 있으면 우선 사용
    if 'points' in data:
        points = data['points']
        
        # mmdet3d의 points 객체 처리
        if hasattr(points, 'data'):
            points = points.data
        elif hasattr(points, 'tensor'):
            points = points.tensor
        
        # 리스트인 경우 첫 번째 요소 사용
        if isinstance(points, list) and len(points) > 0:
            points = points[0]
            
        # numpy array를 tensor로 변환
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
            
        log(f"[extract_points_from_data] Successfully extracted points from 'points' key, shape: {points.shape}")
        return points
    
    # fallback: pts_filename이 있으면 파일에서 직접 로드
    if 'pts_filename' in data:
        pts_filename = data['pts_filename']
        # H.data.data_root 사용
        if H and hasattr(H, 'data') and hasattr(H.data, 'data_root'):
            data_root = H.data.data_root
        else:
            data_root = '/data1/nuScenes/'  # fallback
            
        log(f"[extract_points_from_data] data_root: {data_root}")
        log(f"[extract_points_from_data] pts_filename(raw): {pts_filename}")
        
        # '/'로 시작하지 않으면 무조건 prepend
        if not str(pts_filename).startswith('/'):
            pts_filename_full = os.path.join(data_root, pts_filename)
        else:
            pts_filename_full = pts_filename
            
        log(f"[extract_points_from_data] Try loading: {pts_filename_full}")
        
        # 파일을 직접 읽어서 points로 변환
        if os.path.exists(pts_filename_full):
            points = np.fromfile(pts_filename_full, dtype=np.float32)
            # points shape 보정 필요시 추가 (예: reshape)
            if len(points) % 5 == 0:  # radar points: x, y, z, rcs, v_comp
                points = points.reshape(-1, 5)
            elif len(points) % 6 == 0:  # lidar points: x, y, z, intensity, ring, time
                points = points.reshape(-1, 6)
            points = torch.tensor(points, dtype=torch.float32)
            log(f"[extract_points_from_data] Successfully loaded from file, shape: {points.shape}")
            return points
        else:
            log(f"[extract_points_from_data] File not found: {pts_filename_full}")
            return None
    
    log(f"[extract_points_from_data] No valid points data found in keys: {data.keys()}")
    return None

def points_to_bev(points, bev_size=320, range_limit=50.0):
    """Points를 BEV 이미지로 변환 (radar 시각화 코드와 동일한 방식)"""
    if points.dim() == 3:
        points = points[0]  # (N, 6)
    
    points_np = points.cpu().numpy()
    
    # bev_size가 리스트인 경우 처리
    if isinstance(bev_size, (list, tuple)):
        if len(bev_size) == 2:
            bev_h, bev_w = bev_size
        else:
            bev_h = bev_w = bev_size[0]
    else:
        bev_h = bev_w = bev_size
    
    # BEV 이미지 생성
    bev_image = np.zeros((bev_h, bev_w), dtype=np.float32)
    
    # 좌표 변환: [-50, 50] → [0, bev_size]
    x_coords = points_np[:, 0]
    y_coords = points_np[:, 1]
    
    # RCS 값 또는 intensity 값 사용
    if points_np.shape[1] >= 4:
        intensity_values = points_np[:, 3]  # RCS 값
    else:
        intensity_values = np.ones(len(points_np))  # 기본값
    
    # 범위 내 점들만 선택
    valid_mask = (np.abs(x_coords) <= range_limit) & (np.abs(y_coords) <= range_limit)
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]
    intensity_values = intensity_values[valid_mask]
    
    if len(x_coords) == 0:
        return torch.tensor(bev_image).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    # 픽셀 좌표로 변환
    x_pixels = ((x_coords + range_limit) / (2 * range_limit) * bev_w).astype(int)
    y_pixels = ((y_coords + range_limit) / (2 * range_limit) * bev_h).astype(int)
    
    # 범위 체크
    valid_pixels = (x_pixels >= 0) & (x_pixels < bev_w) & (y_pixels >= 0) & (y_pixels < bev_h)
    x_pixels = x_pixels[valid_pixels]
    y_pixels = y_pixels[valid_pixels]
    intensity_values = intensity_values[valid_pixels]
    
    # BEV에 값 할당 (같은 픽셀에 여러 점이 있으면 최대값 사용)
    for x_pix, y_pix, intensity in zip(x_pixels, y_pixels, intensity_values):
        bev_image[y_pix, x_pix] = max(bev_image[y_pix, x_pix], intensity)
    
    return torch.tensor(bev_image).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

def load_dataset_mmdet3d(config_path, data_root, ann_file, train_mode=True):
    """mmdet3d 방식으로 데이터셋 로드 (원본 points)"""
    import os
    cfg = Config.fromfile(config_path)

    ann_file_abs = os.path.join(data_root, ann_file)
    log(f"Loading dataset with config: {config_path}, ann_file: {ann_file_abs}, train_mode: {train_mode}")

    # Force test_mode=True to skip annotations
    test_mode = True

    if train_mode:
        if hasattr(cfg.data, 'train'):
            cfg.data.train.data_root = data_root
            cfg.data.train.ann_file = ann_file_abs
            cfg.data.train.test_mode = test_mode
            dataset = build_dataset(cfg.data.train)
        else:
            raise ValueError("No train config found in cfg.data")
    else:
        if hasattr(cfg.data, 'val'):
            cfg.data.val.data_root = data_root
            cfg.data.val.ann_file = ann_file_abs
            cfg.data.val.test_mode = test_mode
            dataset = build_dataset(cfg.data.val)
        elif hasattr(cfg.data, 'test'):
            cfg.data.test.data_root = data_root
            cfg.data.test.ann_file = ann_file_abs
            cfg.data.test.test_mode = test_mode
            dataset = build_dataset(cfg.data.test)
        else:
            raise ValueError("No val/test config found in cfg.data")

    log(f"Dataset loaded successfully. Number of samples: {len(dataset)}")
    return dataset

@torch.no_grad()
def generate_latent_ids(H, ae_radar, ae_lidar, train_radar_dataset, train_lidar_dataset, test_radar_dataset, test_lidar_dataset):
    """Radar BEV와 Lidar BEV 데이터로부터 잠재 코드 생성 (mmdet3d 데이터셋 사용)"""
    log("Generating latent codes...")
    log_gpu_memory()
    
    def generate_latents_from_dataset(radar_dataset, lidar_dataset, split_name):
        latent_ids = []
        num_samples = min(len(radar_dataset), len(lidar_dataset))
        log(f"Processing {split_name} dataset with {num_samples} samples")
        for i in tqdm(range(num_samples), desc=f"Processing {split_name}"):
            radar_data = radar_dataset[i]
            lidar_data = lidar_dataset[i]
            
            # Radar points 추출 및 BEV 변환
            radar_points = extract_points_from_data(radar_data, H)
            if radar_points is None:
                log(f"Warning: No radar points in sample {i}")
                continue
            radar_bev = points_to_bev(radar_points, H.radar_config.data.img_size).to(device)
            log(f"Processed radar sample {i}: BEV shape {radar_bev.shape}")
            
            # Lidar points 추출 및 BEV 변환
            lidar_points = extract_points_from_data(lidar_data, H)
            if lidar_points is None:
                log(f"Warning: No lidar points in sample {i}")
                continue
            lidar_bev = points_to_bev(lidar_points, H.lidar_config.data.img_size).to(device)
            log(f"Processed lidar sample {i}: BEV shape {lidar_bev.shape}")
            
            B = 1  # single sample
            
            # Radar BEV 인코딩 (UltraLiDAR wrapper 사용)
            radar_latents = ae_radar.encoder(radar_bev)
            radar_quant, _, _ = ae_radar.quantize(radar_latents)
            
            # 올바른 reshape: (B, C, H, W) -> (B, H*W, C)
            _, C, H_r, W_r = radar_quant.shape
            radar_quant = radar_quant.permute(0, 2, 3, 1).contiguous().view(B, H_r*W_r, C)  # (B, seq_len=H*W, emb_dim=C)
            
            # Lidar BEV 인코딩 (UltraLiDAR wrapper 사용)
            lidar_latents = ae_lidar.encoder(lidar_bev)
            _, _, lidar_quant_stats = ae_lidar.quantize(lidar_latents)
            lidar_min_encoding_indices = lidar_quant_stats["min_encoding_indices"]
            lidar_min_encoding_indices = lidar_min_encoding_indices.view(B, -1)
            
            latent_ids.append({
                "radar_embed": radar_quant.cpu().contiguous(),  # (B, seq_len, emb_dim)
                "lidar_codes": lidar_min_encoding_indices.cpu().contiguous()  # (B, seq_len)
            })
        
        return latent_ids
    
    train_latent_ids = generate_latents_from_dataset(train_radar_dataset, train_lidar_dataset, "train")
    test_latent_ids = generate_latents_from_dataset(test_radar_dataset, test_lidar_dataset, "test")
    
    # 잠재 코드 저장
    os.makedirs(f'logs/{H.run.name}_{H.run.experiment}', exist_ok=True)
    torch.save(train_latent_ids, f'logs/{H.run.name}_{H.run.experiment}/train_latents')
    torch.save(test_latent_ids, f'logs/{H.run.name}_{H.run.experiment}/val_latents')
    
    log("Latent codes generation completed")

def train(H, sampler, sampler_ema, generator_radar, generator_lidar, train_loader, test_loader, optim, start_step, vis=None):
    scaler = None
    if H.train.amp:
        scaler = torch.cuda.amp.GradScaler()
    
    global_step = start_step
    tracked_stats = defaultdict(lambda: np.array([]))
    test_tracked_stats = defaultdict(lambda: np.array([]))
    
    while global_step <= H.train.total_steps:
        for data in train_loader:
            start_time = time.time()
            
            # 데이터 키 확인 및 호환성 처리
            if "radar_embed" in data:
                # 이미 처리된 latent 데이터
                context = data["radar_embed"].to(device, non_blocking=True)  # Radar context: (B, seq_len, emb_dim)
                x = data["lidar_codes"].to(device, non_blocking=True)         # Lidar target codes: (B, seq_len)
            else:
                # 예상치 못한 데이터 형식
                log(f"Warning: Unexpected data format. Keys: {list(data.keys())}")
                continue
            
            # Context는 이미 올바른 형태 (B, seq_len, emb_dim)이므로 추가 reshape 불필요
            # 단, 기존 코드와의 호환성을 위해 차원 확인
            if len(context.shape) == 4 and context.size(1) == 1:
                # (B, 1, seq_len, emb_dim) 형태인 경우 squeeze
                context = context.squeeze(1)  # (B, seq_len, emb_dim)
            elif len(context.shape) != 3:
                log(f"Warning: Unexpected context shape: {context.shape}")
                continue
            
            # Target codes도 올바른 형태 확인  
            if len(x.shape) == 3 and x.size(1) == 1:
                # (B, 1, seq_len) 형태인 경우 squeeze
                x = x.squeeze(1)  # (B, seq_len)
            elif len(x.shape) != 2:
                log(f"Warning: Unexpected target shape: {x.shape}")
                continue

            if global_step < H.optimizer.warmup_steps:
                optim_warmup(global_step, optim, H.optimizer.learning_rate, H.optimizer.warmup_steps)

            global_step += 1

            with torch.cuda.amp.autocast(enabled=H.train.amp):
                stats = sampler.train_iter(x, context=context)
            
            update_model_weights(optim, stats["loss"], amp=H.train.amp, scaler=scaler)

            if global_step % H.train.ema_update_every == 0:
                update_ema(sampler, sampler_ema, H.train.ema_decay)
            
            stats["step_time"] = time.time() - start_time
            track_variables(tracked_stats, stats)

            wandb_dict = dict()
            
            ## Plot graphs
            if global_step % H.train.plot_graph_steps == 0 and global_step > 0:
                wandb_dict.update(log_stats(H, global_step, tracked_stats, log_to_file=H.run.log_to_file))
            
            ## Plot reconstructions
            # if global_step % H.train.plot_recon_steps == 0 and global_step > 0:
            #     # 원본 Lidar BEV 재구성
            #     x_img = reconstruct_from_codes(H, sampler, x[:H.diffusion.sampling_batch_size], generator_lidar)
            #     wandb_dict.update(plot_images(H, x_img, title='gt_lidar_bev', vis=vis))
                
            #     # 샘플링된 Lidar BEV
            #     with torch.no_grad():
            #         with torch.cuda.amp.autocast(enabled=H.train.amp):
            #             x_sampled = sampler.sample(
            #                 context=context[:H.diffusion.sampling_batch_size], 
            #                 sample_steps=H.diffusion.sampling_steps, 
            #                 temp=H.diffusion.sampling_temp
            #             )
            #     x_sampled_img = reconstruct_from_codes(H, sampler, x_sampled, generator_lidar)
            #     wandb_dict.update(plot_images(H, x_sampled_img, title='sampled_lidar_bev', vis=vis))
                
                # 최대 확률 샘플링
                # with torch.no_grad():
                #     with torch.cuda.amp.autocast(enabled=H.train.amp):
                #         x_maxprob = sampler.sample_max_probability(
                #             context=context[:H.diffusion.sampling_batch_size], 
                #             sample_steps=H.diffusion.sampling_steps
                #         )
                # x_maxprob_img = reconstruct_from_codes(H, sampler, x_maxprob, generator_lidar)
                # wandb_dict.update(plot_images(H, x_maxprob_img, title='maxprob_lidar_bev', vis=vis))
            
            ## Evaluate on test set
            if global_step % H.train.eval_steps == 0 and global_step > 0:
                log("Evaluating...")
                for _ in tqdm(range(H.train.eval_repeats)):
                    for test_data in test_loader:
                        test_context = test_data["radar_embed"].to(device, non_blocking=True)
                        test_x = test_data["lidar_codes"].to(device, non_blocking=True)
                        
                        # Training loop와 동일한 로직 적용
                        if len(test_context.shape) == 4 and test_context.size(1) == 1:
                            test_context = test_context.squeeze(1)
                        elif len(test_context.shape) != 3:
                            log(f"Warning: Unexpected test context shape: {test_context.shape}")
                            continue
                        
                        if len(test_x.shape) == 3 and test_x.size(1) == 1:
                            test_x = test_x.squeeze(1)
                        elif len(test_x.shape) != 2:
                            log(f"Warning: Unexpected test target shape: {test_x.shape}")
                            continue
                        
                        with torch.no_grad():
                            with torch.cuda.amp.autocast(enabled=H.train.amp):
                                test_stats = sampler.train_iter(test_x, context=test_context, test=True)
                        track_variables(test_tracked_stats, test_stats)
                wandb_dict.update(log_stats(H, global_step, test_tracked_stats, test=True, log_to_file=H.run.log_to_file))

            ## Checkpoint
            if global_step % H.train.checkpoint_steps == 0 and global_step > 0:
                save_model(sampler, H.model.name, global_step, f"{H.run.name}_{H.run.experiment}")
                save_model(sampler_ema, f"{H.model.name}_ema", global_step, f"{H.run.name}_{H.run.experiment}")

            ## Log to wandb
            if wandb_dict:
                wandb.log(wandb_dict, step=global_step)

def main(argv):
    global device
    H = FLAGS.config
    H.radar_config = FLAGS.radar_config
    H.lidar_config = FLAGS.lidar_config
    
    # GPU 설정
    device = setup_device(H)
    
    train_kwargs = {}

    # wandb 초기화
    wandb.init(
        name=H.run.experiment, 
        project=H.run.name, 
        config=flatten_collection(H), 
        save_code=True, 
        dir=H.run.wandb_dir, 
        mode=H.run.wandb_mode
    )
    
    if H.run.enable_visdom:
        train_kwargs['vis'] = visdom.Visdom(server=H.run.visdom_server, port=H.run.visdom_port)
    
    if H.run.log_to_file:
        config_log(H.run.name)

    # 잠재 코드 생성 (필요한 경우)
    latents_filepath = f'logs/{H.run.name}_{H.run.experiment}/train_latents'
    if not os.path.exists(latents_filepath):
        log("Creating latent codes...")
        
        # 사전 훈련된 VQGAN 모델 로드 (full loading)
        ae_radar = load_pretrained_vqgan(H.model_paths.radar_vqgan_path, FLAGS.radar_ultralidar_config, device, model_type='radar')
        ae_lidar = load_pretrained_vqgan(H.model_paths.lidar_vqgan_path, FLAGS.lidar_ultralidar_config, device, model_type='lidar')

        # 데이터셋 로드 (mmdet3d 사용)
        train_radar_ann_file = H.data.radar_train_ann_file
        val_radar_ann_file = H.data.radar_val_ann_file
        train_lidar_ann_file = H.data.lidar_train_ann_file
        val_lidar_ann_file = H.data.lidar_val_ann_file
        
        # Radar datasets
        train_radar_dataset = load_dataset_mmdet3d(FLAGS.radar_ultralidar_config, H.data.data_root, train_radar_ann_file, train_mode=True)
        test_radar_dataset = load_dataset_mmdet3d(FLAGS.radar_ultralidar_config, H.data.data_root, val_radar_ann_file, train_mode=False)
        
        # Lidar datasets
        train_lidar_dataset = load_dataset_mmdet3d(FLAGS.lidar_ultralidar_config, H.data.data_root, train_lidar_ann_file, train_mode=True)
        test_lidar_dataset = load_dataset_mmdet3d(FLAGS.lidar_ultralidar_config, H.data.data_root, val_lidar_ann_file, train_mode=False)
        
        generate_latent_ids(H, ae_radar, ae_lidar, train_radar_dataset, train_lidar_dataset, test_radar_dataset, test_lidar_dataset)
        
        # 메모리 정리
        del ae_radar
        del ae_lidar
    
    # 잠재 코드 로더
    train_latent_loader, test_latent_loader = get_latent_loaders(H)

    # Generator 로드 (decoder part)
    full_radar = load_pretrained_vqgan(H.model_paths.radar_vqgan_path, FLAGS.radar_ultralidar_config, device, model_type='radar')
    generator_radar = full_radar.lidar_decoder  # assuming named lidar_decoder even for radar
    full_lidar = load_pretrained_vqgan(H.model_paths.lidar_vqgan_path, FLAGS.lidar_ultralidar_config, device, model_type='lidar')
    generator_lidar = full_lidar.lidar_decoder
    
    # 모델들을 eval 모드로 설정 (추론용)
    generator_radar.eval()
    generator_lidar.eval()

    # Sampler 생성
    # lidar VQGAN에서 embedding weight 추출 (from full model)
    try:
        lidar_embedding_weight = full_lidar.vector_quantizer.embedding.weight
        log(f"Extracted embedding weight shape: {lidar_embedding_weight.shape}")
    except Exception as e:
        log(f"Error extracting embedding weight: {e}")
        log("Using random initialization for embedding weight.")
        lidar_embedding_weight = torch.randn(H.lidar_config.model.codebook_size, H.lidar_config.model.emb_dim)
    
    # ct_config를 lidar_config로 대체하여 sampler 생성
    H.ct_config = H.lidar_config  # 호환성을 위해 ct_config에 lidar_config 할당
    sampler = get_sampler(H, lidar_embedding_weight.to(device)).to(device)
    sampler_ema = copy.deepcopy(sampler).to(device)

    # Optimizer
    if H.optimizer.weight_decay > 0:
        optim = torch.optim.AdamW(sampler.parameters(), lr=H.optimizer.learning_rate, weight_decay=H.optimizer.weight_decay)
    else:
        optim = torch.optim.Adam(sampler.parameters(), lr=H.optimizer.learning_rate)

    start_step = 0
    if H.train.load_step > 0:
        start_step = H.train.load_step + 1
        sampler = load_model(sampler, H.model.name, H.train.load_step, f"{H.run.name}_{H.run.experiment}").to(device)
        sampler_ema = load_model(sampler_ema, f'{H.model.name}_ema', H.train.load_step, f"{H.run.name}_{H.run.experiment}")

    # 훈련 시작
    train(H, sampler, sampler_ema, generator_radar, generator_lidar, train_latent_loader, test_latent_loader, optim, start_step, **train_kwargs)

if __name__ == '__main__':
    app.run(main)