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

# Commandline arguments
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
config_flags.DEFINE_config_file("radar_config", "configs/default_radar_vqgan_config.py", "Radar VQGAN training configuration.", lock_config=True)
config_flags.DEFINE_config_file("lidar_config", "configs/default_lidar_vqgan_config.py", "Lidar VQGAN training configuration.", lock_config=True)
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

def load_pretrained_vqgan(model_path, model_class, config, device, is_ultralidar=False, model_type='radar'):
    """사전 훈련된 VQGAN 모델 로드"""
    if is_ultralidar:
        # 간단한 UltraLiDAR 모델 로드 (mmcv 의존성 없음)
        try:
            model = load_simple_ultralidar_model(model_path, model_type)
            log(f"Successfully loaded simple UltraLiDAR {model_type} model from {model_path}")
            return model.to(device)
            
        except Exception as e:
            log(f"Error loading simple UltraLiDAR {model_type} model: {e}")
            log("Falling back to standard VQGAN loading...")
    
    # 기존 VQGAN 로딩 방식
    model = model_class(config)
    
    if os.path.exists(model_path):
        log(f"Loading pretrained model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # 체크포인트 구조에 따라 state_dict 추출
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # 키 이름 조정 (필요한 경우)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            # 'ae.' prefix 제거
            if key.startswith('ae.'):
                new_key = key[3:]
            else:
                new_key = key
            cleaned_state_dict[new_key] = value
            
        model.load_state_dict(cleaned_state_dict, strict=False)
        log(f"Successfully loaded model from {model_path}")
    else:
        log(f"Warning: Model file {model_path} not found. Using randomly initialized weights.")
    
    return model.to(device)

@torch.no_grad()
def generate_latent_ids(H, ae_radar, ae_lidar, train_loader, test_loader):
    """Radar BEV와 Lidar BEV 데이터로부터 잠재 코드 생성"""
    log("Generating latent codes...")
    log_gpu_memory()
    
    def generate_latents_from_loader(dataloader, split_name):
        latent_ids = []
        for i, data in enumerate(tqdm(dataloader, desc=f"Processing {split_name}")):
            
            # NuScenes 데이터 형식 처리
            if "radar_bev" in data and "lidar_bev" in data:
                # NuScenesRadarLidarDataset 형식
                radar_bev = data["radar_bev"].to(device)    # (B, 1, H, W)
                lidar_bev = data["lidar_bev"].to(device)    # (B, 1, H, W)
                
                # Radar BEV 인코딩 (UltraLiDAR wrapper 사용)
                radar_latents = ae_radar.encoder(radar_bev)
                radar_quant, _, _ = ae_radar.quantize(radar_latents)
                
                # 올바른 reshape: (B, C, H, W) -> (B, H*W, C)
                B, C, H, W = radar_quant.shape
                radar_quant = radar_quant.permute(0, 2, 3, 1).contiguous().view(B, H*W, C)  # (B, seq_len=H*W, emb_dim=C)
                
                # Lidar BEV 인코딩 (UltraLiDAR wrapper 사용)
                lidar_latents = ae_lidar.encoder(lidar_bev)
                _, _, lidar_quant_stats = ae_lidar.quantize(lidar_latents)
                lidar_min_encoding_indices = lidar_quant_stats["min_encoding_indices"]
                lidar_min_encoding_indices = lidar_min_encoding_indices.view(B, -1)
                
            elif "radar_bevs" in data and "lidar_bev" in data:
                # 기존 RadarLidarBEV_dataset 형식
                radar_bevs = data["radar_bevs"].to(device)  # (B, num_views, C, H, W)
                lidar_bev = data["lidar_bev"].to(device)    # (B, C, H, W)
                
                # Radar BEV 인코딩
                B, num_views, C, H, W = radar_bevs.shape
                radar_bevs_flat = rearrange(radar_bevs, "b v c h w -> (b v) c h w")
                radar_latents = ae_radar.encoder(radar_bevs_flat)
                radar_quant, _, _ = ae_radar.quantize(radar_latents)
                radar_quant = rearrange(radar_quant, "(b v) c h w -> b v (h w) c", b=B)
                
                # Lidar BEV 인코딩
                lidar_latents = ae_lidar.encoder(lidar_bev)
                _, _, lidar_quant_stats = ae_lidar.quantize(lidar_latents)
                lidar_min_encoding_indices = lidar_quant_stats["min_encoding_indices"]
                lidar_min_encoding_indices = lidar_min_encoding_indices.view(B, -1)
            
            else:
                log(f"Warning: Unexpected data format in batch. Keys: {list(data.keys())}")
                continue
            
            latent_ids.append({
                "radar_embed": radar_quant.cpu().contiguous(),  # (B, seq_len, emb_dim)
                "lidar_codes": lidar_min_encoding_indices.cpu().contiguous()  # (B, seq_len)
            })
        
        return latent_ids
    
    train_latent_ids = generate_latents_from_loader(train_loader, "train")
    test_latent_ids = generate_latents_from_loader(test_loader, "test")
    
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
            if global_step % H.train.plot_recon_steps == 0 and global_step > 0:
                # 원본 Lidar BEV 재구성
                x_img = reconstruct_from_codes(H, sampler, x[:H.diffusion.sampling_batch_size], generator_lidar)
                wandb_dict.update(plot_images(H, x_img, title='gt_lidar_bev', vis=vis))
                
                # 샘플링된 Lidar BEV
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=H.train.amp):
                        x_sampled = sampler.sample(
                            context=context[:H.diffusion.sampling_batch_size], 
                            sample_steps=H.diffusion.sampling_steps, 
                            temp=H.diffusion.sampling_temp
                        )
                x_sampled_img = reconstruct_from_codes(H, sampler, x_sampled, generator_lidar)
                wandb_dict.update(plot_images(H, x_sampled_img, title='sampled_lidar_bev', vis=vis))
                
                # 최대 확률 샘플링
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=H.train.amp):
                        x_maxprob = sampler.sample_max_probability(
                            context=context[:H.diffusion.sampling_batch_size], 
                            sample_steps=H.diffusion.sampling_steps
                        )
                x_maxprob_img = reconstruct_from_codes(H, sampler, x_maxprob, generator_lidar)
                wandb_dict.update(plot_images(H, x_maxprob_img, title='maxprob_lidar_bev', vis=vis))
            
            ## Evaluate on test set
            if global_step % H.train.eval_steps == 0 and global_step > 0:
                log("Evaluating...")
                for _ in tqdm(range(H.train.eval_repeats)):
                    for test_data in test_loader:
                        test_context = test_data["radar_embed"].to(device, non_blocking=True)
                        test_x = test_data["lidar_codes"].to(device, non_blocking=True)
                        test_context = rearrange(test_context, "b () r l c -> b (r l) c")
                        test_x = rearrange(test_x, "b () l -> b l")
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
        
        # 사전 훈련된 VQGAN 모델 로드
        ae_radar = load_pretrained_vqgan(
            H.model_paths.radar_vqgan_path, 
            VQAutoEncoder2D, 
            H.radar_config, 
            device,
            is_ultralidar=True,  # UltraLiDAR 모델임을 명시
            model_type='radar'
        )
        ae_lidar = load_pretrained_vqgan(
            H.model_paths.lidar_vqgan_path, 
            VQAutoEncoder2D,  # 또는 VQAutoEncoder3D (lidar BEV 차원에 따라)
            H.lidar_config, 
            device,
            is_ultralidar=True,  # lidar도 UltraLiDAR 모델 사용
            model_type='lidar'
        )

        # 데이터셋 로드
        if H.data.loader == "nuscenes_radar":
            train_dataset = NuScenesRadarDataset(
                data_root=H.data.data_root,
                ann_file=os.path.join(H.data.data_root, "nuscenes_infos_train_radar.pkl"),
                train=True,
                bev_size=H.radar_config.data.img_size
            )
            test_dataset = NuScenesRadarDataset(
                data_root=H.data.data_root,
                ann_file=os.path.join(H.data.data_root, "nuscenes_infos_val_radar.pkl"),
                train=False,
                bev_size=H.radar_config.data.img_size
            )
        elif H.data.loader == "nuscenes_radar_lidar":
            # 설정에서 지정된 pkl 파일 사용
            train_ann_file = getattr(H.data, 'train_ann_file', 'nuscenes_infos_train_radar.pkl')
            val_ann_file = getattr(H.data, 'val_ann_file', 'nuscenes_infos_val_radar.pkl')
            
            train_dataset = NuScenesRadarLidarDataset(
                data_root=H.data.data_root,
                ann_file=os.path.join(H.data.data_root, train_ann_file),
                train=True,
                bev_size=H.radar_config.data.img_size
            )
            test_dataset = NuScenesRadarLidarDataset(
                data_root=H.data.data_root,
                ann_file=os.path.join(H.data.data_root, val_ann_file),
                train=False,
                bev_size=H.radar_config.data.img_size
            )
        elif H.data.loader == "radar_lidar_bev_simple":
            train_dataset = RadarLidarBEV_dataset_simple(
                data_file=os.path.join(H.data.data_dir, "train_data.npz"),
                train=True,
                radar_scale=H.radar_config.data.img_size,
                lidar_scale=H.lidar_config.data.img_size
            )
            test_dataset = RadarLidarBEV_dataset_simple(
                data_file=os.path.join(H.data.data_dir, "test_data.npz"),
                train=False,
                radar_scale=H.radar_config.data.img_size,
                lidar_scale=H.lidar_config.data.img_size
            )
        else:
            train_dataset = RadarLidarBEV_dataset(
                data_dir=H.data.data_dir,
                train=True,
                radar_scale=H.radar_config.data.img_size,
                lidar_scale=H.lidar_config.data.img_size,
                num_radar_views=H.data.num_radar_views
            )
            test_dataset = RadarLidarBEV_dataset(
                data_dir=H.data.data_dir,
                train=False,
                radar_scale=H.radar_config.data.img_size,
                lidar_scale=H.lidar_config.data.img_size,
                num_radar_views=H.data.num_radar_views
            )
        
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, 
                                  num_workers=2, pin_memory=True, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                                 num_workers=2, pin_memory=True, drop_last=False)
        
        generate_latent_ids(H, ae_radar, ae_lidar, train_loader, test_loader)
        
        # 메모리 정리
        del ae_radar
        del ae_lidar
    
    # 잠재 코드 로더
    train_latent_loader, test_latent_loader = get_latent_loaders(H)

    # Generator 로드
    generator_radar = load_pretrained_vqgan(
        H.model_paths.radar_vqgan_path, 
        Generator2D, 
        H.radar_config, 
        device,
        is_ultralidar=True,
        model_type='radar'
    )
    generator_lidar = load_pretrained_vqgan(
        H.model_paths.lidar_vqgan_path, 
        Generator2D,  # 또는 Generator3D
        H.lidar_config, 
        device,
        is_ultralidar=True,  # lidar도 UltraLiDAR 모델 사용
        model_type='lidar'
    )

    # Sampler 생성
    # lidar VQGAN에서 embedding weight 추출
    try:
        # UltraLiDAR 모델에서 임베딩 가중치 추출
        if hasattr(generator_lidar, 'get_embedding_weight'):
            lidar_embedding_weight = generator_lidar.get_embedding_weight()
            log(f"Successfully extracted embedding weight from UltraLiDAR model: {lidar_embedding_weight.shape}")
        else:
            # 직접 체크포인트에서 추출
            lidar_checkpoint = torch.load(H.model_paths.lidar_vqgan_path, map_location=device)
            lidar_embedding_weight = None
            
            # 다양한 키 형식 시도
            possible_keys = [
                'vector_quantizer.embedding.weight',
                'quantize.embedding.weight', 
                'ae.quantize.embedding.weight',
                'model.vector_quantizer.embedding.weight',
                'state_dict.vector_quantizer.embedding.weight'
            ]
            
            for key in possible_keys:
                if key in lidar_checkpoint:
                    lidar_embedding_weight = lidar_checkpoint[key]
                    log(f"Found lidar embedding weight with key: {key}")
                    break
            
            # state_dict가 중첩된 경우
            if lidar_embedding_weight is None and 'state_dict' in lidar_checkpoint:
                state_dict = lidar_checkpoint['state_dict']
                for key in possible_keys:
                    if key in state_dict:
                        lidar_embedding_weight = state_dict[key]
                        log(f"Found lidar embedding weight in state_dict with key: {key}")
                        break
            
            # 키를 직접 순회하며 embedding.weight 포함된 키 찾기
            if lidar_embedding_weight is None:
                all_keys = list(lidar_checkpoint.keys())
                if 'state_dict' in lidar_checkpoint:
                    all_keys.extend([f"state_dict.{k}" for k in lidar_checkpoint['state_dict'].keys()])
                
                for key in all_keys:
                    if 'embedding.weight' in key and 'vector_quantizer' in key:
                        if key.startswith('state_dict.'):
                            lidar_embedding_weight = lidar_checkpoint['state_dict'][key[11:]]
                        else:
                            lidar_embedding_weight = lidar_checkpoint[key]
                        log(f"Found lidar embedding weight with key: {key}")
                        break
            
            if lidar_embedding_weight is None:
                log("Warning: Could not find lidar embedding weight. Using random initialization.")
                lidar_embedding_weight = torch.randn(H.lidar_config.model.codebook_size, H.lidar_config.model.emb_dim)
            
    except Exception as e:
        log(f"Error loading lidar embedding weight: {e}")
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
