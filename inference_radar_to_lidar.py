import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import ml_collections
import time
try:
    from mmcv.utils import Config
except ImportError:
    from mmcv.utils.config import Config
from mmdet3d.datasets import build_dataset

# UltraLiDAR 플러그인 임포트 (모델 등록을 위해)
sys.path.append('/home/byounggun/r2l/UltraLiDAR_nusc_waymo')
import plugin

# 기존 모델들
from models.vqgan_2d import Generator as Generator2D
from utils.simple_ultralidar_wrapper import load_simple_ultralidar_model
from utils.sampler_utils import get_sampler, latent_ids_to_onehot
from utils.log_utils import log, flatten_collection, load_model

# Commandline arguments
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
config_flags.DEFINE_config_file("radar_config", "configs/default_radar_vqgan_config.py", "Radar VQGAN training configuration.", lock_config=True)
config_flags.DEFINE_config_file("lidar_config", "configs/default_lidar_vqgan_config.py", "Lidar VQGAN training configuration.", lock_config=True)
flags.DEFINE_string("model_path", "/home/byounggun/r2l/x2ct-vqvae/logs/radar_to_lidar_sampler_radar_lidar/saved_models/absorbing_5000.th", "Path to trained model")
flags.DEFINE_string("output_dir", "./inference_results", "Output directory for results")
flags.DEFINE_integer("num_samples", 5, "Number of samples to inference")
flags.DEFINE_string("radar_ultralidar_config", "configs/ultralidar_nusc_radar_debug.py", "UltraLiDAR radar config path")
flags.DEFINE_string("lidar_ultralidar_config", "configs/ultralidar_nusc.py", "UltraLiDAR lidar config path")
flags.mark_flags_as_required(["config"])

def setup_device(config):
    """GPU 설정 및 디바이스 선택"""
    if hasattr(config.run, 'use_cuda') and config.run.use_cuda and torch.cuda.is_available():
        if hasattr(config.run, 'gpu_id'):
            gpu_id = config.run.gpu_id
            if gpu_id < torch.cuda.device_count():
                torch.cuda.set_device(gpu_id)
                device = torch.device(f'cuda:{gpu_id}')
                log(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
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

def load_pretrained_ultralidar(model_path, model_type, device):
    """UltraLiDAR 모델 로드"""
    try:
        model = load_simple_ultralidar_model(model_path, model_type)
        log(f"Successfully loaded UltraLiDAR {model_type} model from {model_path}")
        return model.to(device)
    except Exception as e:
        log(f"Error loading UltraLiDAR {model_type} model: {e}")
        raise e

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

def load_dataset_mmdet3d(config_path, data_root):
    """mmdet3d 방식으로 데이터셋 로드 (원본 points)"""
    import os
    cfg = Config.fromfile(config_path)

    ann_file_abs = os.path.join(data_root, 'nuscenes_infos_val_radar_tiny.pkl')

    if hasattr(cfg.data, 'test'):
        cfg.data.test.data_root = data_root
        cfg.data.test.ann_file = ann_file_abs
        cfg.data.test.test_mode = True
        dataset = build_dataset(cfg.data.test)
    else:
        cfg.data.train.data_root = data_root
        cfg.data.train.ann_file = ann_file_abs
        cfg.data.train.test_mode = True
        dataset = build_dataset(cfg.data.train)

    return dataset

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

def reconstruct_from_codes(H, sampler, codes, generator):
    """코드로부터 이미지 재구성 - eval_sampler.py 방식 사용"""
    # codes: (B, seq_len) - 이산 코드 인덱스
    latents_one_hot = latent_ids_to_onehot(codes, H.lidar_config.model.latent_shape, H.lidar_config.model.codebook_size)
    q = sampler.embed(latents_one_hot)
    images = generator(q.float())
    return images

def bev_to_points(bev_image, intensity_threshold=0.01):
    """BEV 이미지를 포인트로 변환 (radar 시각화 코드 참조)"""
    if isinstance(bev_image, torch.Tensor):
        bev_image = bev_image.cpu().numpy()
    
    # 임계값보다 큰 값들의 위치 찾기
    y_indices, x_indices = np.where(bev_image > intensity_threshold)
    
    if len(x_indices) == 0:
        return np.zeros((0, 3))  # x, y, intensity
    
    # 픽셀 좌표를 실제 좌표로 변환 (예: -50m ~ +50m)
    height, width = bev_image.shape
    x_coords = (x_indices / width) * 100.0 - 50.0
    y_coords = (y_indices / height) * 100.0 - 50.0
    
    # 강도값 추출
    intensities = bev_image[y_indices, x_indices]
    
    points = np.stack([x_coords, y_coords, intensities], axis=1)
    return points

def visualize_results_with_points(radar_points, radar_bev, lidar_points, lidar_gt, lidar_pred, sample_idx, output_dir):
    """원본 points와 BEV 결과를 모두 시각화 - 라이다는 포인트 클라우드로"""
    
    # 원본 데이터를 numpy로 변환
    radar_img = radar_bev[0, 0].cpu().numpy()  # (H, W)
    lidar_gt_img = lidar_gt[0, 0].cpu().numpy() if lidar_gt is not None else None  # (H, W)
    lidar_pred_img = lidar_pred[0, 0].cpu().numpy()  # (H, W)
    
    # 크기 불일치 해결 - 예측 결과를 GT 크기에 맞춤
    if lidar_gt_img is not None and lidar_pred_img.shape != lidar_gt_img.shape:
        from scipy.ndimage import zoom
        scale_factor = (lidar_gt_img.shape[0] / lidar_pred_img.shape[0], 
                       lidar_gt_img.shape[1] / lidar_pred_img.shape[1])
        lidar_pred_img = zoom(lidar_pred_img, scale_factor, order=1)
        log(f"Resized prediction from {lidar_pred.shape} to {lidar_pred_img.shape}")
    
    # 데이터 범위 확인
    log(f"Data ranges - Radar: [{radar_img.min():.3f}, {radar_img.max():.3f}]")
    if lidar_gt_img is not None:
        log(f"Data ranges - Lidar GT: [{lidar_gt_img.min():.3f}, {lidar_gt_img.max():.3f}]")
    log(f"Data ranges - Lidar Pred: [{lidar_pred_img.min():.3f}, {lidar_pred_img.max():.3f}]")
    
    # BEV를 포인트로 변환
    radar_bev_points = bev_to_points(radar_img, intensity_threshold=0.01)  # Radar는 기존 방식
    
    # LiDAR는 포인트 클라우드로 변환
    lidar_gt_points = bev_to_points_lidar(lidar_gt_img, intensity_threshold=0.01) if lidar_gt_img is not None else np.zeros((0, 4))
    lidar_pred_points = bev_to_points_lidar(lidar_pred_img, intensity_threshold=0.01)
    
    # 원본 points 준비
    radar_points_np = radar_points.cpu().numpy() if radar_points is not None else np.zeros((0, 6))
    lidar_points_np = lidar_points.cpu().numpy() if lidar_points is not None else np.zeros((0, 6))
    
    log(f"Original point counts - Radar: {len(radar_points_np)}, Lidar: {len(lidar_points_np)}")
    log(f"Generated point counts - Radar BEV: {len(radar_bev_points)}, Lidar GT: {len(lidar_gt_points)}, Lidar Pred: {len(lidar_pred_points)}")
    
    # 시각화 - 원본 points와 생성된 결과
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 첫 번째 행: 원본 points
    # Original Radar Points
    if len(radar_points_np) > 0:
        scatter1 = axes[0, 0].scatter(radar_points_np[:, 0], radar_points_np[:, 1], 
                                     c=radar_points_np[:, 3] if radar_points_np.shape[1] > 3 else 'blue', 
                                     cmap='viridis', s=2, alpha=0.7)
        if radar_points_np.shape[1] > 3:
            plt.colorbar(scatter1, ax=axes[0, 0], shrink=0.8)
    axes[0, 0].set_title(f'Original Radar Points\n{len(radar_points_np)} points', 
                        fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('X (m)')
    axes[0, 0].set_ylabel('Y (m)')
    axes[0, 0].set_xlim(-50, 50)
    axes[0, 0].set_ylim(-50, 50)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Original Lidar Points
    if len(lidar_points_np) > 0:
        scatter2 = axes[0, 1].scatter(lidar_points_np[:, 0], lidar_points_np[:, 1], 
                                     c=lidar_points_np[:, 3] if lidar_points_np.shape[1] > 3 else 'red', 
                                     cmap='hot', s=1, alpha=0.6)
        if lidar_points_np.shape[1] > 3:
            plt.colorbar(scatter2, ax=axes[0, 1], shrink=0.8)
    axes[0, 1].set_title(f'Original Lidar Points\n{len(lidar_points_np)} points', 
                        fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('X (m)')
    axes[0, 1].set_ylabel('Y (m)')
    axes[0, 1].set_xlim(-50, 50)
    axes[0, 1].set_ylim(-50, 50)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Empty plot for symmetry
    axes[0, 2].axis('off')
    axes[0, 2].text(0.5, 0.5, 'Original Points\nComparison', ha='center', va='center', 
                   transform=axes[0, 2].transAxes, fontsize=14, fontweight='bold')
    
    # 두 번째 행: 생성된 결과
    # Radar BEV (기존 방식)
    if len(radar_bev_points) > 0:
        scatter3 = axes[1, 0].scatter(radar_bev_points[:, 0], radar_bev_points[:, 1], 
                                     c=radar_bev_points[:, 2], cmap='viridis', s=2, alpha=0.7)
        plt.colorbar(scatter3, ax=axes[1, 0], shrink=0.8)
    axes[1, 0].set_title(f'Radar BEV (Input)\n{len(radar_bev_points)} points', 
                        fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('X (m)')
    axes[1, 0].set_ylabel('Y (m)')
    axes[1, 0].set_xlim(-50, 50)
    axes[1, 0].set_ylim(-50, 50)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Ground Truth Lidar Points (포인트 클라우드)
    if len(lidar_gt_points) > 0:
        scatter4 = axes[1, 1].scatter(lidar_gt_points[:, 0], lidar_gt_points[:, 1], 
                                     c=lidar_gt_points[:, 3], cmap='hot', s=1, alpha=0.6)
        plt.colorbar(scatter4, ax=axes[1, 1], shrink=0.8)
    axes[1, 1].set_title(f'GT Lidar Points\n{len(lidar_gt_points)} points', 
                        fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('X (m)')
    axes[1, 1].set_ylabel('Y (m)')
    axes[1, 1].set_xlim(-50, 50)
    axes[1, 1].set_ylim(-50, 50)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Predicted Lidar Points (포인트 클라우드)
    if len(lidar_pred_points) > 0:
        scatter5 = axes[1, 2].scatter(lidar_pred_points[:, 0], lidar_pred_points[:, 1], 
                                     c=lidar_pred_points[:, 3], cmap='hot', s=1, alpha=0.6)
        plt.colorbar(scatter5, ax=axes[1, 2], shrink=0.8)
    axes[1, 2].set_title(f'Predicted Lidar Points\n{len(lidar_pred_points)} points', 
                        fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('X (m)')
    axes[1, 2].set_ylabel('Y (m)')
    axes[1, 2].set_xlim(-50, 50)
    axes[1, 2].set_ylim(-50, 50)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'inference_result_{sample_idx}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 추가: 히트맵 비교도 별도로 저장
    if lidar_gt_img is not None:
        visualize_heatmaps(radar_img, lidar_gt_img, lidar_pred_img, sample_idx, output_dir)
    
    # 차이 맵 생성 (GT가 있는 경우에만)
    if lidar_gt_img is not None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        diff = np.abs(lidar_gt_img - lidar_pred_img)
        im = ax.imshow(diff, cmap='coolwarm', origin='lower')
        ax.set_title(f'Absolute Difference (Sample {sample_idx})', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.grid(True, alpha=0.3)
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'difference_map_{sample_idx}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 품질 메트릭 계산
        mse = np.mean((lidar_gt_img - lidar_pred_img) ** 2)
        mae = np.mean(np.abs(lidar_gt_img - lidar_pred_img))
        
        # 상관계수 계산
        if lidar_gt_img.std() > 0 and lidar_pred_img.std() > 0:
            correlation = np.corrcoef(lidar_gt_img.flatten(), lidar_pred_img.flatten())[0, 1]
        else:
            correlation = 0
        
        # 바이너리 IoU 계산
        gt_threshold = max(lidar_gt_img.max() * 0.1, 0.01) if lidar_gt_img.max() > 0 else 0.01
        pred_threshold = max(lidar_pred_img.max() * 0.1, 0.01) if lidar_pred_img.max() > 0 else 0.01
        
        gt_binary = lidar_gt_img > gt_threshold
        pred_binary = lidar_pred_img > pred_threshold
        intersection = (gt_binary & pred_binary).sum()
        union = (gt_binary | pred_binary).sum()
        iou = intersection / union if union > 0 else 0
        
        log(f"Quality metrics - MSE: {mse:.6f}, MAE: {mae:.6f}, Corr: {correlation:.4f}, IoU: {iou:.4f}")
        return mse, mae, correlation, iou
    else:
        log("No GT available for metric calculation")
        return None, None, None, None

def visualize_heatmaps(radar_bev, lidar_gt, lidar_pred, sample_idx, output_dir):
    """BEV를 직접 히트맵으로 시각화 (추가 시각화)"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Radar BEV 히트맵
    im1 = axes[0].imshow(radar_bev, cmap='viridis', origin='lower')
    axes[0].set_title('Radar BEV (Input)')
    plt.colorbar(im1, ax=axes[0])
    
    # GT Lidar BEV 히트맵
    if lidar_gt is not None:
        im2 = axes[1].imshow(lidar_gt, cmap='hot', origin='lower')
        axes[1].set_title('GT Lidar BEV')
        plt.colorbar(im2, ax=axes[1])
    else:
        axes[1].set_title('No GT Available')
        axes[1].axis('off')
    
    # Predicted Lidar BEV 히트맵
    im3 = axes[2].imshow(lidar_pred, cmap='hot', origin='lower')
    axes[2].set_title('Predicted Lidar BEV')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'heatmap_{sample_idx}.png'), dpi=300)
    plt.close()

def save_pointclouds_as_ply(lidar_gt_points, lidar_pred_points, sample_idx, output_dir):
    """포인트 클라우드를 PLY 파일로 저장 (Open3D 사용)"""
    try:
        import open3d as o3d
        
        # GT 포인트 클라우드 저장
        if len(lidar_gt_points) > 0:
            gt_pcd = o3d.geometry.PointCloud()
            gt_pcd.points = o3d.utility.Vector3dVector(lidar_gt_points[:, :3])  # x, y, z
            if lidar_gt_points.shape[1] > 3:
                # intensity를 색상으로 변환
                intensities = lidar_gt_points[:, 3]
                colors = plt.cm.hot(intensities / intensities.max())[:, :3]
                gt_pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(os.path.join(output_dir, f'gt_lidar_sample_{sample_idx}.ply'), gt_pcd)
        
        # Predicted 포인트 클라우드 저장
        if len(lidar_pred_points) > 0:
            pred_pcd = o3d.geometry.PointCloud()
            pred_pcd.points = o3d.utility.Vector3dVector(lidar_pred_points[:, :3])  # x, y, z
            if lidar_pred_points.shape[1] > 3:
                # intensity를 색상으로 변환
                intensities = lidar_pred_points[:, 3]
                colors = plt.cm.hot(intensities / intensities.max())[:, :3]
                pred_pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(os.path.join(output_dir, f'pred_lidar_sample_{sample_idx}.ply'), pred_pcd)
        
        log(f"Saved point clouds as PLY files for sample {sample_idx}")
        
    except ImportError:
        log("Open3D not available, skipping PLY file generation")

@torch.no_grad()
def inference_single_sample_direct(H, sampler, ae_radar, ae_lidar, generator_lidar, radar_points, lidar_points, device):
    """미리 추출된 points로 직접 inference 수행"""
    
    if radar_points is None:
        log("No radar points provided")
        return None
    
    # points 차원 확인 및 보정
    if radar_points.dim() == 1:
        # 1D array인 경우 reshape 시도
        if len(radar_points) % 5 == 0:  # radar: x, y, z, rcs, v_comp
            radar_points = radar_points.reshape(-1, 5)
        elif len(radar_points) % 6 == 0:  # 혹은 6차원
            radar_points = radar_points.reshape(-1, 6)
        else:
            log(f"Cannot reshape radar points with length {len(radar_points)}")
            return None
    
    radar_points = radar_points.to(device)
    
    # img_size 처리 - 리스트인 경우 첫 번째 값 사용하거나 기본값 사용
    radar_img_size = H.radar_config.data.img_size
    if isinstance(radar_img_size, (list, tuple)):
        radar_img_size = radar_img_size[0] if len(radar_img_size) > 0 else 320
    
    radar_bev = points_to_bev(radar_points, radar_img_size).to(device)
    
    log(f"Original radar points: {radar_points.shape}")
    log(f"Converted radar BEV: {radar_bev.shape}")
    log(f"Radar BEV range: [{radar_bev.min():.3f}, {radar_bev.max():.3f}]")
    
    # 2. Lidar points 처리 (GT 비교용)
    lidar_bev = None
    if lidar_points is not None:
        # lidar points 차원 확인 및 보정
        if lidar_points.dim() == 1:
            if len(lidar_points) % 4 == 0:  # lidar: x, y, z, intensity
                lidar_points = lidar_points.reshape(-1, 4)
            elif len(lidar_points) % 5 == 0:  # x, y, z, intensity, ring
                lidar_points = lidar_points.reshape(-1, 5)
            elif len(lidar_points) % 6 == 0:  # x, y, z, intensity, ring, time
                lidar_points = lidar_points.reshape(-1, 6)
            else:
                log(f"Cannot reshape lidar points with length {len(lidar_points)}")
                lidar_points = None
        
        if lidar_points is not None:
            lidar_points = lidar_points.to(device)
            
            # lidar img_size 처리
            lidar_img_size = H.lidar_config.data.img_size
            if isinstance(lidar_img_size, (list, tuple)):
                lidar_img_size = lidar_img_size[0] if len(lidar_img_size) > 0 else 320
            
            lidar_bev = points_to_bev(lidar_points, lidar_img_size).to(device)
            log(f"Original lidar points: {lidar_points.shape}")
            log(f"Converted lidar BEV: {lidar_bev.shape}")
            log(f"Lidar BEV range: [{lidar_bev.min():.3f}, {lidar_bev.max():.3f}]")
    
    # 3. Radar BEV 인코딩하여 context 생성
    radar_latents = ae_radar.encoder(radar_bev)
    radar_quant, _, _ = ae_radar.quantize(radar_latents)
    
    B, C, H_dim, W_dim = radar_quant.shape
    context = radar_quant.permute(0, 2, 3, 1).contiguous().view(B, H_dim*W_dim, C)
    
    log(f"Context shape: {context.shape}")
    
    # 4. GT 코드 생성 (있는 경우)
    lidar_gt_codes = None
    gt_lidar_reconstructed = None
    if lidar_bev is not None:
        lidar_latents = ae_lidar.encoder(lidar_bev)
        _, _, lidar_quant_stats = ae_lidar.quantize(lidar_latents)
        lidar_gt_codes = lidar_quant_stats["min_encoding_indices"].view(B, -1)
        gt_lidar_reconstructed = reconstruct_from_codes(H, sampler, lidar_gt_codes, generator_lidar)
        log(f"GT codes shape: {lidar_gt_codes.shape}, unique codes: {len(torch.unique(lidar_gt_codes))}")
    
    # 5. Sampler로 lidar codes 예측
    start_time = time.time()
    predicted_codes = sampler.sample(
        context=context,
        sample_steps=H.diffusion.sampling_steps,
        temp=H.diffusion.sampling_temp
    )
    inference_time = time.time() - start_time
    
    log(f"Predicted codes shape: {predicted_codes.shape}, unique codes: {len(torch.unique(predicted_codes))}")
    
    # 6. 예측된 codes로부터 lidar BEV 재구성
    predicted_lidar = reconstruct_from_codes(H, sampler, predicted_codes, generator_lidar)
    
    log(f"Generated Lidar shape: {predicted_lidar.shape}")
    log(f"Generated Lidar range: [{predicted_lidar.min():.3f}, {predicted_lidar.max():.3f}]")
    
    return {
        'radar_points': radar_points,
        'radar_bev': radar_bev,
        'lidar_points': lidar_points,
        'lidar_bev': lidar_bev,
        'lidar_gt_reconstructed': gt_lidar_reconstructed,
        'lidar_predicted': predicted_lidar,
        'inference_time': inference_time
    }


@torch.no_grad()
def inference_single_sample_from_points(H, sampler, ae_radar, ae_lidar, generator_lidar, radar_data, lidar_data, device):
    """원본 points에서 시작하는 inference"""
    
    # 1. Radar points 추출 및 BEV 변환
    radar_points = extract_points_from_data(radar_data, H)
    if radar_points is None:
        log("No radar points found in data")
        return None
    
    # points 차원 확인 및 보정
    if radar_points.dim() == 1:
        # 1D array인 경우 reshape 시도
        if len(radar_points) % 5 == 0:  # radar: x, y, z, rcs, v_comp
            radar_points = radar_points.reshape(-1, 5)
        elif len(radar_points) % 6 == 0:  # 혹은 6차원
            radar_points = radar_points.reshape(-1, 6)
        else:
            log(f"Cannot reshape radar points with length {len(radar_points)}")
            return None
    
    radar_points = radar_points.to(device)
    radar_bev = points_to_bev(radar_points, H.radar_config.data.img_size).to(device)
    
    log(f"Original radar points: {radar_points.shape}")
    log(f"Converted radar BEV: {radar_bev.shape}")
    log(f"Radar BEV range: [{radar_bev.min():.3f}, {radar_bev.max():.3f}]")
    
    # 2. Lidar points 추출 및 BEV 변환 (GT 비교용)
    lidar_points = None
    lidar_bev = None
    if lidar_data is not None:
        lidar_points = extract_points_from_data(lidar_data, H)
        if lidar_points is not None:
            # lidar points 차원 확인 및 보정
            if lidar_points.dim() == 1:
                if len(lidar_points) % 4 == 0:  # lidar: x, y, z, intensity
                    lidar_points = lidar_points.reshape(-1, 4)
                elif len(lidar_points) % 5 == 0:  # x, y, z, intensity, ring
                    lidar_points = lidar_points.reshape(-1, 5)
                elif len(lidar_points) % 6 == 0:  # x, y, z, intensity, ring, time
                    lidar_points = lidar_points.reshape(-1, 6)
                else:
                    log(f"Cannot reshape lidar points with length {len(lidar_points)}")
                    lidar_points = None
            
            if lidar_points is not None:
                lidar_points = lidar_points.to(device)
                lidar_bev = points_to_bev(lidar_points, H.lidar_config.data.img_size).to(device)
                log(f"Original lidar points: {lidar_points.shape}")
                log(f"Converted lidar BEV: {lidar_bev.shape}")
                log(f"Lidar BEV range: [{lidar_bev.min():.3f}, {lidar_bev.max():.3f}]")
    
    # 3. Radar BEV 인코딩하여 context 생성
    radar_latents = ae_radar.encoder(radar_bev)
    radar_quant, _, _ = ae_radar.quantize(radar_latents)
    
    B, C, H_dim, W_dim = radar_quant.shape
    context = radar_quant.permute(0, 2, 3, 1).contiguous().view(B, H_dim*W_dim, C)
    
    log(f"Context shape: {context.shape}")
    
    # 4. GT 코드 생성 (있는 경우)
    lidar_gt_codes = None
    gt_lidar_reconstructed = None
    if lidar_bev is not None:
        lidar_latents = ae_lidar.encoder(lidar_bev)
        _, _, lidar_quant_stats = ae_lidar.quantize(lidar_latents)
        lidar_gt_codes = lidar_quant_stats["min_encoding_indices"].view(B, -1)
        gt_lidar_reconstructed = reconstruct_from_codes(H, sampler, lidar_gt_codes, generator_lidar)
        log(f"GT codes shape: {lidar_gt_codes.shape}, unique codes: {len(torch.unique(lidar_gt_codes))}")
    
    # 5. Sampler로 lidar codes 예측
    start_time = time.time()
    predicted_codes = sampler.sample(
        context=context,
        sample_steps=H.diffusion.sampling_steps,
        temp=H.diffusion.sampling_temp
    )
    inference_time = time.time() - start_time
    
    log(f"Predicted codes shape: {predicted_codes.shape}, unique codes: {len(torch.unique(predicted_codes))}")
    
    # 6. 예측된 codes로부터 lidar BEV 재구성
    predicted_lidar = reconstruct_from_codes(H, sampler, predicted_codes, generator_lidar)
    
    log(f"Generated Lidar shape: {predicted_lidar.shape}")
    log(f"Generated Lidar range: [{predicted_lidar.min():.3f}, {predicted_lidar.max():.3f}]")
    
    return {
        'radar_points': radar_points,
        'radar_bev': radar_bev,
        'lidar_points': lidar_points,
        'lidar_bev': lidar_bev,
        'lidar_gt_reconstructed': gt_lidar_reconstructed,
        'lidar_predicted': predicted_lidar,
        'inference_time': inference_time
    }

def bev_to_points_lidar(bev_image, intensity_threshold=0.01, range_limit=50.0):
    """BEV 이미지를 LiDAR 포인트로 변환 (3D 포인트 클라우드 형태)"""
    if isinstance(bev_image, torch.Tensor):
        bev_image = bev_image.cpu().numpy()
    
    # 2D BEV에서 임계값보다 큰 값들의 위치 찾기
    if bev_image.ndim == 4:  # (1, 1, H, W)
        bev_image = bev_image[0, 0]
    elif bev_image.ndim == 3:  # (1, H, W)
        bev_image = bev_image[0]
    
    y_indices, x_indices = np.where(bev_image > intensity_threshold)
    
    if len(x_indices) == 0:
        return np.zeros((0, 4))  # x, y, z, intensity
    
    # 픽셀 좌표를 실제 좌표로 변환
    height, width = bev_image.shape
    x_coords = (x_indices / width) * (2 * range_limit) - range_limit  # [-50, 50]
    y_coords = (y_indices / height) * (2 * range_limit) - range_limit  # [-50, 50]
    
    # Z 좌표는 0으로 설정 (BEV이므로)
    z_coords = np.zeros_like(x_coords)
    
    # 강도값 추출
    intensities = bev_image[y_indices, x_indices]
    
    # 포인트 클라우드 형태로 결합: [x, y, z, intensity]
    points = np.stack([x_coords, y_coords, z_coords, intensities], axis=1)
    return points


def main(argv):
    H = FLAGS.config
    H.radar_config = FLAGS.radar_config
    H.lidar_config = FLAGS.lidar_config
    
    # GPU 설정
    device = setup_device(H)

    # data_root 설정 - H.data.data_root가 없으면 기본값 사용
    if not hasattr(H, 'data') or not hasattr(H.data, 'data_root'):
        if not hasattr(H, 'data'):
            H.data = ml_collections.ConfigDict()
        H.data.data_root = '/data1/nuScenes/'
        log(f"[main] Set default data_root: {H.data.data_root}")
    else:
        log(f"[main] Using existing data_root: {H.data.data_root}")

    # 출력 디렉토리 생성
    output_dir = FLAGS.output_dir
    os.makedirs(output_dir, exist_ok=True)

    log("Loading models...")
    
    # UltraLiDAR 모델들 로드
    ae_radar = load_pretrained_ultralidar(
        H.model_paths.radar_vqgan_path,
        'radar',
        device
    )
    ae_lidar = load_pretrained_ultralidar(
        H.model_paths.lidar_vqgan_path,
        'lidar', 
        device
    )
    
    # Generator 로드 (lidar 재구성용)
    generator_lidar = load_pretrained_ultralidar(
        H.model_paths.lidar_vqgan_path,
        'lidar',
        device
    )
    
    # 모델들을 eval 모드로 설정
    ae_radar.eval()
    ae_lidar.eval()
    generator_lidar.eval()
    
    # Lidar embedding weight 추출
    try:
        lidar_embedding_weight = None
        
        # UltraLiDAR 모델의 get_embedding_weight 메서드 사용
        if hasattr(ae_lidar, 'get_embedding_weight'):
            lidar_embedding_weight = ae_lidar.get_embedding_weight()
            log(f"Found embedding weight using get_embedding_weight(), shape: {lidar_embedding_weight.shape}")
        elif hasattr(ae_lidar, 'vector_quantizer') and hasattr(ae_lidar.vector_quantizer, 'embedding'):
            lidar_embedding_weight = ae_lidar.vector_quantizer.embedding.weight
            log(f"Found embedding weight from ae_lidar.vector_quantizer.embedding.weight, shape: {lidar_embedding_weight.shape}")
        elif hasattr(ae_lidar, 'quantize') and hasattr(ae_lidar.quantize, 'embedding'):
            lidar_embedding_weight = ae_lidar.quantize.embedding.weight
            log(f"Found embedding weight from ae_lidar.quantize.embedding.weight, shape: {lidar_embedding_weight.shape}")
        elif hasattr(ae_lidar, 'quantizer') and hasattr(ae_lidar.quantizer, 'embedding'):
            lidar_embedding_weight = ae_lidar.quantizer.embedding.weight
            log(f"Found embedding weight from ae_lidar.quantizer.embedding.weight, shape: {lidar_embedding_weight.shape}")
        else:
            # 모델의 모든 파라미터를 확인해서 embedding weight 찾기
            for name, param in ae_lidar.named_parameters():
                if 'embedding.weight' in name and param.shape[0] == H.lidar_config.model.codebook_size:
                    lidar_embedding_weight = param
                    log(f"Found embedding weight from {name}, shape: {lidar_embedding_weight.shape}")
                    break
        
        if lidar_embedding_weight is None:
            raise ValueError("Could not find embedding weight in the loaded model")
            
    except Exception as e:
        log(f"Error extracting embedding weight: {e}")
        return
    
    # Sampler 생성 및 학습된 가중치 로드
    H.ct_config = H.lidar_config  # 호환성을 위해
    
    sampler = get_sampler(H, lidar_embedding_weight.to(device)).to(device)
    
    # 학습된 sampler 가중치 로드
    log(f"Loading trained sampler from {FLAGS.model_path}")
    checkpoint = torch.load(FLAGS.model_path, map_location=device)
    sampler.load_state_dict(checkpoint, strict=True)
    sampler.eval()
    
    log("Models loaded successfully!")
    
    # mmdet3d 방식으로 데이터셋 로드
    log("Loading datasets using mmdet3d...")
    radar_dataset = load_dataset_mmdet3d(FLAGS.radar_ultralidar_config, H.data.data_root)
    lidar_dataset = load_dataset_mmdet3d(FLAGS.lidar_ultralidar_config, H.data.data_root)
    
    log(f"Loaded radar dataset: {len(radar_dataset)} samples")
    log(f"Loaded lidar dataset: {len(lidar_dataset)} samples")
    
    # 첫 번째 샘플 확인 및 디버깅
    if len(radar_dataset) > 0:
        sample_data = radar_dataset[0]
        log(f"Sample data keys: {sample_data.keys()}")
        
        radar_points = extract_points_from_data(sample_data, H)
        if radar_points is not None:
            log(f"Sample radar points shape: {radar_points.shape}")
    
    log(f"Starting inference on {min(FLAGS.num_samples, len(radar_dataset))} samples...")
    
    # Inference 수행 - 미리 points를 추출해서 사용
    total_inference_time = 0
    all_metrics = []
    
    num_success = 0
    max_samples = min(FLAGS.num_samples, len(radar_dataset))
    
    for i in range(max_samples):
        log(f"Processing sample {i+1}/{max_samples}")
        
        try:
            # 데이터 로드 및 points 미리 추출
            radar_data = radar_dataset[i]
            
            # lidar 데이터 로드 시 에러 처리
            lidar_data = None
            if i < len(lidar_dataset):
                try:
                    lidar_data = lidar_dataset[i]
                    log(f"Successfully loaded lidar data for sample {i+1}")
                except Exception as lidar_error:
                    log(f"Failed to load lidar data for sample {i+1}: {lidar_error}")
                    log("Continuing with radar-only inference...")
                    lidar_data = None
            
            # points 미리 추출
            radar_points = extract_points_from_data(radar_data, H)
            if radar_points is None:
                log(f"Skipping sample {i+1}: no radar points found")
                continue
                
            lidar_points = None
            if lidar_data is not None:
                lidar_points = extract_points_from_data(lidar_data, H)
            
            log(f"Successfully extracted points - Radar: {radar_points.shape}, Lidar: {lidar_points.shape if lidar_points is not None else 'None'}")
            
            # 추출된 points로 직접 inference 수행 (올바른 함수 호출)
            results = inference_single_sample_direct(
                H, sampler, ae_radar, ae_lidar, generator_lidar, 
                radar_points, lidar_points, device
            )

            if results is None:
                log(f"Skipping sample {i+1} due to inference failure")
                continue

            total_inference_time += results['inference_time']

            # 결과 시각화 및 저장 (points와 BEV 모두)
            mse, mae, corr, iou = visualize_results_with_points(
                results['radar_points'],
                results['radar_bev'],
                results['lidar_points'],
                results['lidar_bev'],
                results['lidar_predicted'],
                num_success+1,
                output_dir
            )

            if mse is not None:
                all_metrics.append((mse, mae, corr, iou))

            # 추가 정보 저장
            save_data = {
                'radar_bev': results['radar_bev'].cpu().numpy(),
                'lidar_predicted': results['lidar_predicted'].cpu().numpy(),
                'inference_time': results['inference_time']
            }

            if results['radar_points'] is not None:
                save_data['radar_points'] = results['radar_points'].cpu().numpy()
            if results['lidar_points'] is not None:
                save_data['lidar_points'] = results['lidar_points'].cpu().numpy()
            if results['lidar_bev'] is not None:
                save_data['lidar_bev'] = results['lidar_bev'].cpu().numpy()
            if results['lidar_gt_reconstructed'] is not None:
                save_data['lidar_gt_reconstructed'] = results['lidar_gt_reconstructed'].cpu().numpy()

            if mse is not None:
                save_data['metrics'] = {'mse': mse, 'mae': mae, 'correlation': corr, 'iou': iou}

            np.savez(
                os.path.join(output_dir, f'sample_{num_success+1}_data.npz'),
                **save_data
            )

            num_success += 1
            log(f"Successfully processed sample {i+1} (total success: {num_success})")
            
        except Exception as e:
            log(f"Error processing sample {i+1}: {e}")
            import traceback
            log(f"Traceback: {traceback.format_exc()}")
            continue
    
    # 종합 성능 요약
    if len(all_metrics) > 0:
        avg_inference_time = total_inference_time / len(all_metrics)
        avg_mse = np.mean([m[0] for m in all_metrics])
        avg_mae = np.mean([m[1] for m in all_metrics])
        avg_corr = np.mean([m[2] for m in all_metrics])
        avg_iou = np.mean([m[3] for m in all_metrics])
        
        log(f"Inference completed!")
        log(f"Processed {num_success} samples successfully")
        log(f"Average inference time: {avg_inference_time:.4f} seconds")
        log(f"Average metrics - MSE: {avg_mse:.6f}, MAE: {avg_mae:.6f}, Corr: {avg_corr:.4f}, IoU: {avg_iou:.4f}")
    else:
        log(f"Inference completed! (No metrics calculated due to missing GT)")
        log(f"Processed {num_success} samples successfully")
    
    log(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    app.run(main)