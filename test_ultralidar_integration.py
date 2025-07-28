#!/usr/bin/env python3
"""
UltraLiDAR radar 모델 로딩 테스트 스크립트
"""
import os
import sys
import torch
from pathlib import Path

# UltraLiDAR 경로 추가
ultralidar_path = "/home/byounggun/r2l/UltraLiDAR_nusc_waymo"
sys.path.insert(0, ultralidar_path)

def test_ultralidar_loading():
    """UltraLiDAR 모델 로딩 테스트"""
    print("🧪 Testing UltraLiDAR radar and lidar model loading...")
    
    # 모델 경로들
    radar_checkpoint_path = "/home/byounggun/r2l/UltraLiDAR_nusc_waymo/work_dirs/nusc_radar_debug/epoch_121.pth"
    lidar_checkpoint_path = "/home/byounggun/r2l/UltraLiDAR_nusc_waymo/work_dirs/nusc_stage2/epoch_200.pth"
    config_path = "/home/byounggun/r2l/UltraLiDAR_nusc_waymo/configs/ultralidar_nusc_radar_debug.py"
    
    results = {}
    
    # Radar 모델 테스트
    print("\n📡 Testing radar model...")
    if not os.path.exists(radar_checkpoint_path):
        print(f"❌ Radar checkpoint not found: {radar_checkpoint_path}")
        results['radar'] = False
    else:
        try:
            from utils.simple_ultralidar_wrapper import load_simple_ultralidar_model
            
            print(f"📂 Loading radar model from: {radar_checkpoint_path}")
            radar_model = load_simple_ultralidar_model(radar_checkpoint_path, 'radar')
            print(f"✅ Radar model loaded successfully!")
            
            # 간단한 forward 테스트
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            radar_model = radar_model.to(device)
            
            dummy_bev = torch.randn(1, 1, 640, 640).to(device)
            with torch.no_grad():
                radar_output = radar_model.forward(dummy_bev)
                print(f"✅ Radar model forward test passed: {radar_output['reconstructed'].shape}")
            
            results['radar'] = True
            
        except Exception as e:
            print(f"❌ Radar model test failed: {e}")
            results['radar'] = False
    
    # Lidar 모델 테스트
    print("\n� Testing lidar model...")
    if not os.path.exists(lidar_checkpoint_path):
        print(f"❌ Lidar checkpoint not found: {lidar_checkpoint_path}")
        results['lidar'] = False
    else:
        try:
            from utils.simple_ultralidar_wrapper import load_simple_ultralidar_model
            
            print(f"📂 Loading lidar model from: {lidar_checkpoint_path}")
            lidar_model = load_simple_ultralidar_model(lidar_checkpoint_path, 'lidar')
            print(f"✅ Lidar model loaded successfully!")
            
            # 간단한 forward 테스트
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            lidar_model = lidar_model.to(device)
            
            dummy_bev = torch.randn(1, 1, 640, 640).to(device)
            with torch.no_grad():
                lidar_output = lidar_model.forward(dummy_bev)
                print(f"✅ Lidar model forward test passed: {lidar_output['reconstructed'].shape}")
            
            results['lidar'] = True
            
        except Exception as e:
            print(f"❌ Lidar model test failed: {e}")
            results['lidar'] = False
    
    success = results.get('radar', False) and results.get('lidar', False)
    if success:
        print("\n🎉 Both radar and lidar models loaded successfully!")
    
    return success

def test_nuscenes_data():
    """NuScenes 데이터 로딩 테스트"""
    print("\n📊 Testing NuScenes radar data loading...")
    
    try:
        from utils.nuscenes_radar_dataloader import NuScenesRadarDataset
        
        data_root = "/data1/nuScenes/"
        ann_file = "/data1/nuScenes/nuscenes_infos_val_radar_tiny.pkl"
        
        # tiny 파일이 없으면 일반 파일 시도
        if not os.path.exists(ann_file):
            ann_file = "/data1/nuScenes/nuscenes_infos_val_radar.pkl"
        
        if not os.path.exists(ann_file):
            print(f"⚠️ Annotation file not found: {ann_file}")
            print("🔍 Available files in data_root:")
            if os.path.exists(data_root):
                files = [f for f in os.listdir(data_root) if f.endswith('.pkl')]
                print(f"   {files}")
            return False
        
        print(f"📂 Loading dataset from: {ann_file}")
        dataset = NuScenesRadarDataset(
            data_root=data_root,
            ann_file=ann_file,
            train=False,
            bev_size=640
        )
        
        print(f"✅ Dataset loaded with {len(dataset)} samples")
        
        if len(dataset) > 0:
            # 첫 번째 샘플 테스트
            print("🧪 Testing first sample...")
            sample = dataset[0]
            print(f"✅ Sample keys: {list(sample.keys())}")
            print(f"   - radar_bev shape: {sample['radar_bev'].shape}")
            print(f"   - radar_points shape: {sample['radar_points'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during data testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 Starting UltraLiDAR radar+lidar integration tests...")
    
    # UltraLiDAR 모델들 테스트
    model_test = test_ultralidar_loading()
    
    # NuScenes 데이터 테스트
    data_test = test_nuscenes_data()
    
    print(f"\n📋 Test Results:")
    print(f"   - UltraLiDAR Models: {'✅ PASS' if model_test else '❌ FAIL'}")
    print(f"   - NuScenes Data: {'✅ PASS' if data_test else '❌ FAIL'}")
    
    if model_test and data_test:
        print("\n🎉 All tests passed! Ready for radar→lidar training.")
        print("\n🚀 Next steps:")
        print("   1. Run: ./run_radar_to_lidar_training.sh")
        print("   2. Monitor training progress in logs/")
        print("   3. Check wandb for visualization")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        if not model_test:
            print("   - Check UltraLiDAR model paths and config files")
        if not data_test:
            print("   - Check NuScenes data paths and annotation files")

if __name__ == "__main__":
    main()
