import torch

def check_model_config():
    model_path = "/home/byounggun/r2l/x2ct-vqvae/logs/radar_to_lidar_sampler_radar_lidar/saved_models/absorbing_10000.th"
    
    print(f"Loading model from: {model_path}")
    
    try:
        # 모델 로드
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print("=" * 50)
        print("CHECKPOINT KEYS:")
        print("=" * 50)
        if isinstance(checkpoint, dict):
            for key in checkpoint.keys():
                print(f"- {key}")
        
        # config나 hyperparameters 확인
        if 'config' in checkpoint:
            print("\n" + "=" * 50)
            print("CONFIG FOUND:")
            print("=" * 50)
            print(checkpoint['config'])
        
        if 'H' in checkpoint:
            print("\n" + "=" * 50)
            print("H (HYPERPARAMETERS) FOUND:")
            print("=" * 50)
            print(checkpoint['H'])
            
        if 'hyperparams' in checkpoint:
            print("\n" + "=" * 50)
            print("HYPERPARAMS FOUND:")
            print("=" * 50)
            print(checkpoint['hyperparams'])
        
        # 모델 state dict 키만 확인 (값은 출력하지 않음)
        if 'model_state_dict' in checkpoint or isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            if hasattr(state_dict, 'keys'):
                print("\n" + "=" * 50)
                print("MODEL STATE DICT KEYS (first 20):")
                print("=" * 50)
                keys = list(state_dict.keys())
                for i, key in enumerate(keys[:20]):
                    print(f"  {i+1}. {key}")
                if len(keys) > 20:
                    print(f"  ... and {len(keys) - 20} more keys")
        
        print("\n" + "=" * 50)
        print("SUMMARY:")
        print("=" * 50)
        print(f"Checkpoint type: {type(checkpoint)}")
        if isinstance(checkpoint, dict):
            print(f"Number of top-level keys: {len(checkpoint.keys())}")
            print(f"Top-level keys: {list(checkpoint.keys())}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_model_config()