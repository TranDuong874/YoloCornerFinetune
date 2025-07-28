from ultralytics import YOLO
import yaml
import torch

def load_config(config_path):
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_model(config_path):
    """Train YOLOv8 pose model with configuration from YAML"""
    config = load_config(config_path)
    
    training_config = config['Training']
    dataset_config = config['Dataset']
    aug_config = config['Augmentations']
    
    model = YOLO(training_config['model_path'])
    loss_config = training_config.get('loss', {})

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    results = model.train(
        data=dataset_config['data_yaml_path'],
        epochs=training_config['epochs'],
        imgsz=training_config['img_size'],
        batch=training_config['batch_size'],
        device=training_config['device'],
        project=training_config['project'],
        name=training_config['name'],
        save=True,
        save_period=training_config['save_period'],
        cache=training_config['cache'],
        augment=training_config['augment'],
        
        # Augmentation parameters
        mosaic=aug_config['mosaic'],
        mixup=aug_config['mixup'],
        degrees=aug_config['degrees'],
        translate=aug_config['translate'],
        scale=aug_config['scale'],
        shear=aug_config['shear'],
        perspective=aug_config['perspective'],
        flipud=aug_config['flipud'],
        fliplr=aug_config['fliplr'],
        hsv_h=aug_config['hsv_h'],
        hsv_s=aug_config['hsv_s'],
        hsv_v=aug_config['hsv_v'],

        # Loss config
        kpt=loss_config.get('kpt', 1.0),  # Default = 1.0
        box=loss_config.get('box', 7.5), # Default = 7.5
    )

    if hasattr(model, "trainer"):
        print("Training completed!")
        print(f"Results saved to: {model.trainer.save_dir}")
        if model.metrics:
            print(f"mAP50: {model.metrics.get('metrics/mAP50(B)', 'N/A')}")
    else:
        print("Training may have failed or was distributed.")

    
    print("Training completed!")
    print(f"Results saved to: {results.save_dir}")
    return model, results

if __name__ == "__main__":
    # Train with config file
    config_file = "training_config.yaml"
    model, results = train_model(config_file)