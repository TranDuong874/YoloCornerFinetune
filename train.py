from ultralytics import YOLO
import yaml
import torch
import albumentations as A
from ultralytics.data.augment import Albumentations
from ultralytics.data.augment import AlbumentationsTransform
from ultralytics.data.augment import register_augmentations

def get_custom_augmentations(cfg):
    transforms = []

    if cfg.get('gaussian_noise'):
        transforms.append(A.GaussNoise(
            var_limit=tuple(cfg['gaussian_noise']['var_limit']), 
            p=cfg['gaussian_noise']['prob']))

    if cfg.get('blur'):
        transforms.append(A.Blur(
            blur_limit=tuple(cfg['blur']['blur_limit']), 
            p=cfg['blur']['prob']))

    if cfg.get('motion_blur'):
        transforms.append(A.MotionBlur(
            blur_limit=tuple(cfg['motion_blur']['blur_limit']), 
            p=cfg['motion_blur']['prob']))

    if cfg.get('image_compression'):
        transforms.append(A.ImageCompression(
            quality_lower=cfg['image_compression']['lower'], 
            quality_upper=cfg['image_compression']['upper'], 
            p=cfg['image_compression']['prob']))

    return AlbumentationsTransform(A.Compose(transforms)) if transforms else None

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

    custom_aug = get_custom_augmentations(aug_config)
    register_augmentations(custom_aug)

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
        
        augmentations=custom_aug
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