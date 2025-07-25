import pandas as pd
import os
import cv2
from pathlib import Path
import yaml
import shutil
from sklearn.model_selection import train_test_split

class YOLOv8PoseConverter:
    def __init__(self, csv_path, images_dir, output_dir):
        self.csv_path = csv_path
        self.images_dir = images_dir
        self.output_dir = Path(output_dir)
        self.num_keypoints = 4  # 4 corners
        
    def create_directory_structure(self):
        """Create YOLOv8 directory structure"""
        dirs = [
            'train/images', 'train/labels',
            'val/images', 'val/labels'
        ]
        
        for dir_path in dirs:
            (self.output_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """Load the CSV data"""
        df = pd.read_csv(self.csv_path)
        return df
    
    def convert_to_yolo_pose_format(self, row, img_width, img_height):
        """
        Convert corner coordinates to YOLOv8 pose format
        Format: class x_center y_center width height x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4
        where v is visibility (2=visible, 1=occluded, 0=not labeled)
        """
        # Extract corner coordinates
        corners = [
            (row['x1'], row['y1']),
            (row['x2'], row['y2']),
            (row['x3'], row['y3']),
            (row['x4'], row['y4'])
        ]
        
        # Calculate bounding box from corners
        x_coords = [c[0] for c in corners]
        y_coords = [c[1] for c in corners]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Calculate center, width, height (normalized)
        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        # Class ID (assuming single class 'card')
        class_id = 0
        
        # Normalize keypoints and set visibility
        keypoints = []
        for x, y in corners:
            x_norm = x / img_width
            y_norm = y / img_height
            visibility = 2  # visible
            keypoints.extend([x_norm, y_norm, visibility])
        
        # Format: class x_center y_center width height + keypoints
        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        for kp in keypoints:
            yolo_line += f" {kp:.6f}"
        
        return yolo_line
    
    def get_image_dimensions(self, image_path):
        """Get image width and height"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        height, width = img.shape[:2]
        return width, height
    
    def convert_dataset(self, train_ratio=0.8, val_ratio=0.2):
        """Convert the entire dataset"""
        print("Loading data...")
        df = self.load_data()
        
        print("Creating directory structure...")
        self.create_directory_structure()
        
        # Split data into train/val only
        train_df, val_df = train_test_split(df, test_size=val_ratio, random_state=42)
        
        splits = {
            'train': train_df,
            'val': val_df
        }
        
        print("Converting datasets...")
        for split_name, split_df in splits.items():
            print(f"Processing {split_name} split ({len(split_df)} images)...")
            
            for idx, row in split_df.iterrows():
                # Get image path
                image_filename = os.path.basename(row['file_path'])
                image_path = Path(self.images_dir) / image_filename
                
                if not image_path.exists():
                    print(f"Warning: Image not found: {image_path}")
                    continue
                
                # Get image dimensions
                try:
                    img_width, img_height = self.get_image_dimensions(image_path)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    continue
                
                # Convert to YOLO format
                yolo_line = self.convert_to_yolo_pose_format(row, img_width, img_height)
                
                # Copy image to destination
                dest_image_path = self.output_dir / split_name / 'images' / image_filename
                shutil.copy2(image_path, dest_image_path)
                
                # Write label file
                label_filename = image_filename.replace('.jpg', '.txt').replace('.png', '.txt')
                label_path = self.output_dir / split_name / 'labels' / label_filename
                
                with open(label_path, 'w') as f:
                    f.write(yolo_line + '\n')
        
        # Create dataset configuration file
        self.create_dataset_config()
        
        print(f"Dataset conversion completed! Output directory: {self.output_dir}")
        print(f"Train: {len(splits['train'])} images")
        print(f"Val: {len(splits['val'])} images")
    
    def create_dataset_config(self):
        """Create YAML configuration file for YOLOv8"""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': 1,  # number of classes
            'names': ['card'],  # class names
            'kpt_shape': [4, 3],  # [num_keypoints, (x,y,visibility)]
            'flip_idx': []  # no symmetric keypoints for card corners
        }
        
        config_path = self.output_dir / 'dataset.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Dataset config saved to: {config_path}")

# Usage example
def main():
    # Update these paths according to your setup
    csv_path = "real_and_synthetic_corner_dataset/labels.csv"
    images_dir = "real_and_synthetic_corner_dataset/images"
    output_dir = "yolov8_pose_dataset"
    
    # Create converter and convert dataset
    converter = YOLOv8PoseConverter(csv_path, images_dir, output_dir)
    converter.convert_dataset()
    
    print("\nDataset conversion completed!")
    print("You can now train YOLOv8 pose model using:")
    print(f"yolo pose train data={output_dir}/dataset.yaml model=yolov8n-pose.pt epochs=100")

if __name__ == "__main__":
    main()