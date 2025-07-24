import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from ultralytics import YOLO

model = YOLO('pretrained_model/yolov8s.pt')
results = model.train(results='config.yaml')