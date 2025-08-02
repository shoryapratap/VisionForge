import argparse
import os
from ultralytics import YOLO
from datetime import datetime

# Default Training Hyperparameters
DEFAULT_EPOCHS = 50
DEFAULT_OPTIMIZER = "AdamW"
DEFAULT_MOSAIC = 0.1
DEFAULT_MOMENTUM = 0.937
DEFAULT_LR0 = 0.001
DEFAULT_LRF = 0.0001
DEFAULT_BATCH = 16
DEFAULT_IMAGE_SIZE = 640
DEFAULT_MODEL = "yolov8s.pt"
DEFAULT_DATA = "data/yolo_params.yaml"
DEFAULT_PROJECT = "runs"
DEFAULT_NAME = "detect"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train YOLOv8 model with custom parameters")

    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help='Pretrained model path')
    parser.add_argument('--data', type=str, default=DEFAULT_DATA, help='Path to YAML data config')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, default=DEFAULT_OPTIMIZER, help='Optimizer type')
    parser.add_argument('--mosaic', type=float, default=DEFAULT_MOSAIC, help='Mosaic augmentation factor')
    parser.add_argument('--momentum', type=float, default=DEFAULT_MOMENTUM, help='Momentum for SGD')
    parser.add_argument('--lr0', type=float, default=DEFAULT_LR0, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=DEFAULT_LRF, help='Final learning rate')
    parser.add_argument('--batch', type=int, default=DEFAULT_BATCH, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=DEFAULT_IMAGE_SIZE, help='Image size')
    parser.add_argument('--device', default=0, help='CUDA device ID (e.g., 0 or cpu)')
    parser.add_argument('--name', type=str, default=DEFAULT_NAME, help='Run name (inside project folder)')
    parser.add_argument('--project', type=str, default=DEFAULT_PROJECT, help='Project directory')

    return parser.parse_args()

def train_yolo(args):
    print(f"\n🚀 Starting YOLOv8 Training: {args.model}")
    print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Dataset: {args.data}")
    print(f"🧠 Optimizer: {args.optimizer}, LR0: {args.lr0}, LRF: {args.lrf}")
    print(f"🎯 Epochs: {args.epochs}, Batch: {args.batch}, Image Size: {args.imgsz}")
    print(f"🖥️ Device: {'CPU' if args.device == 'cpu' else f'CUDA:{args.device}'}\n")

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        mosaic=args.mosaic,
        momentum=args.momentum,
        device=args.device,
        project=args.project,
        name=args.name,
        verbose=True
    )
    
    print("\n✅ Training complete.")
    print(f"📦 Best weights saved to: {results.save_dir / 'weights' / 'best.pt'}")
    return results

if __name__ == "__main__":
    args = parse_arguments()
    train_yolo(args)