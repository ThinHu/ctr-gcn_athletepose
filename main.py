import os
import argparse
from torch.utils.data import DataLoader
from src.data.dataset import AthletePose3DDataset
from src.engine.trainer import train_and_evaluate
from src.utils.metrics import evaluate_and_plot_confusion_matrix

def main(args):
    # 1. Initialize Datasets
    train_dataset = AthletePose3DDataset(os.path.join(args.data_path, 'train'), is_train=True)
    val_dataset = AthletePose3DDataset(os.path.join(args.data_path, 'valid'), is_train=False)
    
    # 2. Initialize DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    num_classes = len(train_dataset.class_names)
    
    # 3. Train
    if not args.evaluate_only:
        train_and_evaluate(train_loader, val_loader, num_classes)
        
    # 4. Evaluate
    evaluate_and_plot_confusion_matrix('best_ctrgcn.pth', val_loader, train_dataset.class_names)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="Path to AthletePose3D dataset")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--evaluate_only', action='store_true')
    args = parser.parse_args()
    main(args)