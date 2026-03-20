import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import random
from src.models.ctrgcn import CTRGCN_Model 
from src.models.graph import GraphCOCO

def init_seed(seed=1):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def adjust_learning_rate(optimizer, epoch, base_lr, warm_up_epoch, step, lr_decay_rate):
    """Replicates the custom warmup and step decay from the official repo"""
    if epoch < warm_up_epoch:
        lr = base_lr * (epoch + 1) / warm_up_epoch
    else:
        lr = base_lr * (lr_decay_rate ** np.sum(epoch >= np.array(step)))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_and_evaluate(train_loader, val_loader, num_classes):
    init_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = CTRGCN_Model(num_class=num_classes, num_point=17, num_person=1, 
                         graph_class=GraphCOCO, in_channels=2, drop_out=0.5).to(device)
    
    print(f'# Model Parameters: {count_parameters(model):,}')
    
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Training Hyperparameters matching the repo style
    base_lr = 0.05
    warm_up_epoch = 10
    step = [30, 60, 80]
    lr_decay_rate = 0.1
    epochs = 100
    
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
    
    best_val_acc = 0.0
    best_epoch = 0
    patience = 20
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # --- Training Phase ---
        model.train()
        current_lr = adjust_learning_rate(optimizer, epoch, base_lr, warm_up_epoch, step, lr_decay_rate)
        
        train_loss = 0.0
        correct = 0
        total = 0
        
        timer = dict(dataloader=0.0, model=0.0)
        cur_time = time.time()
        
        for data, labels in train_loader:
            # Profile dataloader time
            timer['dataloader'] += time.time() - cur_time
            cur_time = time.time()
            
            data, labels = data.to(device, dtype=torch.float32), labels.to(device, dtype=torch.long)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Profile model compute time
            timer['model'] += time.time() - cur_time
            cur_time = time.time()
            
            train_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100. * correct / total
        avg_train_loss = train_loss / total
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device, dtype=torch.float32), labels.to(device, dtype=torch.long)
                output = model(data)
                loss = criterion(output, labels)
                
                val_loss += loss.item() * data.size(0)
                _, predicted = output.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / val_total
        epoch_time = time.time() - start_time
        
        # Calculate bottleneck percentages
        total_compute_time = sum(timer.values()) + 1e-8
        prop_data = int(round(timer['dataloader'] * 100 / total_compute_time))
        prop_model = int(round(timer['model'] * 100 / total_compute_time))
        
        # Clean Single-Line Logging
        print(f"Epoch [{epoch+1:03d}/{epochs}] | Time: {epoch_time:.1f}s | LR: {current_lr:.4f} | "
              # f"Data: {prop_data:02d}% Net: {prop_model:02d}% | "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # --- Checkpointing & Early Stopping ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_ctrgcn.pth')
            print(f"  -> New best validation accuracy! Saved to 'best_ctrgcn.pth'")
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs!")
            break
            
    print(f"\nTraining Complete. Best Validation Accuracy: {best_val_acc:.2f}% achieved at Epoch {best_epoch}.")

# Execute Training
train_and_evaluate(train_loader, val_loader, num_classes)
