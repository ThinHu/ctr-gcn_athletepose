import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from src.models.ctrgcn import CTRGCN_Model
from src.models.graph import GraphCOCO

def evaluate_and_plot_confusion_matrix(model_weights_path, val_loader, class_names):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")
    
    # 1. Initialize model and load best weights
    model = CTRGCN_Model(num_class=len(class_names), num_point=17, num_person=1, 
                         graph_class=GraphCOCO, in_channels=2).to(device)
    
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval() # Crucial: turns off dropout during inference
    
    all_preds = []
    all_labels = []
    
    print("Running inference over the validation set...")
    
    # 2. Gather all predictions and true labels
    with torch.no_grad():
        for data, labels in val_loader:
            data = data.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            # Move data back to CPU for Scikit-Learn
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # 3. Compute Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Normalize the confusion matrix to show percentages instead of raw counts
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 4. Plotting
    plt.figure(figsize=(14, 10)) # Large figure to fit 18 classes cleanly
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Accuracy Proportion'})
    
    plt.title('Normalized Confusion Matrix: AthletePose3D (CTR-GCN)', fontsize=16, pad=20)
    plt.ylabel('True Action', fontsize=12)
    plt.xlabel('Predicted Action', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # 5. Print a detailed text report (Precision, Recall, F1-Score per class)
    print("\n--- Detailed Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))