import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from captum.attr import LayerGradCam
from torchinfo import summary
from .train import evaluate_detailed_metrics

def plot_separate_validation_figures(results, label_prefix='Param', name_suffix='sweep'):
# def plot_separate_sweep_figures(results, label_prefix='Param', name_suffix='sweep'):
    os.makedirs('images', exist_ok=True)
    
    # Figure 1: Training Loss
    plt.figure(figsize=(8, 5))
    for key, history in results.items():
        plt.plot(history['train_loss'], label=f'{label_prefix}: {key}', linewidth=2)
    plt.title(f'Training Loss ({label_prefix} Sweep)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'images/{name_suffix}_train_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: Validation Loss
    plt.figure(figsize=(8, 5))
    for key, history in results.items():
        plt.plot(history['val_loss'], label=f'{label_prefix}: {key}', linewidth=2)
    plt.title(f'Validation Loss ({label_prefix} Sweep)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'images/{name_suffix}_val_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Validation Accuracy
    plt.figure(figsize=(8, 5))
    for key, history in results.items():
        plt.plot(history['val_acc'], label=f'{label_prefix}: {key}', linewidth=2)
    plt.title(f'Validation Accuracy ({label_prefix} Sweep)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'images/{name_suffix}_val_acc.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_sweep_metrics_table(results, param_name='Learning Rate'):
    table_data = []
    for param, history in results.items():
        best_epoch_idx = np.argmin(history['val_loss'])
        table_data.append({
            param_name: param,
            'Best Epoch': best_epoch_idx + 1,
            'Val Loss': f"{history['val_loss'][best_epoch_idx]:.4f}",
            'Val Class Error': f"{history['val_class_error'][best_epoch_idx]:.4f}",
            'Sensitivity': f"{history['val_sensitivity'][best_epoch_idx]:.4f}",
            'Specificity': f"{history['val_specificity'][best_epoch_idx]:.4f}"
        })
    return pd.DataFrame(table_data)

def create_architecture_comparison_table(models, test_loader, device, batch_size):
    comparison_data = []
    for name, model in models.items():
        metrics = evaluate_detailed_metrics(model, test_loader, device)
        model_summary = summary(model, input_size=(batch_size, 1, 64, 64), verbose=0)
        
        comparison_data.append({
            'Architecture': name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Classification Error': f"{metrics['classification_error']:.4f}",
            'Sensitivity': f"{metrics['sensitivity']:.4f}",
            'Specificity': f"{metrics['specificity']:.4f}",
            'Learnable Parameters': f"{model_summary.total_params:,}",
            'Mult-Adds (FLOPs)': f"{model_summary.total_mult_adds:,}"
        })
    return pd.DataFrame(comparison_data)

def get_conv_from_block(block):
    for m in block.modules():
        if isinstance(m, torch.nn.Conv2d):
            return m
    raise ValueError("No Conv2d found in this block")

def generate_and_save_gradcam(models, sample_images, sample_labels, device, save_dir='images/gradcam'):
    os.makedirs(save_dir, exist_ok=True)
    
    cam_b1 = LayerGradCam(models['B1_AlexNet'], get_conv_from_block(models['B1_AlexNet'].features[4]))
    cam_b2 = LayerGradCam(models['B2_VGG'], get_conv_from_block(models['B2_VGG'].features[6]))
    cam_b3 = LayerGradCam(models['B3_ResNet'], models['B3_ResNet'].layer3)

    cams = [cam_b1, cam_b2, cam_b3]
    model_names = ['B1: AlexNet', 'B2: VGG', 'B3: ResNet']
    model_keys = list(models.keys())

    for i in range(len(sample_images)):
        img = sample_images[i:i+1].to(device)
        true_label = int(sample_labels[i].item())
        orig_img_np = img.squeeze(0).cpu().permute(1, 2, 0).numpy().squeeze()
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        axes[0].imshow(orig_img_np, cmap='gray')
        axes[0].set_title(f"Original X-Ray {i+1}\nTrue Label: {true_label}")
        axes[0].axis('off')
        
        for idx, (cam, name, key) in enumerate(zip(cams, model_names, model_keys), start=1):
            model = models[key]
            model.eval()
            with torch.no_grad():
                prob = torch.sigmoid(model(img)).item()
                pred = 1 if prob > 0.5 else 0
                
            attr = cam.attribute(img, target=None)
            attr_up = F.interpolate(attr, size=img.shape[2:], mode='bilinear', align_corners=False)
            
            hm = attr_up.squeeze().detach().cpu().numpy()
            if hm.max() > hm.min():
                hm = (hm - hm.min()) / (hm.max() - hm.min())
                
            axes[idx].imshow(orig_img_np, cmap='gray')
            axes[idx].imshow(hm, cmap='jet', alpha=0.4, interpolation='bilinear')
            axes[idx].set_title(f"{name}\nPred: {pred} (Prob: {prob:.2f})")
            axes[idx].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'sample_{i+1}_gradcam.png'), dpi=300, bbox_inches='tight')
        plt.close()

def evaluate_and_plot_roc(models, test_loader, device, save_filename='roc_curve_comparison.png'):
    os.makedirs('images', exist_ok=True)
    plt.figure(figsize=(8, 6))
    
    for name, model in models.items():
        model.eval()
        y_true, y_probs = [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                probs = torch.sigmoid(model(inputs.to(device)))
                y_true.extend(targets.cpu().numpy())
                y_probs.extend(probs.cpu().numpy())
                
        fpr, tpr, _ = roc_curve(np.array(y_true), np.array(y_probs))
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.3f})', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join('images', save_filename), dpi=300)
    plt.close()