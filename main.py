import torch
import numpy as np
from src.data_loader import load_and_preprocess_data
from src.models import B1_AlexNet, B2_VGG, B3_ResNet
from src.train import train_model_comprehensive, evaluate_detailed_metrics
from src.utils import (
    plot_separate_validation_figures, 
    generate_sweep_metrics_table, 
    create_architecture_comparison_table,
    generate_and_save_gradcam,
    evaluate_and_plot_roc
)

def main():
    # Setup
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # ==========================================
    # B1: AlexNet Learning Rate Sweep
    # ==========================================
    print("--- Starting B1 (AlexNet) Learning Rate Sweep ---")
    lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    train_loader, val_loader, test_loader, X_test_t, y_test_t = load_and_preprocess_data(batch_size=128)

    print("\n--- Class Distribution Check ---")
    def print_dist(loader, name):
        targets = torch.cat([y for _, y in loader])
        healthy = (targets == 0).sum().item()
        pneumonia = (targets == 1).sum().item()
        print(f"{name} Set -> Healthy (0): {healthy} | Pneumonia (1): {pneumonia} (Ratio ~ 1:{pneumonia/healthy:.2f})")
    
    print_dist(train_loader, "Train")
    print_dist(val_loader, "Validation")
    print_dist(test_loader, "Test")
    
    b1_results = {}
    for lr in lrs:
        print(f"Training B1 with LR: {lr}")
        model_b1 = B1_AlexNet().to(device)
        b1_results[lr] = train_model_comprehensive(model_b1, train_loader, val_loader, lr, device)

    # Plot & Tabulate B1 Sweep
    plot_separate_validation_figures(b1_results, label_prefix='LR', name_suffix='b1_lr_sweep')
    b1_table = generate_sweep_metrics_table(b1_results, param_name='Learning Rate')
    print("\nB1 Sweep Results Table:")
    print(b1_table.to_string(index=False))

    best_lr = max(b1_results, key=lambda k: max(b1_results[k]['val_acc']))
    print(f"\n=> Best LR for B1 selected: {best_lr}\n")

    # Train Final B1 with Best LR
    final_model_b1 = B1_AlexNet().to(device)
    train_model_comprehensive(final_model_b1, train_loader, val_loader, best_lr, device)

    # ==========================================
    # B2: VGG Batch Size Sweep
    # ==========================================
    print("--- Starting B2 (VGG) Batch Size Sweep ---")
    batch_sizes = [16, 32, 64, 128, 256]
    b2_results = {}
    
    for bs in batch_sizes:
        print(f"Training VGG with Batch Size: {bs}")
        train_loader_bs, val_loader_bs, _, _, _ = load_and_preprocess_data(batch_size=bs)
        model_b2 = B2_VGG().to(device)
        b2_results[bs] = train_model_comprehensive(model_b2, train_loader_bs, val_loader_bs, best_lr, device)

    # Plot & Tabulate B2 Sweep
    plot_separate_validation_figures(b2_results, label_prefix='Batch Size', name_suffix='b2_bs_sweep')
    b2_table = generate_sweep_metrics_table(b2_results, param_name='Batch Size')
    print("\nB2 Sweep Results Table:")
    print(b2_table.to_string(index=False))

    best_bs = max(b2_results, key=lambda k: max(b2_results[k]['val_acc']))
    print(f"\n=> Best Batch Size for B2 selected: {best_bs}\n")

    # Train Final B2 with Best LR & Best BS
    train_loader_final, val_loader_final, test_loader_final, _, _ = load_and_preprocess_data(batch_size=best_bs)
    final_model_b2 = B2_VGG().to(device)
    train_model_comprehensive(final_model_b2, train_loader_final, val_loader_final, best_lr, device)

    # ==========================================
    # B3: ResNet Final Training
    # ==========================================
    print("--- Training Final B3 (ResNet) Model ---")
    final_model_b3 = B3_ResNet().to(device)
    b3_history = train_model_comprehensive(final_model_b3, train_loader_final, val_loader_final, best_lr, device)
    
    # Optional: Plot B3 history
    plot_separate_validation_figures({'ResNet': b3_history}, label_prefix='Model', name_suffix='b3_final')

    # ==========================================
    # Final Architecture Comparison (B1, B2, B3)
    # ==========================================
    print("\n--- Final Test Set Evaluation & Architecture Summary ---")
    models_dict = {
        'B1_AlexNet': final_model_b1,
        'B2_VGG': final_model_b2,
        'B3_ResNet': final_model_b3
    }
    final_comparison_df = create_architecture_comparison_table(models_dict, test_loader_final, device, best_bs)
    print(final_comparison_df.to_string(index=False))

    # ==========================================
    # B4: Grad-CAM & ROC Evaluations
    # ==========================================
    print("\n--- Generating Grad-CAM and ROC Visualizations ---")
    
    # Select 5 random pneumonia positive samples (target == 1)
    pneumonia_indices = torch.where(y_test_t == 1)[0]
    sample_indices = pneumonia_indices[torch.randperm(len(pneumonia_indices))[:5]]
    sample_images = X_test_t[sample_indices]
    sample_labels = y_test_t[sample_indices]

    generate_and_save_gradcam(models_dict, sample_images, sample_labels, device)
    evaluate_and_plot_roc(models_dict, test_loader_final, device)
    
    print("\nPipeline Complete! Check the 'images/' folder for all plots.")

    # Save B1 Table
    print("\nB1 Sweep Results Table:")
    print(b1_table.to_string(index=False))
    b1_table.to_csv('b1_results.csv', index=False)

    # Save B2 Table
    print("\nB2 Sweep Results Table:")
    print(b2_table.to_string(index=False))
    b2_table.to_csv('b2_results.csv', index=False)

    # Save Final Comparison Table
    print("\n--- Final Test Set Evaluation & Architecture Summary ---")
    print(final_comparison_df.to_string(index=False))
    final_comparison_df.to_csv('final_comparison.csv', index=False)

if __name__ == "__main__":
    main()