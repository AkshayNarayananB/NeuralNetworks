import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_and_preprocess_data(filepath='data/chestxray_binary_64.npz', batch_size=128):
    data = np.load(filepath)
    
    X_train, y_train = data['train_images'], data['train_labels']
    X_val, y_val = data['val_images'], data['val_labels']
    X_test, y_test = data['test_images'], data['test_labels']
    
    # Scale to [0, 1]
    X_train, X_val, X_test = X_train / 255.0, X_val / 255.0, X_test / 255.0
    
    # Normalize using train mean and std
    train_mean = np.mean(X_train)
    train_std = np.std(X_train)
    
    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std
    
    # Convert to PyTorch tensors (N, C, H, W)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, X_test_t, y_test_t