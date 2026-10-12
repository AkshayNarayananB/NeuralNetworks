import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from .models import EarlyStopping

def evaluate_detailed_metrics(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    criterion = nn.BCEWithLogitsLoss()
    val_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    loss_val = val_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    class_error = 1.0 - acc
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0
        
    return {
        'val_loss': loss_val,
        'accuracy': acc,
        'classification_error': class_error,
        'sensitivity': sensitivity,
        'specificity': specificity
    }

def train_model_comprehensive(model, train_loader, val_loader, lr, device, max_epochs=20):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=4, min_delta=1e-4)
    
    history = {
        'train_loss': [], 'val_loss': [], 'val_acc': [], 
        'val_class_error': [], 'val_sensitivity': [], 'val_specificity': []
    }
    
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = train_loss / len(train_loader.dataset)
        val_metrics = evaluate_detailed_metrics(model, val_loader, device)
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(val_metrics['val_loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_class_error'].append(val_metrics['classification_error'])
        history['val_sensitivity'].append(val_metrics['sensitivity'])
        history['val_specificity'].append(val_metrics['specificity'])
        
        early_stopping(val_metrics['val_loss'])
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
            
    return history