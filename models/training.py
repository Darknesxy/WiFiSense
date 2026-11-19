"""
WiFiSense - Training Pipeline
Handles PyTorch DataLoader creation, stratified splitting, model training, early stopping, and checkpointing.
"""

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from .cnn_models import CSI_1D_CNN, CSI_2D_CNN, CSI_CNN_Attention


class CSIDataset(Dataset):
    """
    PyTorch Dataset for CSI Amplitude / Phase matrices.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, model_type: str = "1d"):
        """
        Args:
            X: Array of shape (N, time_steps, n_subcarriers)
            y: Array of shape (N,)
            model_type: '1d' -> (N, n_subcarriers, time_steps), '2d' -> (N, 1, time_steps, n_subcarriers)
        """
        self.y = torch.tensor(y, dtype=torch.long)
        self.model_type = model_type.lower()

        if self.model_type == "1d":
            # Transpose to (N, Channels=subcarriers, Length=time_steps)
            transposed = np.transpose(X, (0, 2, 1))
            self.X = torch.tensor(transposed, dtype=torch.float32)
        elif self.model_type == "2d":
            # Expand to (N, Channels=1, Height=time_steps, Width=subcarriers)
            expanded = np.expand_dims(X, axis=1)
            self.X = torch.tensor(expanded, dtype=torch.float32)
        else:
            self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def prepare_dataloaders(
    X: np.ndarray, 
    y: np.ndarray, 
    model_type: str = "1d",
    batch_size: int = 16, 
    val_size: float = 0.2, 
    test_size: float = 0.1,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates stratified Train, Validation, and Test DataLoaders.
    """
    # First split off test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Then split train and validation
    adjusted_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=adjusted_val_size, stratify=y_train_val, random_state=random_state
    )

    train_ds = CSIDataset(X_train, y_train, model_type=model_type)
    val_ds = CSIDataset(X_val, y_val, model_type=model_type)
    test_ds = CSIDataset(X_test, y_test, model_type=model_type)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_cnn_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 25,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[str] = None,
    save_path: Optional[str] = "models/best_csi_cnn.pth",
    patience: int = 7
) -> Dict[str, Any]:
    """
    Trains a CNN model with AdamW optimizer, CrossEntropy loss, and Early Stopping.

    Returns:
        history: Dictionary containing loss and accuracy curves.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_epoch": 0,
        "best_val_acc": 0.0
    }

    best_val_acc = 0.0
    patience_counter = 0

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        scheduler.step(epoch_val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            history["best_val_acc"] = best_val_acc
            history["best_epoch"] = epoch
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc*100:.2f}%")

        if patience_counter >= patience:
            print(f"[Early Stopping] Triggered at epoch {epoch}. Restoring best val acc: {best_val_acc*100:.2f}%")
            break

    return history
