import pytest
import torch
import numpy as np
from models.cnn_models import CSI_1D_CNN, CSI_2D_CNN, CSI_CNN_Attention
from models.training import CSIDataset, prepare_dataloaders


def test_1d_cnn_forward():
    batch_size = 4
    n_subcarriers = 56
    seq_length = 200
    num_classes = 6

    model = CSI_1D_CNN(in_channels=n_subcarriers, num_classes=num_classes)
    x = torch.randn(batch_size, n_subcarriers, seq_length)
    out = model(x)
    assert out.shape == (batch_size, num_classes)


def test_2d_cnn_forward():
    batch_size = 4
    seq_length = 200
    n_subcarriers = 56
    num_classes = 6

    model = CSI_2D_CNN(in_channels=1, num_classes=num_classes)
    x = torch.randn(batch_size, 1, seq_length, n_subcarriers)
    out = model(x)
    assert out.shape == (batch_size, num_classes)


def test_attention_cnn_forward():
    batch_size = 4
    n_subcarriers = 56
    seq_length = 200
    num_classes = 6

    model = CSI_CNN_Attention(in_channels=n_subcarriers, num_classes=num_classes)
    x = torch.randn(batch_size, n_subcarriers, seq_length)
    out = model(x)
    assert out.shape == (batch_size, num_classes)


def test_dataloaders():
    X = np.random.randn(30, 100, 20)
    y = np.random.randint(0, 3, size=30)

    train_l, val_l, test_l = prepare_dataloaders(X, y, model_type="1d", batch_size=4, val_size=0.2, test_size=0.2)
    assert len(train_l) > 0
    assert len(val_l) > 0
    assert len(test_l) > 0
