"""
WiFiSense - Model Evaluation & Comparison Module
Benchmarks PyTorch CNNs against baseline Machine Learning algorithms (Random Forest, SVM)
across accuracy, F1-score, latency (ms), and model footprint.
"""

import time
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .cnn_models import CSI_1D_CNN, CSI_2D_CNN
from .training import CSIDataset, train_cnn_model, prepare_dataloaders
from signal_processing.feature_extraction import extract_full_feature_vector


def evaluate_model(
    model: nn.Module, 
    test_loader: DataLoader, 
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates PyTorch model on a test set.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "predictions": all_preds.tolist(),
        "targets": all_targets.tolist()
    }


def benchmark_model_inference_latency(
    model: nn.Module, 
    input_sample: torch.Tensor, 
    n_runs: int = 100, 
    device: Optional[str] = None
) -> float:
    """
    Calculates average inference latency per sample in milliseconds.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)
    input_sample = input_sample.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_sample)

    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(input_sample)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    return float(np.mean(latencies))


def compare_all_models(
    X_amp: np.ndarray, 
    y: np.ndarray, 
    activities: List[str],
    epochs: int = 15,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Conducts comparative analysis across:
    1. Random Forest (Handcrafted Doppler + Statistical features)
    2. Support Vector Machine (RBF Kernel)
    3. 1D-CNN (PyTorch temporal multi-channel)
    4. 2D-CNN (PyTorch time-subcarrier image)

    Returns:
        Structured dictionary comparing Accuracy, Precision, Recall, F1, Latency, and Parameters.
    """
    n_samples, n_packets, n_subcarriers = X_amp.shape
    num_classes = len(activities)

    # 1. Feature Extraction for traditional ML
    print("[Comparison] Extracting hand-crafted Doppler & Statistical features for ML baselines...")
    X_features = np.array([extract_full_feature_vector(sample) for sample in X_amp])

    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
        X_features, y, test_size=0.2, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_f_scaled = scaler.fit_transform(X_train_f)
    X_test_f_scaled = scaler.transform(X_test_f)

    results = []

    # Model 1: Random Forest
    print("[Comparison] Benchmarking Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    t0 = time.time()
    rf.fit(X_train_f_scaled, y_train_f)
    train_time_rf = time.time() - t0
    y_pred_rf = rf.predict(X_test_f_scaled)

    # Latency test
    sample_f = X_test_f_scaled[:1]
    rf_latencies = [((time.perf_counter() - t0) * 1000) for _ in range(100) if (t0 := time.perf_counter()) and rf.predict(sample_f) is not None]

    rf_acc = accuracy_score(y_test_f, y_pred_rf)
    p_rf, r_rf, f1_rf, _ = precision_recall_fscore_support(y_test_f, y_pred_rf, average="macro", zero_division=0)

    results.append({
        "Model": "Random Forest",
        "Type": "Classical ML",
        "Accuracy (%)": round(rf_acc * 100, 2),
        "F1-Score": round(f1_rf, 4),
        "Inference Latency (ms)": round(float(np.mean(rf_latencies)), 3),
        "Trainable Params": "N/A (100 Trees)"
    })

    # Model 2: SVM (RBF)
    print("[Comparison] Benchmarking Support Vector Machine (SVM)...")
    svm = SVC(kernel="rbf", probability=True, random_state=random_state)
    svm.fit(X_train_f_scaled, y_train_f)
    y_pred_svm = svm.predict(X_test_f_scaled)

    svm_latencies = [((time.perf_counter() - t0) * 1000) for _ in range(100) if (t0 := time.perf_counter()) and svm.predict(sample_f) is not None]
    svm_acc = accuracy_score(y_test_f, y_pred_svm)
    p_svm, r_svm, f1_svm, _ = precision_recall_fscore_support(y_test_f, y_pred_svm, average="macro", zero_division=0)

    results.append({
        "Model": "SVM (RBF Kernel)",
        "Type": "Classical ML",
        "Accuracy (%)": round(svm_acc * 100, 2),
        "F1-Score": round(f1_svm, 4),
        "Inference Latency (ms)": round(float(np.mean(svm_latencies)), 3),
        "Trainable Params": f"{len(svm.support_)} Support Vectors"
    })

    # Model 3: 1D-CNN (PyTorch)
    print("[Comparison] Training and evaluating PyTorch 1D-CNN...")
    train_l_1d, val_l_1d, test_l_1d = prepare_dataloaders(X_amp, y, model_type="1d", batch_size=16)
    cnn_1d = CSI_1D_CNN(in_channels=n_subcarriers, num_classes=num_classes)
    params_1d = sum(p.numel() for p in cnn_1d.parameters() if p.requires_grad)

    train_cnn_model(cnn_1d, train_l_1d, val_l_1d, epochs=epochs, save_path="models/cnn_1d_model.pth")
    eval_1d = evaluate_model(cnn_1d, test_l_1d)
    
    sample_1d = torch.zeros((1, n_subcarriers, n_packets), dtype=torch.float32)
    lat_1d = benchmark_model_inference_latency(cnn_1d, sample_1d)

    results.append({
        "Model": "1D-CNN",
        "Type": "Deep Learning (PyTorch)",
        "Accuracy (%)": round(eval_1d["accuracy"] * 100, 2),
        "F1-Score": round(eval_1d["f1_score"], 4),
        "Inference Latency (ms)": round(lat_1d, 3),
        "Trainable Params": f"{params_1d:,}"
    })

    # Model 4: 2D-CNN (PyTorch)
    print("[Comparison] Training and evaluating PyTorch 2D-CNN...")
    train_l_2d, val_l_2d, test_l_2d = prepare_dataloaders(X_amp, y, model_type="2d", batch_size=16)
    cnn_2d = CSI_2D_CNN(in_channels=1, num_classes=num_classes)
    params_2d = sum(p.numel() for p in cnn_2d.parameters() if p.requires_grad)

    train_cnn_model(cnn_2d, train_l_2d, val_l_2d, epochs=epochs, save_path="models/cnn_2d_model.pth")
    eval_2d = evaluate_model(cnn_2d, test_l_2d)

    sample_2d = torch.zeros((1, 1, n_packets, n_subcarriers), dtype=torch.float32)
    lat_2d = benchmark_model_inference_latency(cnn_2d, sample_2d)

    results.append({
        "Model": "2D-CNN",
        "Type": "Deep Learning (PyTorch)",
        "Accuracy (%)": round(eval_2d["accuracy"] * 100, 2),
        "F1-Score": round(eval_2d["f1_score"], 4),
        "Inference Latency (ms)": round(lat_2d, 3),
        "Trainable Params": f"{params_2d:,}"
    })

    return {
        "comparison_table": results,
        "eval_1d": eval_1d,
        "eval_2d": eval_2d,
        "classes": activities
    }
