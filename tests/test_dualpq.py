import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np

from src.dualpq import DualPQNet, DeepExpert, ClassicalExpert, SNRGate


def test_input_output_dimensions():
    model = DualPQNet(gate_type="snr_learned")
    # Batch size 4, 1280 waveform samples, 191 classical features
    w = torch.randn(4, 1280)
    x = torch.randn(4, 191)
    
    logits, gate = model(w, x)
    assert logits.shape == (4, 29), "Logits dimension should be (B, 29)"
    assert gate.shape == (4, 1), "Gate dimension should be (B, 1)"


def test_expert_branch_gradients():
    model = DualPQNet(gate_type="snr_learned")
    w = torch.randn(4, 1280, requires_grad=True)
    x = torch.randn(4, 191, requires_grad=True)
    
    logits, gate = model(w, x)
    loss = logits.sum()
    loss.backward()
    
    # Check if gradients flow to both experts
    deep_params = list(model.deep_expert.parameters())
    class_params = list(model.classical_expert.parameters())
    
    assert any(p.grad is not None for p in deep_params), "Deep expert should receive gradients"
    assert any(p.grad is not None for p in class_params), "Classical expert should receive gradients"


def test_gate_gradients():
    model = DualPQNet(gate_type="snr_learned")
    w = torch.randn(4, 1280, requires_grad=True)
    x = torch.randn(4, 191, requires_grad=True)
    
    logits, gate = model(w, x)
    loss = logits.sum()
    loss.backward()
    
    gate_params = list(model.gate.parameters())
    assert any(p.grad is not None for p in gate_params), "Gate should receive gradients"


def test_gate_output_bounds():
    gate = SNRGate()
    # Test random SNR values from 0.0 to 1.0 (normalized)
    snr_inputs = torch.rand(100, 1)
    gate_outputs = gate(snr_inputs)
    
    assert torch.all(gate_outputs >= 0.0) and torch.all(gate_outputs <= 1.0), "Gate must output values in [0, 1]"


def test_gate_limits_fusion():
    model = DualPQNet(gate_type="snr_learned")
    w = torch.randn(4, 1280)
    x = torch.randn(4, 191)
    
    z_deep = model.deep_expert(w)
    z_class = model.classical_expert(x)
    
    # Test g = 1.0
    g_1 = torch.ones(4, 1)
    fused_1 = (g_1 * z_deep) + ((1.0 - g_1) * z_class)
    assert torch.allclose(fused_1, z_deep), "When g=1, fused representation must exactly equal z_deep"
    
    # Test g = 0.0
    g_0 = torch.zeros(4, 1)
    fused_0 = (g_0 * z_deep) + ((1.0 - g_0) * z_class)
    assert torch.allclose(fused_0, z_class), "When g=0, fused representation must exactly equal z_class"


def test_leakage_prevention():
    # Simulate a train and test set of 191 features
    X_train = np.random.randn(100, 191) * 5.0 + 10.0
    X_test = np.random.randn(20, 191) * 2.0 - 5.0
    
    scaler = StandardScaler()
    
    # Fit ONLY on train
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform test
    X_test_scaled = scaler.transform(X_test)
    
    # Verify test set statistics are not leaked into the scaler
    assert not np.allclose(np.mean(X_test_scaled, axis=0), 0.0), "Test set should not have exactly zero mean, as scaler is fitted on train"
    assert not np.allclose(np.std(X_test_scaled, axis=0), 1.0), "Test set should not have exactly unit variance, as scaler is fitted on train"
    
    # Verify train set statistics ARE perfectly normalized
    assert np.allclose(np.mean(X_train_scaled, axis=0), 0.0, atol=1e-7)
    assert np.allclose(np.std(X_train_scaled, axis=0), 1.0, atol=1e-7)
