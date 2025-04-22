# c:\Users\rjjaf\_Projects\pytorch-ultimate\docs\deep_learning_pytorch_summary.md
# Deep Learning with PyTorch Summary

This document summarizes core concepts for building deep learning models using PyTorch.

## 1. Tensors

*   **Definition:** Multi-dimensional arrays, similar to NumPy arrays, but with GPU acceleration capabilities.
*   **Creation:** `torch.tensor()`, `torch.zeros()`, `torch.ones()`, `torch.rand()`.
*   **Operations:** Standard mathematical operations (+, -, \*, /), matrix multiplication (`torch.matmul` or `@`), broadcasting.
*   **GPU Usage:** Move tensors to GPU using `.to('cuda')` or `.cuda()`.

## 2. Autograd: Automatic Differentiation

*   **Purpose:** Automatically computes gradients for tensor operations, essential for backpropagation.
*   **`requires_grad=True`:** Tensors that need gradient computation must have this attribute set.
*   **`backward()`:** Called on a scalar tensor (e.g., loss) to compute gradients for all tensors that contributed to it.
*   **`torch.no_grad()`:** Context manager to disable gradient tracking, useful during inference or when updating weights manually.

## 3. `nn.Module`: Building Neural Networks

*   **Base Class:** All neural network models should subclass `nn.Module`.
*   **`__init__()`:** Define layers (e.g., `nn.Linear`, `nn.Conv2d`, `nn.ReLU`) as attributes.
*   **`forward()`:** Defines the computation performed at every call. Input tensors are passed through the defined layers.
*   **Containers:** `nn.Sequential` provides a simple way to stack layers.

## 4. Optimizers

*   **Purpose:** Implement various optimization algorithms (e.g., SGD, Adam) to update model parameters based on computed gradients.
*   **Initialization:** `torch.optim.Adam(model.parameters(), lr=0.001)`.
*   **Usage Loop:**
    *   `optimizer.zero_grad()`: Clear old gradients.
    *   `loss.backward()`: Compute gradients for the current batch.
    *   `optimizer.step()`: Update model parameters.

## 5. Data Loading and Handling (`torch.utils.data`)

*   **`Dataset`:** Abstract class representing a dataset. Requires implementing `__len__` and `__getitem__`.
*   **`DataLoader`:** Wraps a `Dataset` and provides an iterable over the dataset, handling batching, shuffling, and parallel data loading.
*   **Transforms:** Often used with datasets (e.g., `torchvision.transforms`) to preprocess data (normalization, augmentation).

## Example Training Loop Structure

```python
# model = ...
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters())
# dataloader = DataLoader(...)

model.train() # Set model to training mode
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        # Move data to device (CPU/GPU)
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')