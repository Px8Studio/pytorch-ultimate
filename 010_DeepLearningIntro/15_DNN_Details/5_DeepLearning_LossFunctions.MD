# c:\Users\rjjaf\_Projects\pytorch-ultimate\docs\deep_learning_pytorch_summary.md
# Deep Learning with PyTorch Summary

This document summarizes core concepts for building deep learning models using PyTorch.

## 1. Tensors

*   **Definition:** Multi-dimensional arrays, similar to NumPy arrays, but with GPU acceleration capabilities.
*   **Creation:** `torch.tensor()`, `torch.zeros()`, `torch.ones()`, `torch.rand()`.
*   **Operations:** Standard mathematical operations (+, -, \*, /), matrix multiplication (`torch.matmul` or `@`), broadcasting.
*   **GPU Usage:** Move tensors to GPU using `.to('cuda')` or `.cuda()`.
*   **Dimensions:** Tensors can have any number of dimensions - scalars (0D), vectors (1D), matrices (2D), and n-dimensional arrays.
*   **Reshaping:** Use `view()`, `reshape()`, or `permute()` to reorganize tensor dimensions.

## 2. Autograd: Automatic Differentiation

*   **Purpose:** Automatically computes gradients for tensor operations, essential for backpropagation.
*   **`requires_grad=True`:** Tensors that need gradient computation must have this attribute set.
*   **`backward()`:** Called on a scalar tensor (e.g., loss) to compute gradients for all tensors that contributed to it.
*   **`torch.no_grad()`:** Context manager to disable gradient tracking, useful during inference or when updating weights manually.
*   **Computational Graph:** PyTorch builds a dynamic computational graph that tracks operations for automatic differentiation.
*   **Gradient Accumulation:** Gradients are accumulated by default; use `optimizer.zero_grad()` to reset them before each backward pass.

## 3. `nn.Module`: Building Neural Networks

*   **Base Class:** All neural network models should subclass `nn.Module`.
*   **`__init__()`:** Define layers (e.g., `nn.Linear`, `nn.Conv2d`, `nn.ReLU`) as attributes.
*   **`forward()`:** Defines the computation performed at every call. Input tensors are passed through the defined layers.
*   **Containers:** `nn.Sequential` provides a simple way to stack layers.
*   **Parameter Management:** `nn.Module` automatically tracks parameters defined as module attributes.
*   **Nested Modules:** Complex architectures can be built by nesting modules within modules.

## 4. Loss Functions

*   **Purpose:** Measure the difference between model predictions and ground truth targets.
*   **Common Loss Functions:**
    * **Classification:** `nn.CrossEntropyLoss()` (combines softmax and negative log likelihood), `nn.BCELoss()` (binary classification)
    * **Regression:** `nn.MSELoss()` (mean squared error), `nn.L1Loss()` (mean absolute error)
    * **Advanced:** `nn.KLDivLoss()` (KL divergence), `nn.CTCLoss()` (for sequence prediction)
*   **Custom Losses:** Can be implemented by creating a function or subclassing `nn.Module`.

## 5. Optimizers

*   **Purpose:** Implement various optimization algorithms (e.g., SGD, Adam) to update model parameters based on computed gradients.
*   **Initialization:** `torch.optim.Adam(model.parameters(), lr=0.001)`.
*   **Usage Loop:**
    *   `optimizer.zero_grad()`: Clear old gradients.
    *   `loss.backward()`: Compute gradients for the current batch.
    *   `optimizer.step()`: Update model parameters.
*   **Common Optimizers:**
    * **SGD:** Simple but effective with proper learning rate scheduling
    * **Adam:** Adaptive learning rates, usually converges faster
    * **RMSprop:** Good for RNNs and non-stationary objectives
    * **AdamW:** Adam with proper weight decay regularization

## 6. Data Loading and Handling (`torch.utils.data`)

*   **`Dataset`:** Abstract class representing a dataset. Requires implementing `__len__` and `__getitem__`.
*   **`DataLoader`:** Wraps a `Dataset` and provides an iterable over the dataset, handling batching, shuffling, and parallel data loading.
*   **Transforms:** Often used with datasets (e.g., `torchvision.transforms`) to preprocess data (normalization, augmentation).
*   **Custom Datasets:** Create your own by subclassing `Dataset` and implementing required methods.
*   **Sampling:** Control batch composition with `WeightedRandomSampler` for imbalanced datasets.
*   **Data Prefetching:** `DataLoader` can load batches in parallel (`num_workers>0`) while GPU computes gradients.

## 7. Regularization Techniques

*   **Dropout:** `nn.Dropout(p=0.5)` - Randomly zeros elements during training to prevent co-adaptation.
*   **Batch Normalization:** `nn.BatchNorm2d()` - Normalizes activations, stabilizes learning, and acts as regularization.
*   **Weight Decay:** Add L2 penalty through optimizer (e.g., `weight_decay=1e-4` parameter).
*   **Early Stopping:** Monitor validation performance and stop training when it begins to degrade.
*   **Data Augmentation:** Apply random transformations to training data to improve generalization.

## 8. Model Evaluation and Saving

*   **Evaluation Mode:** Use `model.eval()` before inference to disable dropout and use batch norm statistics.
*   **Training Mode:** Use `model.train()` during training to enable dropout and update batch norm statistics.
*   **Saving Models:** `torch.save(model.state_dict(), 'model.pth')` to save parameters.
*   **Loading Models:** `model.load_state_dict(torch.load('model.pth'))` to restore parameters.
*   **Inference:** Use `with torch.no_grad():` to disable gradient tracking for faster inference.

## Example Training Loop Structure

```python
# model = ...
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters())
# train_loader, val_loader = DataLoader(...), DataLoader(...)

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, '
          f'Val Loss: {val_loss/len(val_loader):.4f}, '
          f'Accuracy: {100 * correct / total:.2f}%')
```