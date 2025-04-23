# Module 5: Neural Network from Scratch

In this module, we will guide you through the process of constructing a neural network entirely from scratch using Python and NumPy. You will start with data preparation and model initialization, proceed to implement essential functions like forward and backward propagation, and cover training and evaluation techniques. This hands-on approach helps solidify understanding of the underlying mechanics before leveraging frameworks like PyTorch.

**Learning Objectives:**

*   Build a neural network from scratch, including data preparation and model initialization.
*   Implement forward and backward functions for data propagation and gradient calculation.
*   Train and evaluate a custom neural network model.
*   Understand the role of core components like activation functions, loss functions, gradient descent, and the chain rule.

## Table of Contents
- [Section Overview](#section-overview)
- [Neural Network from Scratch (101 - Theory)](#neural-network-from-scratch-101---theory)
  - [Forward Pass Explained](#forward-pass-explained)
  - [Backward Pass Explained (Weight Updates)](#backward-pass-explained-weight-updates)
  - [The Dot Product](#the-dot-product)
- [Calculating the Dot Product (Coding)](#calculating-the-dot-product-coding)
- [Neural Network from Scratch (Data Prep)](#neural-network-from-scratch-data-prep)
- [Neural Network from Scratch Modeling](#neural-network-from-scratch-modeling)
  - [`__init__` Function](#neural-network-from-scratch-modeling-__init__-function)
  - [Helper Functions (Activation)](#neural-network-from-scratch-modeling-helper-functions-activation)
  - [Forward Function](#neural-network-from-scratch-modeling-forward-function)
  - [Backward Function](#neural-network-from-scratch-modeling-backward-function)
  - [Optimizer Function](#neural-network-from-scratch-modeling-optimizer-function)
  - [Train Function](#neural-network-from-scratch-modeling-train-function)
- [Neural Network from Scratch: Model Training](#neural-network-from-scratch-model-training)
- [Neural Network from Scratch: Model Evaluation](#neural-network-from-scratch-model-evaluation)

---

## Section Overview

[Timestamp 0:01]
Hello and welcome to this section. Here, we'll develop our own neural network completely from scratch.

[Timestamp 0:10]
The process involves these steps:
1.  **Theory:** A foundational lecture on Neural Networks from scratch. [Timestamp 0:10]
2.  **Dot Product (Coding):** A small coding lecture focused on the dot product calculation. [Timestamp 0:22]
3.  **Implementation (Coding):** Several coding lectures implementing the neural network in Python. [Timestamp 0:22]

[Timestamp 0:37]
While not strictly mandatory, following this section is highly recommended. Implementing a neural network yourself significantly deepens your understanding of how they work internally. This knowledge will be invaluable later on. Let's begin with the theory in the next lecture.

---

## Neural Network from Scratch (101 - Theory)

[Timestamp 0:01]
Hello and welcome. In this lecture, we'll cover the theory behind implementing a neural network from scratch, without using PyTorch. We'll develop every function ourselves.

### Forward Pass Explained

[Timestamp 0:19]
Let's revisit the forward pass structure:
1.  **Input Data (`X`):** Starts with our training data.
2.  **Layer 1 (`l1`):** Processes the input using weights (`w`) and biases (`b`).
    *   [Timestamp 0:57] Calculation: `l1 = X * w + b` (simplified representation).
3.  **Activation Function:** Applies a non-linearity. We'll use the Sigmoid function for classification.
    *   [Timestamp 1:14] Sigmoid Function: `S(x) = 1 / (1 + exp(-x))`.
    *   [Timestamp 1:27] The output `S` depends on the output of layer 1: `S = sigmoid(l1)`.
4.  **Predictions (`y_pred`):** The output of the activation function serves as our predictions.
    *   [Timestamp 1:45] `y_pred = S`.
5.  **Loss Calculation:** Compares predictions (`y_pred`) with true values (`y_true`) using a loss function.
    *   [Timestamp 1:53] We'll use Mean Squared Error (MSE): `Loss = (y_pred - y_true)^2`. [Timestamp 1:59]
    *   The result is a single loss score.

![Neural Network Forward Pass](https://raw.githubusercontent.com/yourusername/pytorch-ultimate/main/015_NeuralNetworkFromScratch/images/forward_pass_diagram.png)
*Figure 1: Diagram showing the forward pass through our simple neural network, from input X through weights, bias, activation to final prediction.*

[Timestamp 2:12]
That completes the forward pass. The next question is: how do we update the weights and biases?

### Backward Pass Explained (Weight Updates)

[Timestamp 2:20]
The backward pass adjusts weights to minimize the loss.

**Understanding the Loss Landscape:**
[Timestamp 2:23]
Imagine plotting the loss based on the difference between predictions and true values (`y_pred - y_true`). Since we square the difference (MSE), this forms a parabola.
[Timestamp 2:37] Our goal is to reach the minimum point (loss = 0).
*   If our current loss is at point `L1` (predictions too low), we need to *increase* predictions. [Timestamp 3:13]
*   If at point `L2` (predictions too high), we need to *decrease* predictions. [Timestamp 3:24]

**Gradient Descent:**
[Timestamp 3:40] How do we know which way to adjust? We use the *gradient* (the slope or derivative) of the loss function at our current point.
*   The gradient tells us the direction of the steepest ascent. To minimize loss, we move in the *opposite* direction of the gradient. [Timestamp 3:58]
*   This process is called **Gradient Descent**.
*   A key property of MSE: larger errors result in larger gradients, leading to bigger initial adjustments. [Timestamp 4:21]

![Gradient Descent Visualization](https://raw.githubusercontent.com/yourusername/pytorch-ultimate/main/015_NeuralNetworkFromScratch/images/gradient_descent.png)
*Figure 2: Visualization of gradient descent optimization on a loss landscape, showing how parameters are updated to minimize loss.*

**The Learning Rate (`alpha`):**
[Timestamp 4:53] Simply subtracting the full gradient might cause us to overshoot the minimum, potentially oscillating back and forth (`L1` to `L2` to `L1`...).
[Timestamp 5:20] To control the step size, we introduce a **learning rate** (`alpha`), a small positive number (e.g., 0.01, 0.001).
*   **Weight Update Rule:** `new_weights = old_weights - alpha * gradient` [Timestamp 5:20]
*   [Timestamp 5:49] By multiplying the gradient by `alpha`, we take smaller, more controlled steps towards the minimum. [Timestamp 5:56]

**The Chain Rule:**
[Timestamp 6:24] Our network has multiple steps (Layer 1 -> Activation -> Loss). How do we calculate the gradient of the *loss* with respect to the *weights* (`dL/dw`), which are used much earlier?
[Timestamp 6:50] We use the **Chain Rule** from calculus.
*   Example: Input `X` -> Function `Y(X)` -> Function `Z(Y)`.
*   To find `dZ/dX`, we multiply the derivatives of each step: `dZ/dX = dZ/dY * dY/dX`. [Timestamp 7:22, 7:52]

**Applying the Chain Rule to Our Network:**
[Timestamp 8:05]
To find `dL/dw` (the gradient needed to update weights):
1.  Calculate `dL/dy_pred`: Derivative of the Loss function w.r.t. predictions.
    *   For MSE: `dL/dy_pred = 2 * (y_pred - y_true)`.
2.  Calculate `dy_pred/dl1`: Derivative of the Activation function (Sigmoid) w.r.t. the output of layer 1.
3.  Calculate `dl1/dw`: Derivative of Layer 1's output w.r.t. the weights.
    *   This is simply the input `X`. [Timestamp 8:30]

[Timestamp 8:44]
**Final Gradient for Weights:** `dL/dw = (dL/dy_pred) * (dy_pred/dl1) * (dl1/dw)`

**Gradient for Biases (`dL/db`):**
[Timestamp 9:00] The process is similar, but the last derivative changes:
*   `dl1/db`: Derivative of Layer 1's output w.r.t. the bias.
    *   This is simply `1`. [Timestamp 9:10]
**Final Gradient for Biases:** `dL/db = (dL/dy_pred) * (dy_pred/dl1) * (dl1/db)`

[Timestamp 9:22] That covers the core mechanics of the backward pass.

### The Dot Product

[Timestamp 9:22] One more technical aspect: the **dot product**.
Why use it? It efficiently calculates the weighted sum in our layers (`X * w`) and helps measure similarity between vectors (like input `X` and weight vector `w`).

[Timestamp 9:41]
Consider an input vector `X` and two weight vectors `w1` and `w2`. Which weight vector is more "aligned" or similar to the input?
*   [Timestamp 10:14] Similarity depends on both the *magnitude* (length) of the vectors and the *angle* between them.
*   [Timestamp 10:25] The dot product captures this relationship efficiently.

**Calculating the Dot Product:**
[Timestamp 10:48] Given `X = [x1, x2, ...]` and `w1 = [w1_1, w1_2, ...]`:
*   Dot Product (`X . w1`) = `(x1 * w1_1) + (x2 * w1_2) + ...`
*   It's the sum of the element-wise products. [Timestamp 11:00]
*   Larger dot product values generally indicate greater similarity/alignment between the vectors.

We'll see how to compute this easily with NumPy in the next video. Thank you.

**Section Summary:**
> **Key Takeaways from Theory:**
> - The forward pass consists of: input → weighted sum → activation function → predictions → loss calculation
> - The backward pass uses gradient descent to update weights by minimizing loss
> - The chain rule allows us to calculate gradients through multiple layers
> - Learning rate controls step size during optimization
> - Dot products efficiently calculate weighted sums and measure vector similarity

---

## Calculating the Dot Product (Coding)

[Timestamp 0:01]
Hello and welcome. Let's implement the dot product calculation using Python and NumPy. You can find the code in the `015_NeuralNetworkFromScratch` subfolder.

[Timestamp 0:01] We only need NumPy:
```python
import numpy as np
```

[Timestamp 0:20] Let's define sample input and weight vectors:
```python
X = np.array([1, 2, 3])
w1 = np.array([0.2, 0.8, -0.5])
w2 = np.array([-0.5, 0.1, 0.9])
```

**Manual Calculation:**
[Timestamp 0:28] Calculating the dot product manually for `X` and `w1`:
```python
dot_X_w1_manual = (X[0] * w1[0]) + (X[1] * w1[1]) + (X[2] * w1[2])
# [Timestamp 0:43] -> (1 * 0.2) + (2 * 0.8) + (3 * -0.5)
# [Timestamp 0:56] -> 0.2 + 1.6 - 1.5 = 0.3
print(dot_X_w1_manual) # Output: 0.3 (Note: Video calculation seems slightly off, result should be 0.3)
```
[Timestamp 1:13] This is tedious for larger vectors.

[Timestamp 1:29] Similarly for `X` and `w2`:
```python
dot_X_w2_manual = (X[0] * w2[0]) + (X[1] * w2[1]) + (X[2] * w2[2])
# -> (1 * -0.5) + (2 * 0.1) + (3 * 0.9)
# -> -0.5 + 0.2 + 2.7 = 2.4
print(dot_X_w2_manual) # Output: 2.4 (Note: Video calculation seems slightly off, result should be 2.4)
```
[Timestamp 1:43] Comparing the results (0.3 vs 2.4), `w2` has a larger dot product with `X`, suggesting more similarity in this case (assuming the corrected calculations). [Timestamp 1:58]

**NumPy's `dot` Function:**
[Timestamp 1:58] NumPy provides a much simpler way: `np.dot()`.
```python
dot_X_w1_numpy = np.dot(X, w1)
print(dot_X_w1_numpy) # Output: 0.3 [Timestamp 2:17]

dot_X_w2_numpy = np.dot(X, w2)
print(dot_X_w2_numpy) # Output: 2.4 [Timestamp 2:24]
```
Perfect! This matches our manual (corrected) calculation and is much more concise.

[Timestamp 2:24] This `np.dot` function is the final piece we need before building the full network from scratch, which we'll start in the next lecture.

**Section Summary:**
> **Key Takeaways from Dot Product:**
> - Dot products calculate the sum of element-wise multiplication between vectors
> - NumPy's `np.dot()` provides an efficient implementation for vectors and matrices
> - This operation is fundamental to neural network calculations
> - Larger dot product values often indicate greater similarity between vectors

---

## Neural Network from Scratch (Data Prep)

[Timestamp 0:01]
Hello and welcome. Now we begin developing our neural network from scratch. We'll start with data preparation. I've prepared a script (`datascript`) for imports.

**Imports:**
[Timestamp 0:01]
```python
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
# (Assuming matplotlib is also potentially used with seaborn)
import matplotlib.pyplot as plt
```

**Dataset:**
[Timestamp 0:41] We'll use a heart attack prediction dataset (available on Kaggle).
*   [Timestamp 0:49] It's a classification problem.
*   [Timestamp 1:04] Features include age, sex, and other health metrics.
*   [Timestamp 1:08] The target variable (`output`) is binary (0: less chance, 1: higher chance of heart attack).

**Loading Data:**
[Timestamp 1:33] Assuming the data (`heart.csv`) is in the working directory:
```python
df = pd.read_csv('heart.csv')
print(df.head())
```
[Timestamp 1:48] We can see the features and the `output` column.

**Separating Features and Target:**
[Timestamp 1:48] We need to separate independent features (`X`) from the dependent target variable (`Y`).
```python
X = df.drop('output', axis=1).values # Convert to NumPy array
Y = df['output'].values           # Convert to NumPy array
```
[Timestamp 2:17] Check shapes:
```python
print("X shape:", X.shape) # E.g., (303, 13) -> 303 observations, 13 features
print("Y shape:", Y.shape) # E.g., (303,) -> 303 target values
```

**Train/Test Split:**
[Timestamp 2:19] We split the data to train the model and evaluate it on unseen data.
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, Y,
    test_size=0.2,    # 20% for testing [Timestamp 2:43]
    random_state=42   # For reproducibility [Timestamp 2:58]
)
```

**Feature Scaling:**
[Timestamp 3:08] Neural networks often perform better when input features are scaled. Different features have varying ranges (e.g., age vs. cholesterol).
[Timestamp 3:08] We'll use `StandardScaler` (transforms data to have mean=0, stddev=1).
```python
scaler = StandardScaler()

# Fit on training data ONLY and transform it
X_train_scale = scaler.fit_transform(X_train) [Timestamp 3:40]

# Use the SAME scaler (fitted on train data) to transform test data
X_test_scale = scaler.transform(X_test) [Timestamp 3:40]
```
**Important:** Fit the scaler *only* on the training data to avoid data leakage from the test set.

[Timestamp 4:10] With data prepared and scaled, we can now start building the neural network model itself.

**Section Summary:**
> **Key Takeaways from Data Preparation:**
> - Split data into features (X) and target (Y) variables
> - Further divide into training and testing sets using `train_test_split`
> - Scale features using `StandardScaler` to normalize different feature ranges
> - Apply the scaler only to training data first, then use the same scaler for test data
> - Properly preprocessed data is crucial for neural network performance

---

## Neural Network from Scratch Modeling: `__init__` Function

[Timestamp 0:00]
We'll structure our network within a Python class for better organization, similar to how PyTorch models are often defined. [Timestamp 0:11]

```python
import numpy as np

class NeuralNetworkFromScratch:
    # [Timestamp 0:11]
    def __init__(self, learning_rate, X_train, y_train, X_test, y_test):
        """
        Initializes the Neural Network.
        Args:
            learning_rate (float): The step size for gradient descent.
            X_train, y_train (np.array): Scaled training data and labels.
            X_test, y_test (np.array): Scaled test data and labels.
        """
        # [Timestamp 0:34] Parameters passed during instantiation

        # --- Initialize Weights and Biases ---
        # [Timestamp 0:53] Randomly initialize weights. One weight per input feature.
        num_features = X_train.shape[1]
        self.weights = np.random.randn(num_features) # [Timestamp 1:06]
        # [Timestamp 1:10] Initialize bias (single value for this simple network)
        self.bias = np.random.randn(1) # [Timestamp 1:40]

        # --- Store Data and Parameters ---
        # [Timestamp 1:49] Assign passed parameters to internal attributes
        self.learning_rate = learning_rate # [Timestamp 1:56]
        self.X_train = X_train # [Timestamp 2:00]
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test # [Timestamp 2:27]

        # --- Initialize Loss Tracking ---
        # [Timestamp 2:37] Lists to store loss values during training
        self.L_train = [] # Training losses [Timestamp 2:41]
        self.L_test = []  # Test losses [Timestamp 2:45]

    # ... other methods (activation, forward, backward, etc.) will go here ...

```
This `__init__` method sets up the essential components: weights, bias, learning rate, data storage, and lists for tracking performance.

---

## Neural Network from Scratch Modeling: Helper Functions (Activation)

[Timestamp 0:00]
Now, let's add the helper functions needed within our class, starting with the activation function and its derivative.

```python
# (Inside the NeuralNetworkFromScratch class)

    # [Timestamp 0:09] --- Activation Function (Sigmoid) ---
    def _sigmoid(self, x):
        """ Applies the Sigmoid activation function element-wise. """
        # [Timestamp 0:20] Formula: 1 / (1 + exp(-x))
        return 1 / (1 + np.exp(-x)) # [Timestamp 0:37]

    # [Timestamp 0:43] --- Derivative of Sigmoid ---
    def _sigmoid_derivative(self, x):
        """ Computes the derivative of the Sigmoid function. """
        # [Timestamp 0:52] Formula: sigmoid(x) * (1 - sigmoid(x))
        sig_x = self._sigmoid(x) # [Timestamp 1:15]
        return sig_x * (1 - sig_x) # [Timestamp 1:20, 1:30]
        # [Timestamp 1:30] You can verify this formula online.
```
[Timestamp 1:44] These functions encapsulate the Sigmoid calculation and its derivative, which we'll need for the forward and backward passes, respectively. We use a leading underscore (`_`) by convention to indicate these are intended primarily for internal use within the class.

---

## Neural Network from Scratch Modeling: Forward Function

[Timestamp 0:00]
Next, we define the `forward` pass function within our class. This function takes an input `X` and propagates it through the network to produce a prediction.

```python
# (Inside the NeuralNetworkFromScratch class)

    # [Timestamp 0:05] --- Forward Pass ---
    def forward(self, X):
        """
        Performs the forward pass of the network.
        Args:
            X (np.array): Input data (a single sample or batch).
        Returns:
            np.array: Network's prediction(s).
        """
        # [Timestamp 0:15] Step 1: Calculate weighted sum + bias (Linear Layer)
        # [Timestamp 0:15] Using dot product for weighted sum
        hidden1 = np.dot(X, self.weights) + self.bias # [Timestamp 0:25, 0:36]

        # [Timestamp 0:43] Step 2: Apply activation function
        activated_output = self._sigmoid(hidden1) # [Timestamp 0:47, 0:55]

        # [Timestamp 1:01] Step 3: Return the final prediction
        # (In this simple network, the activated output is the final prediction)
        return activated_output
```
[Timestamp 0:55] This function implements the steps we discussed in the theory lecture: linear combination followed by the Sigmoid activation. It returns the network's output (prediction) for the given input `X`.

---

## Neural Network from Scratch Modeling: Backward Function

[Timestamp 0:00]
Now, we implement the `backward` pass, which calculates the gradients of the loss with respect to the weights and bias. These gradients tell us how to adjust the parameters to reduce the loss.

```python
# (Inside the NeuralNetworkFromScratch class)

    # [Timestamp 0:04] --- Backward Pass (Gradient Calculation) ---
    def backward(self, X_sample, y_true_sample):
        """
        Performs the backward pass to calculate gradients for a single sample.
        Args:
            X_sample (np.array): A single input data sample.
            y_true_sample (float): The corresponding true label.
        Returns:
            tuple: Gradients (dL_db, dL_dw).
        """
        # [Timestamp 0:16] --- Recompute steps from Forward Pass (needed for derivatives) ---
        # 1. Linear layer output
        hidden1 = np.dot(X_sample, self.weights) + self.bias # [Timestamp 0:21, 0:28]
        # 2. Prediction (output after activation)
        y_pred = self._sigmoid(hidden1) # [Timestamp 0:33, 0:39] (Can also call self.forward(X_sample))

        # [Timestamp 0:44] --- Calculate Gradients using Chain Rule ---
        # Derivative of Loss w.r.t. Prediction (dL/d_pred)
        # Assuming Mean Squared Error Loss: Loss = (y_pred - y_true)^2
        # Derivative = 2 * (y_pred - y_true)
        dL_dpred = 2 * (y_pred - y_true_sample) # [Timestamp 1:09, 1:16]

        # Derivative of Prediction w.r.t. Hidden Layer Output (d_pred/d_hidden1)
        # This is the derivative of the sigmoid function
        dpred_dhidden1 = self._sigmoid_derivative(hidden1) # [Timestamp 1:26, 1:38, 1:48]

        # Derivative of Hidden Layer Output w.r.t Bias (d_hidden1/db)
        # d(X*w + b)/db = 1
        dhidden1_db = 1 # [Timestamp 2:01]

        # Derivative of Hidden Layer Output w.r.t Weights (d_hidden1/dw)
        # d(X*w + b)/dw = X
        dhidden1_dw = X_sample # [Timestamp 2:09, 2:14]

        # --- Combine derivatives using Chain Rule ---
        # Gradient for Bias (dL/db = dL/d_pred * d_pred/d_hidden1 * d_hidden1/db)
        dL_db = dL_dpred * dpred_dhidden1 * dhidden1_db # [Timestamp 2:26, 2:40]
        # Need to ensure dimensions match for multiplication if X_sample is multi-feature
        # Often requires summing or averaging gradients across batch/features depending on implementation
        # For simplicity here, assuming element-wise multiplication works or needs adjustment (e.g., sum dL_db)

        # Gradient for Weights (dL/dw = dL/d_pred * d_pred/d_hidden1 * d_hidden1/dw)
        # We need element-wise multiplication of dL_dpred * dpred_dhidden1 with each element of dhidden1_dw (X_sample)
        # Reshape might be needed depending on exact NumPy broadcasting rules
        # A common way: dL_dw = X_sample.T @ (dL_dpred * dpred_dhidden1) - check dimensions
        # Let's refine dL_dw calculation assuming X_sample is (features,) and intermediate grads are scalars/broadcastable
        dL_dw_intermediate = dL_dpred * dpred_dhidden1
        dL_dw = dhidden1_dw * dL_dw_intermediate # Element-wise multiplication

        # Sum gradients if needed (especially dL_db if it resulted from broadcasting)
        dL_db = np.sum(dL_db) # Summing over the potential broadcasted dimension

        # [Timestamp 3:11] --- Return calculated gradients ---
        return dL_db, dL_dw # [Timestamp 3:16, 3:26]

```
[Timestamp 3:26] This function calculates the necessary gradients (`dL_db`, `dL_dw`) for a single data sample using the chain rule, breaking down the derivatives step-by-step. These gradients will then be used by the optimizer to update the network's parameters. *Note: The exact NumPy operations for combining derivatives (`dL_db`, `dL_dw`) might need careful handling of shapes and broadcasting, especially `dL_dw`.*

---

## Neural Network from Scratch Modeling: Optimizer Function

[Timestamp 0:00]
The `optimizer` function takes the calculated gradients and updates the network's weights and bias according to the gradient descent rule.

```python
# (Inside the NeuralNetworkFromScratch class)

    # [Timestamp 0:02] --- Optimizer (Parameter Update) ---
    def optimizer(self, dL_db, dL_dw):
        """
        Updates the weights and bias using the calculated gradients.
        Args:
            dL_db (float): Gradient of the loss w.r.t. the bias.
            dL_dw (np.array): Gradient of the loss w.r.t. the weights.
        """
        # [Timestamp 0:09] Update bias using gradient descent rule
        # new_bias = old_bias - learning_rate * gradient_bias
        self.bias = self.bias - self.learning_rate * dL_db # [Timestamp 0:18, 0:25, 0:30]

        # [Timestamp 0:37] Update weights using gradient descent rule
        # new_weights = old_weights - learning_rate * gradient_weights
        self.weights = self.weights - self.learning_rate * dL_dw # [Timestamp 0:37, 0:56]

```
[Timestamp 0:56] This simple function implements the core update step of gradient descent. It subtracts a fraction (controlled by `self.learning_rate`) of the gradient from the current parameters (`self.bias`, `self.weights`), moving them in the direction that should decrease the loss.

---

## Neural Network from Scratch Modeling: Train Function

[Timestamp 0:00]
The `train` function orchestrates the learning process. It iterates multiple times (epochs or iterations), processing data samples, calculating losses, computing gradients, and updating parameters.

```python
# (Inside the NeuralNetworkFromScratch class)

    # [Timestamp 0:00] --- Training Loop ---
    def train(self, iterations):
        """
        Trains the neural network for a specified number of iterations.
        Args:
            iterations (int): The number of training iterations.
        """
        # [Timestamp 0:18, 0:23] Iterate for the specified number of times
        for i in range(iterations):
            # --- Training Step on a Random Sample ---
            # [Timestamp 0:32] Select a random data point from the training set
            random_pos = np.random.randint(0, len(self.X_train)) # [Timestamp 0:41, 0:54, 1:02]
            X_sample = self.X_train[random_pos] # [Timestamp 1:19]
            y_true_sample = self.y_train[random_pos] # [Timestamp 1:35]

            # [Timestamp 1:27] 1. Forward Pass (get prediction for the sample)
            y_train_pred = self.forward(X_sample) # [Timestamp 1:48, 1:54, 2:02]

            # [Timestamp 2:06] 2. Calculate Training Loss (for this sample)
            # Using Mean Squared Error for this example
            train_loss = np.sum(np.square(y_train_pred - y_true_sample)) # [Timestamp 2:20, 2:30]
            self.L_train.append(train_loss) # Store loss for plotting [Timestamp 2:51]

            # [Timestamp 3:08] 3. Backward Pass (calculate gradients for the sample)
            dL_db, dL_dw = self.backward(X_sample, y_true_sample) # [Timestamp 3:13, 3:19, 3:24]

            # [Timestamp 3:41] 4. Optimize (update weights and bias)
            self.optimizer(dL_db, dL_dw) # [Timestamp 3:47, 3:52, 4:07]

            # --- Evaluation Step on Test Set (Optional, per iteration) ---
            # [Timestamp 4:08] Calculate loss on the *entire* test set periodically
            # This is computationally more expensive but gives insight into generalization
            if i % 50 == 0: # Example: Evaluate every 50 iterations
                test_loss_sum = 0
                # [Timestamp 4:27] Iterate through all test samples
                for j in range(len(self.X_test)): # [Timestamp 4:40]
                    y_test_true = self.y_test[j] # [Timestamp 4:45, 4:54]
                    y_test_pred = self.forward(self.X_test[j]) # [Timestamp 4:59, 5:02, 5:11]
                    # [Timestamp 5:15] Calculate squared error for this test sample
                    test_loss_sum += np.sum(np.square(y_test_pred - y_test_true)) # [Timestamp 5:22, 5:36]
                # [Timestamp 5:48] Average loss over the test set (optional, could just store sum)
                avg_test_loss = test_loss_sum / len(self.X_test)
                self.L_test.append(avg_test_loss) # Append average test loss [Timestamp 6:04]

        # [Timestamp 6:07] Indicate training completion
        print("Training successful.")

```
This `train` function implements stochastic gradient descent (SGD) by processing one random sample at a time. It performs the forward pass, calculates loss, performs the backward pass to get gradients, and then updates the parameters using the optimizer. It also includes optional evaluation on the test set within the loop to monitor generalization performance, storing both training and test losses.

**Section Summary:**
> **Key Takeaways from Neural Network Implementation:**
> - A neural network can be represented as a class with attributes for weights, biases, and data
> - Key components include activation functions (sigmoid), forward propagation, backward propagation, and optimization
> - The backward function implements the chain rule for calculating gradients
> - The training process involves iteratively selecting samples, making predictions, calculating loss, and updating parameters
> - Periodically evaluating on test data helps track generalization performance

---

## Neural Network from Scratch: Model Training

[Timestamp 0:01]
Now that our `NeuralNetworkFromScratch` class is defined, let's instantiate it and run the training process.

**Hyperparameters:**
[Timestamp 0:01] Set the learning rate and number of iterations.
```python
LEARNING_RATE = 0.01 # [Timestamp 0:13] Example value (adjust as needed)
ITERATIONS = 1000   # [Timestamp 0:20] Number of training steps
```

**Instantiate the Model:**
[Timestamp 0:26] Create an instance of our class, passing the hyperparameters and the prepared data.
```python
nn = NeuralNetworkFromScratch(
    learning_rate=LEARNING_RATE, # [Timestamp 0:33]
    X_train=X_train_scale,       # Use scaled data [Timestamp 0:51]
    y_train=y_train,             # [Timestamp 0:58]
    X_test=X_test_scale,         # Use scaled data [Timestamp 1:04]
    y_test=y_test                # [Timestamp 1:09]
)
# [Timestamp 1:15]
```

**Run Training:**
[Timestamp 1:15] Call the `train` method on the instance.
```python
nn.train(ITERATIONS) # [Timestamp 1:21, 1:27]
```
[Timestamp 1:33] *Self-correction during execution:* The original code had a `shape` attribute missing when initializing weights/bias. Assuming that's fixed in the class definition as shown previously.
```python
# Potential fix inside __init__ if error occurs:
# num_features = X_train.shape[1] # Get number of features
# self.weights = np.random.randn(num_features)
# self.bias = np.random.randn(1) # Or np.random.randn() if scalar bias preferred
```
[Timestamp 1:45] After fixing potential initialization errors, running `nn.train(ITERATIONS)` should execute the training loop. "Training successful" indicates completion.

---

## Neural Network from Scratch: Model Evaluation

[Timestamp 0:00]
After training, we need to evaluate how well our network performs, especially on the unseen test data.

**1. Plotting Losses:**
[Timestamp 0:03] Visualize the test loss over iterations to see if the model learned effectively.
```python
import matplotlib.pyplot as plt
import seaborn as sns

# [Timestamp 0:11] Create x-axis values (iterations where test loss was recorded)
# Assuming test loss was recorded every 50 iterations in train()
iterations_recorded = range(0, ITERATIONS, 50) # Adjust step if different in train()
# [Timestamp 0:19] (Or use range(len(nn.L_test)) if recorded every iteration)
x_axis = range(len(nn.L_test)) # Simpler if recorded every iteration or use the list directly

plt.figure(figsize=(10, 6))
sns.lineplot(x=x_axis, y=nn.L_test) # [Timestamp 0:23, 0:45]
plt.title("Test Loss During Training")
plt.xlabel("Training Iteration (or Recording Step)")
plt.ylabel("Average Test Loss")
plt.show()
```
[Timestamp 0:45] A decreasing loss curve that stabilizes indicates successful learning. [Timestamp 0:45]

**2. Calculating Accuracy on Test Set:**
[Timestamp 1:00] Iterate through the test set, get predictions, compare to true labels, and calculate accuracy.
```python
num_test_samples = X_test_scale.shape[0] # [Timestamp 1:15, 1:19]
correct_predictions = 0 # [Timestamp 1:24, 1:27]
y_preds_list = [] # Store predictions for confusion matrix [Timestamp 1:36]

# [Timestamp 1:42] Iterate through each test sample
for i in range(num_test_samples): # [Timestamp 1:45]
    X_sample = X_test_scale[i]
    y_true = y_test[i] # [Timestamp 1:54]

    # Get prediction
    y_pred_proba = nn.forward(X_sample) # [Timestamp 2:11, 2:14]
    # Convert probability to binary prediction (0 or 1)
    y_pred_binary = np.round(y_pred_proba).astype(int) # [Timestamp 2:20, 2:30]

    y_preds_list.append(y_pred_binary.item()) # Store prediction [Timestamp 2:38, 2:48]

    # Check if prediction is correct
    if y_pred_binary == y_true: # [Timestamp 2:57]
        correct_predictions += 1 # [Timestamp 2:50, 3:05]

# [Timestamp 3:11] Calculate overall accuracy
accuracy = correct_predictions / num_test_samples # [Timestamp 3:13]
print(f"Test Accuracy: {accuracy:.2%}") # [Timestamp 3:24] -> e.g., 82%
```

**3. Checking Baseline Accuracy:**
[Timestamp 3:24] Compare the model's accuracy to a naive baseline (e.g., always predicting the majority class).
```python
from collections import Counter

# [Timestamp 4:01] Count occurrences of each class in the test set
y_test_counts = Counter(y_test)
# [Timestamp 4:08] Find the count of the majority class
majority_class_count = max(y_test_counts.values())
# Calculate baseline accuracy
baseline_accuracy = majority_class_count / num_test_samples
print(f"Baseline (Majority Class) Accuracy: {baseline_accuracy:.2%}") # [Timestamp 4:08] -> e.g., 52%
```
[Timestamp 4:08] If our model's accuracy (e.g., 82%) is significantly higher than the baseline (e.g., 52%), it indicates the model has learned meaningful patterns.

**4. Confusion Matrix:**
[Timestamp 4:36] Provides a more detailed breakdown of correct and incorrect predictions for each class.
```python
from sklearn.metrics import confusion_matrix

# [Timestamp 4:42] Ensure y_preds_list is a NumPy array if needed by confusion_matrix
y_preds_array = np.array(y_preds_list)

cm = confusion_matrix(y_test, y_preds_array) # [Timestamp 4:47, 4:50]
print("\nConfusion Matrix:")
print(cm)

# Optional: Visualize confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
```
[Timestamp 4:59] The matrix shows True Positives, True Negatives, False Positives, and False Negatives, giving insight into the types of errors the model makes.

**Experimenting with Hyperparameters:**
[Timestamp 5:00] You can now experiment by changing `LEARNING_RATE` or `ITERATIONS` and retraining to see the impact on the loss curve and final accuracy. [Timestamp 5:50, 5:55, 6:01]

**Summary & Next Steps:**
[Timestamp 6:01] We have successfully:
*   Imported and preprocessed data (scaling, splitting). [Timestamp 6:10]
*   Built a `NeuralNetworkFromScratch` class containing: [Timestamp 6:26]
    *   `__init__`: Initialization.
    *   `_sigmoid`, `_sigmoid_derivative`: Activation functions. [Timestamp 6:50]
    *   `forward`: Prediction generation. [Timestamp 6:50]
    *   `backward`: Gradient calculation. [Timestamp 7:03]
    *   `optimizer`: Parameter updates. [Timestamp 7:03]
    *   `train`: Training loop orchestration. [Timestamp 7:14]
*   Trained the network. [Timestamp 7:31]
*   Evaluated performance using loss plots, accuracy, and a confusion matrix. [Timestamp 7:31, 7:46]

![Neural Network Complete Architecture](https://raw.githubusercontent.com/yourusername/pytorch-ultimate/main/015_NeuralNetworkFromScratch/images/complete_nn_architecture.png)
*Figure 3: Complete architecture of our neural network showing data flow, training process, and evaluation metrics.*

