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

Hello and welcome to this section. Here, we'll develop our own neural network completely from scratch.

The process involves these steps:
1.  **Theory:** A foundational lecture on Neural Networks from scratch.
2.  **Dot Product (Coding):** A small coding lecture focused on the dot product calculation.
3.  **Implementation (Coding):** Several coding lectures implementing the neural network in Python.

While not strictly mandatory, following this section is highly recommended. Implementing a neural network yourself significantly deepens your understanding of how they work internally. This knowledge will be invaluable later on. Let's begin with the theory in the next lecture.

---

## Neural Network from Scratch (101 - Theory)

Hello and welcome. In this lecture, we'll cover the theory behind implementing a neural network from scratch, without using PyTorch. We'll develop every function ourselves.

### Forward Pass Explained

Let's revisit the forward pass structure:
1.  **Input Data (`X`):** Starts with our training data.
2.  **Layer 1 (`l1`):** Processes the input using weights (`w`) and biases (`b`).
    *   Calculation: `l1 = X * w + b` (simplified representation).
3.  **Activation Function:** Applies a non-linearity. We'll use the Sigmoid function for classification.
    *   Sigmoid Function: `S(x) = 1 / (1 + exp(-x))`.
    *   The output `S` depends on the output of layer 1: `S = sigmoid(l1)`.
4.  **Predictions (`y_pred`):** The output of the activation function serves as our predictions.
    *   `y_pred = S`.
5.  **Loss Calculation:** Compares predictions (`y_pred`) with true values (`y_true`) using a loss function.
    *   We'll use Mean Squared Error (MSE): `Loss = (y_pred - y_true)^2`.
    *   The result is a single loss score.

![Neural Network Forward Pass](https://raw.githubusercontent.com/yourusername/pytorch-ultimate/main/015_NeuralNetworkFromScratch/images/forward_pass_diagram.png)
*Figure 1: Diagram showing the forward pass through our simple neural network, from input X through weights, bias, activation to final prediction.*

That completes the forward pass. The next question is: how do we update the weights and biases?

### Backward Pass Explained (Weight Updates)

The backward pass adjusts weights to minimize the loss.

**Understanding the Loss Landscape:**
Imagine plotting the loss based on the difference between predictions and true values (`y_pred - y_true`). Since we square the difference (MSE), this forms a parabola.
Our goal is to reach the minimum point (loss = 0).
*   If our current loss is at point `L1` (predictions too low), we need to *increase* predictions.
*   If at point `L2` (predictions too high), we need to *decrease* predictions.

**Gradient Descent:**
How do we know which way to adjust? We use the *gradient* (the slope or derivative) of the loss function at our current point.
*   The gradient tells us the direction of the steepest ascent. To minimize loss, we move in the *opposite* direction of the gradient.
*   This process is called **Gradient Descent**.
*   A key property of MSE: larger errors result in larger gradients, leading to bigger initial adjustments.

![Gradient Descent Visualization](https://raw.githubusercontent.com/yourusername/pytorch-ultimate/main/015_NeuralNetworkFromScratch/images/gradient_descent.png)
*Figure 2: Visualization of gradient descent optimization on a loss landscape, showing how parameters are updated to minimize loss.*

**The Learning Rate (`alpha`):**
Simply subtracting the full gradient might cause us to overshoot the minimum, potentially oscillating back and forth (`L1` to `L2` to `L1`...).
To control the step size, we introduce a **learning rate** (`alpha`), a small positive number (e.g., 0.01, 0.001).
*   **Weight Update Rule:** `new_weights = old_weights - alpha * gradient`
*   By multiplying the gradient by `alpha`, we take smaller, more controlled steps towards the minimum.

**The Chain Rule:**
Our network has multiple steps (Layer 1 -> Activation -> Loss). How do we calculate the gradient of the *loss* with respect to the *weights* (`dL/dw`), which are used much earlier?
We use the **Chain Rule** from calculus.
*   Example: Input `X` -> Function `Y(X)` -> Function `Z(Y)`.
*   To find `dZ/dX`, we multiply the derivatives of each step: `dZ/dX = dZ/dY * dY/dX`.

**Applying the Chain Rule to Our Network:**
To find `dL/dw` (the gradient needed to update weights):
1.  Calculate `dL/dy_pred`: Derivative of the Loss function w.r.t. predictions.
    *   For MSE: `dL/dy_pred = 2 * (y_pred - y_true)`.
2.  Calculate `dy_pred/dl1`: Derivative of the Activation function (Sigmoid) w.r.t. the output of layer 1.
3.  Calculate `dl1/dw`: Derivative of Layer 1's output w.r.t. the weights.
    *   This is simply the input `X`.

**Final Gradient for Weights:** `dL/dw = (dL/dy_pred) * (dy_pred/dl1) * (dl1/dw)`

**Gradient for Biases (`dL/db`):**
The process is similar, but the last derivative changes:
*   `dl1/db`: Derivative of Layer 1's output w.r.t. the bias.
    *   This is simply `1`.
**Final Gradient for Biases:** `dL/db = (dL/dy_pred) * (dy_pred/dl1) * (dl1/db)`

That covers the core mechanics of the backward pass.

### The Dot Product

One more technical aspect: the **dot product**.
Why use it? It efficiently calculates the weighted sum in our layers (`X * w`) and helps measure similarity between vectors (like input `X` and weight vector `w`).

Consider an input vector `X` and two weight vectors `w1` and `w2`. Which weight vector is more "aligned" or similar to the input?
*   Similarity depends on both the *magnitude* (length) of the vectors and the *angle* between them.
*   The dot product captures this relationship efficiently.

**Calculating the Dot Product:**
Given `X = [x1, x2, ...]` and `w1 = [w1_1, w1_2, ...]`:
*   Dot Product (`X . w1`) = `(x1 * w1_1) + (x2 * w1_2) + ...`
*   It's the sum of the element-wise products.
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

Hello and welcome. Let's implement the dot product calculation using Python and NumPy. You can find the code in the `015_NeuralNetworkFromScratch` subfolder.

We only need NumPy:
```python
import numpy as np
```

Let's define sample input and weight vectors:
```python
X = np.array([1, 2, 3])
w1 = np.array([0.2, 0.8, -0.5])
w2 = np.array([-0.5, 0.1, 0.9])
```

**Manual Calculation:**
Calculating the dot product manually for `X` and `w1`:
```python
dot_X_w1_manual = (X[0] * w1[0]) + (X[1] * w1[1]) + (X[2] * w1[2])
# -> (1 * 0.2) + (2 * 0.8) + (3 * -0.5)
# -> 0.2 + 1.6 - 1.5 = 0.3
print(dot_X_w1_manual) # Output: 0.3
```
This is tedious for larger vectors.

Similarly for `X` and `w2`:
```python
dot_X_w2_manual = (X[0] * w2[0]) + (X[1] * w2[1]) + (X[2] * w2[2])
# -> (1 * -0.5) + (2 * 0.1) + (3 * 0.9)
# -> -0.5 + 0.2 + 2.7 = 2.4
print(dot_X_w2_manual) # Output: 2.4
```
Comparing the results (0.3 vs 2.4), `w2` has a larger dot product with `X`, suggesting more similarity in this case.

**NumPy's `dot` Function:**
NumPy provides a much simpler way: `np.dot()`.
```python
dot_X_w1_numpy = np.dot(X, w1)
print(dot_X_w1_numpy) # Output: 0.3

dot_X_w2_numpy = np.dot(X, w2)
print(dot_X_w2_numpy) # Output: 2.4
```
Perfect! This matches our manual calculation and is much more concise.

This `np.dot` function is the final piece we need before building the full network from scratch, which we'll start in the next lecture.

**Section Summary:**
> **Key Takeaways from Dot Product:**
> - Dot products calculate the sum of element-wise multiplication between vectors
> - NumPy's `np.dot()` provides an efficient implementation for vectors and matrices
> - This operation is fundamental to neural network calculations
> - Larger dot product values often indicate greater similarity between vectors

---

## Neural Network from Scratch (Data Prep)

Hello and welcome. Now we begin developing our neural network from scratch. We'll start with data preparation. I've prepared a script (`datascript`) for imports.

**Imports:**
```python
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
```

**Dataset:**
We'll use a heart attack prediction dataset (available on Kaggle).
*   It's a classification problem.
*   Features include age, sex, and other health metrics.
*   The target variable (`output`) is binary (0: less chance, 1: higher chance of heart attack).

**Loading Data:**
Assuming the data (`heart.csv`) is in the working directory:
```python
df = pd.read_csv('heart.csv')
print(df.head())
```
We can see the features and the `output` column.

**Separating Features and Target:**
We need to separate independent features (`X`) from the dependent target variable (`Y`).
```python
X = df.drop('output', axis=1).values # Convert to NumPy array
Y = df['output'].values           # Convert to NumPy array
```
Check shapes:
```python
print("X shape:", X.shape) # E.g., (303, 13) -> 303 observations, 13 features
print("Y shape:", Y.shape) # E.g., (303,) -> 303 target values
```

**Train/Test Split:**
We split the data to train the model and evaluate it on unseen data.
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, Y,
    test_size=0.2,    # 20% for testing
    random_state=42   # For reproducibility
)
```

**Feature Scaling:**
Neural networks often perform better when input features are scaled. Different features have varying ranges (e.g., age vs. cholesterol).
We'll use `StandardScaler` (transforms data to have mean=0, stddev=1).
```python
scaler = StandardScaler()

# Fit on training data ONLY and transform it
X_train_scale = scaler.fit_transform(X_train)

# Use the SAME scaler (fitted on train data) to transform test data
X_test_scale = scaler.transform(X_test)
```
**Important:** Fit the scaler *only* on the training data to avoid data leakage from the test set.

With data prepared and scaled, we can now start building the neural network model itself.

**Section Summary:**
> **Key Takeaways from Data Preparation:**
> - Split data into features (X) and target (Y) variables
> - Further divide into training and testing sets using `train_test_split`
> - Scale features using `StandardScaler` to normalize different feature ranges
> - Apply the scaler only to training data first, then use the same scaler for test data
> - Properly preprocessed data is crucial for neural network performance

---

## Neural Network from Scratch Modeling: `__init__` Function

We'll structure our network within a Python class for better organization, similar to how PyTorch models are often defined.

```python
import numpy as np

class NeuralNetworkFromScratch:
    def __init__(self, learning_rate, X_train, y_train, X_test, y_test):
        """
        Initializes the Neural Network.
        Args:
            learning_rate (float): The step size for gradient descent.
            X_train, y_train (np.array): Scaled training data and labels.
            X_test, y_test (np.array): Scaled test data and labels.
        """
        # Parameters passed during instantiation

        # --- Initialize Weights and Biases ---
        # Randomly initialize weights. One weight per input feature.
        num_features = X_train.shape[1]
        self.weights = np.random.randn(num_features)
        # Initialize bias (single value for this simple network)
        self.bias = np.random.randn(1)

        # --- Store Data and Parameters ---
        # Assign passed parameters to internal attributes
        self.learning_rate = learning_rate
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # --- Initialize Loss Tracking ---
        # Lists to store loss values during training
        self.L_train = [] # Training losses
        self.L_test = []  # Test losses

    # ... other methods (activation, forward, backward, etc.) will go here ...
```
This `__init__` method sets up the essential components: weights, bias, learning rate, data storage, and lists for tracking performance.

---

## Neural Network from Scratch Modeling: Helper Functions (Activation)

Now, let's add the helper functions needed within our class, starting with the activation function and its derivative.

```python
# (Inside the NeuralNetworkFromScratch class)

    # --- Activation Function (Sigmoid) ---
    def _sigmoid(self, x):
        """ Applies the Sigmoid activation function element-wise. """
        return 1 / (1 + np.exp(-x))

    # --- Derivative of Sigmoid ---
    def _sigmoid_derivative(self, x):
        """ Computes the derivative of the Sigmoid function. """
        sig_x = self._sigmoid(x)
        return sig_x * (1 - sig_x)
```
These functions encapsulate the Sigmoid calculation and its derivative, which we'll need for the forward and backward passes, respectively. We use a leading underscore (`_`) by convention to indicate these are intended primarily for internal use within the class.

---

## Neural Network from Scratch Modeling: Forward Function

Next, we define the `forward` pass function within our class. This function takes an input `X` and propagates it through the network to produce a prediction.

```python
# (Inside the NeuralNetworkFromScratch class)

    # --- Forward Pass ---
    def forward(self, X):
        """
        Performs the forward pass of the network.
        Args:
            X (np.array): Input data (a single sample or batch).
        Returns:
            np.array: Network's prediction(s).
        """
        # Step 1: Calculate weighted sum + bias (Linear Layer)
        hidden1 = np.dot(X, self.weights) + self.bias

        # Step 2: Apply activation function
        activated_output = self._sigmoid(hidden1)

        # Step 3: Return the final prediction
        return activated_output
```
This function implements the steps we discussed in the theory lecture: linear combination followed by the Sigmoid activation. It returns the network's output (prediction) for the given input `X`.

---

## Neural Network from Scratch Modeling: Backward Function

Now, we implement the `backward` pass, which calculates the gradients of the loss with respect to the weights and bias. These gradients tell us how to adjust the parameters to reduce the loss.

```python
# (Inside the NeuralNetworkFromScratch class)

    # --- Backward Pass (Gradient Calculation) ---
    def backward(self, X_sample, y_true_sample):
        """
        Performs the backward pass to calculate gradients for a single sample.
        Args:
            X_sample (np.array): A single input data sample.
            y_true_sample (float): The corresponding true label.
        Returns:
            tuple: Gradients (dL_db, dL_dw).
        """
        # --- Recompute steps from Forward Pass (needed for derivatives) ---
        # 1. Linear layer output
        hidden1 = np.dot(X_sample, self.weights) + self.bias
        # 2. Prediction (output after activation)
        y_pred = self._sigmoid(hidden1)

        # --- Calculate Gradients using Chain Rule ---
        # Derivative of Loss w.r.t. Prediction (dL/d_pred)
        dL_dpred = 2 * (y_pred - y_true_sample)

        # Derivative of Prediction w.r.t. Hidden Layer Output (d_pred/d_hidden1)
        dpred_dhidden1 = self._sigmoid_derivative(hidden1)

        # Derivative of Hidden Layer Output w.r.t Bias (d_hidden1/db)
        dhidden1_db = 1

        # Derivative of Hidden Layer Output w.r.t Weights (d_hidden1/dw)
        dhidden1_dw = X_sample

        # --- Combine derivatives using Chain Rule ---
        # Gradient for Bias (dL/db = dL/d_pred * d_pred/d_hidden1 * d_hidden1/db)
        dL_db = dL_dpred * dpred_dhidden1 * dhidden1_db

        # Gradient for Weights (dL/dw = dL/d_pred * d_pred/d_hidden1 * d_hidden1/dw)
        dL_dw_intermediate = dL_dpred * dpred_dhidden1
        dL_dw = dhidden1_dw * dL_dw_intermediate

        # Sum gradients if needed
        dL_db = np.sum(dL_db)

        # --- Return calculated gradients ---
        return dL_db, dL_dw
```
This function calculates the necessary gradients (`dL_db`, `dL_dw`) for a single data sample using the chain rule, breaking down the derivatives step-by-step. These gradients will then be used by the optimizer to update the network's parameters.

---

## Neural Network from Scratch Modeling: Optimizer Function

The `optimizer` function takes the calculated gradients and updates the network's weights and bias according to the gradient descent rule.

```python
# (Inside the NeuralNetworkFromScratch class)

    # --- Optimizer (Parameter Update) ---
    def optimizer(self, dL_db, dL_dw):
        """
        Updates the weights and bias using the calculated gradients.
        Args:
            dL_db (float): Gradient of the loss w.r.t. the bias.
            dL_dw (np.array): Gradient of the loss w.r.t. the weights.
        """
        # Update bias using gradient descent rule
        self.bias = self.bias - self.learning_rate * dL_db

        # Update weights using gradient descent rule
        self.weights = self.weights - self.learning_rate * dL_dw
```
This simple function implements the core update step of gradient descent. It subtracts a fraction (controlled by `self.learning_rate`) of the gradient from the current parameters (`self.bias`, `self.weights`), moving them in the direction that should decrease the loss.

---

## Neural Network from Scratch Modeling: Train Function

The `train` function orchestrates the learning process. It iterates multiple times (epochs or iterations), processing data samples, calculating losses, computing gradients, and updating parameters.

```python
# (Inside the NeuralNetworkFromScratch class)

    # --- Training Loop ---
    def train(self, iterations):
        """
        Trains the neural network for a specified number of iterations.
        Args:
            iterations (int): The number of training iterations.
        """
        for i in range(iterations):
            # --- Training Step on a Random Sample ---
            # Select a random data point from the training set
            random_pos = np.random.randint(0, len(self.X_train))
            X_sample = self.X_train[random_pos]
            y_true_sample = self.y_train[random_pos]

            # 1. Forward Pass (get prediction for the sample)
            y_train_pred = self.forward(X_sample)

            # 2. Calculate Training Loss (for this sample)
            train_loss = np.sum(np.square(y_train_pred - y_true_sample))
            self.L_train.append(train_loss)

            # 3. Backward Pass (calculate gradients for the sample)
            dL_db, dL_dw = self.backward(X_sample, y_true_sample)

            # 4. Optimize (update weights and bias)
            self.optimizer(dL_db, dL_dw)

            # --- Evaluation Step on Test Set (Optional, per iteration) ---
            if i % 50 == 0:
                test_loss_sum = 0
                for j in range(len(self.X_test)):
                    y_test_true = self.y_test[j]
                    y_test_pred = self.forward(self.X_test[j])
                    test_loss_sum += np.sum(np.square(y_test_pred - y_test_true))
                avg_test_loss = test_loss_sum / len(self.X_test)
                self.L_test.append(avg_test_loss)

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

Now that our `NeuralNetworkFromScratch` class is defined, let's instantiate it and run the training process.

**Hyperparameters:**
Set the learning rate and number of iterations.
```python
LEARNING_RATE = 0.01
ITERATIONS = 1000
```

**Instantiate the Model:**
Create an instance of our class, passing the hyperparameters and the prepared data.
```python
nn = NeuralNetworkFromScratch(
    learning_rate=LEARNING_RATE,
    X_train=X_train_scale,
    y_train=y_train,
    X_test=X_test_scale,
    y_test=y_test
)
```

**Run Training:**
Call the `train` method on the instance.
```python
nn.train(ITERATIONS)
```

---

## Neural Network from Scratch: Model Evaluation

After training, we need to evaluate how well our network performs, especially on the unseen test data.

**1. Plotting Losses:**
Visualize the test loss over iterations to see if the model learned effectively.
```python
import matplotlib.pyplot as plt
import seaborn as sns

iterations_recorded = range(0, ITERATIONS, 50)
x_axis = range(len(nn.L_test))

plt.figure(figsize=(10, 6))
sns.lineplot(x=x_axis, y=nn.L_test)
plt.title("Test Loss During Training")
plt.xlabel("Training Iteration (or Recording Step)")
plt.ylabel("Average Test Loss")
plt.show()
```
A decreasing loss curve that stabilizes indicates successful learning.

**2. Calculating Accuracy on Test Set:**
Iterate through the test set, get predictions, compare to true labels, and calculate accuracy.
```python
num_test_samples = X_test_scale.shape[0]
correct_predictions = 0
y_preds_list = []

for i in range(num_test_samples):
    X_sample = X_test_scale[i]
    y_true = y_test[i]

    y_pred_proba = nn.forward(X_sample)
    y_pred_binary = np.round(y_pred_proba).astype(int)

    y_preds_list.append(y_pred_binary.item())

    if y_pred_binary == y_true:
        correct_predictions += 1

accuracy = correct_predictions / num_test_samples
print(f"Test Accuracy: {accuracy:.2%}")
```

**3. Checking Baseline Accuracy:**
Compare the model's accuracy to a naive baseline (e.g., always predicting the majority class).
```python
from collections import Counter

y_test_counts = Counter(y_test)
majority_class_count = max(y_test_counts.values())
baseline_accuracy = majority_class_count / num_test_samples
print(f"Baseline (Majority Class) Accuracy: {baseline_accuracy:.2%}")
```
If our model's accuracy is significantly higher than the baseline, it indicates the model has learned meaningful patterns.

**4. Confusion Matrix:**
Provides a more detailed breakdown of correct and incorrect predictions for each class.
```python
from sklearn.metrics import confusion_matrix

y_preds_array = np.array(y_preds_list)

cm = confusion_matrix(y_test, y_preds_array)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
```
The matrix shows True Positives, True Negatives, False Positives, and False Negatives, giving insight into the types of errors the model makes.

**Experimenting with Hyperparameters:**
You can now experiment by changing `LEARNING_RATE` or `ITERATIONS` and retraining to see the impact on the loss curve and final accuracy.

**Summary & Next Steps:**
We have successfully:
*   Imported and preprocessed data (scaling, splitting).
*   Built a `NeuralNetworkFromScratch` class containing:
    *   `__init__`: Initialization.
    *   `_sigmoid`, `_sigmoid_derivative`: Activation functions.
    *   `forward`: Prediction generation.
    *   `backward`: Gradient calculation.
    *   `optimizer`: Parameter updates.
    *   `train`: Training loop orchestration.
*   Trained the network.
*   Evaluated performance using loss plots, accuracy, and a confusion matrix.

![Neural Network Complete Architecture](https://raw.githubusercontent.com/yourusername/pytorch-ultimate/main/015_NeuralNetworkFromScratch/images/complete_nn_architecture.png)
*Figure 3: Complete architecture of our neural network showing data flow, training process, and evaluation metrics.*

