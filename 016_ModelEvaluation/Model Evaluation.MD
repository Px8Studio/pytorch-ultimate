# Model Evaluation in Machine Learning

## Introduction

Model evaluation is a critical step in the machine learning workflow that helps us understand how well our models perform and whether they can generalize to unseen data. In this module, we will focus on evaluating machine learning models. You will learn about underfitting and overfitting, and how to mitigate these issues. The module will also cover the train-test split method and its importance in model evaluation, along with various resampling techniques to manage imbalanced datasets effectively.

## Learning Objectives

By the end of this module, you will understand:
- The concepts of underfitting and overfitting and how to identify them
- How to properly split data into training, validation, and test sets
- Cross-validation techniques and their implementation
- Resampling methods for handling imbalanced datasets

## 1. Underfitting and Overfitting

Understanding the bias-variance tradeoff is fundamental to building effective machine learning models. 

### Key Concepts:
- **Underfitting**: When a model is too simple to capture the underlying patterns in the data
- **Overfitting**: When a model learns the training data too well, including its noise and outliers
- **Balanced fitting**: Finding the optimal complexity that generalizes well to new data

### Visual Understanding of Underfitting and Overfitting

#### Regression Example
- In **underfitting**, a linear model attempts to fit data with a non-linear (e.g., quadratic) relationship, resulting in high bias and large prediction errors
- In **overfitting**, an overly complex model captures all noise in the training data, showing nearly perfect fit but poor generalization ability

#### Classification Example
- In **underfitting**, a simple linear decision boundary fails to separate classes that require a non-linear boundary, resulting in high misclassification rates
- In **overfitting**, an extremely complex decision boundary perfectly separates all training points but incorporates noise rather than true patterns

### Bias-Variance Tradeoff

Think of this like dart players aiming at a target:
- **High bias, low variance**: The darts consistently hit the same area, but away from the bullseye
- **High bias, high variance**: The darts are scattered widely and also miss the target center
- **Low bias, high variance**: The darts are centered around the bullseye but widely scattered
- **Low bias, low variance**: The darts consistently hit the bullseye (ideal situation)

In machine learning terms:
- **Bias** is the difference between predictions and actual values on the training data
- **Variance** is the difference between performance on training data versus validation data

As model complexity increases:
- Bias decreases (better fit to training data)
- Variance increases (more likely to overfit)

### Algorithm Tendencies
- **Low variance algorithms**: Linear regression, LDA, logistic regression
- **High variance algorithms**: Decision trees, K-Nearest Neighbors, Support Vector Machines

### Detecting and Mitigating Overfitting

**Signs of overfitting:**
- Large gap between training and validation performance
- Model performs increasingly worse on validation data as training continues
- Extremely complex model structure relative to the dataset size

**Mitigation techniques:**
- **Regularization**: L1 (Lasso) and L2 (Ridge) regularization add penalties to the loss function to constrain model parameters
- **Early stopping**: Halt training when validation performance starts to degrade
- **Pruning**: Removing unnecessary complexity from models (especially decision trees)
- **Dropout**: Randomly deactivating neurons during training (for neural networks)
- **Data augmentation**: Artificially increasing the training dataset size with modified examples

The goal is to find the optimal model complexity that minimizes the sum of bias error and variance error.

> **Supplementary Reading**: For detailed explanations and visualizations of underfitting and overfitting concepts, refer to [Underfitting and Overfitting Guide](UnderfittingOverfitting.pdf)

## 2. Data Splitting Strategies

Before training any model, we need to properly split our data to enable robust evaluation.

### Common Approaches:
- **Training set**: Used to train the model (typically 60-80% of data)
- **Validation set**: Used for hyperparameter tuning and model selection
- **Test set**: Used only once to evaluate the final model

### Purpose of Each Split

- **Training data**: Data used to fit the model parameters
- **Validation data**: Data used to evaluate and fine-tune the model during development. The model occasionally sees this data but doesn't directly learn from it
- **Test data**: Provides an unbiased evaluation of the final model. This data should only be used once after the model is completely developed

### Stratified Splitting

When working with classification problems, especially imbalanced datasets, it's important to maintain the class distribution across all splits. This is achieved through stratified sampling, where the proportion of each class is preserved in training, validation, and test sets.

### Temporal Splitting

For time series data, random splits can lead to data leakage. Instead:
- Use past data for training
- More recent data for validation
- Most recent data for testing

This approach respects the temporal nature of the data and better simulates real-world forecasting scenarios.

### Determining Split Proportions

The optimal split depends on:
1. The total number of samples available
2. The complexity of the model
3. Number of hyperparameters to tune

Models with more hyperparameters require more validation data to properly tune. Similarly, more complex models may need larger training sets to learn effectively.

Train-validation-test splitting is considered the gold standard in machine learning evaluation.

> **Supplementary Reading**: See [Train-Validation-Test Split Guide](TrainValidationTest_101.pdf) for detailed procedures and best practices.

## 3. Cross-Validation Techniques

Cross-validation is a powerful method for assessing a model's performance and stability.

### The Resampling Challenge

Cross-validation addresses a fundamental problem in machine learning: we want to maximize data used for training to build the best model, but we also need sufficient data for validation to ensure reliable performance assessment.

### K-Fold Cross-Validation

In k-fold cross-validation, the data is divided into k equal parts (folds). The model is trained k times, each time using k-1 folds for training and the remaining fold for validation.

#### 5-Fold Cross-Validation Example:

In 5-fold cross-validation, the data is divided into five equal parts. For example, with 500 observations:

1. **First iteration**: The first 100 observations (20%) are reserved for validation, and the remaining 80% for training.
   - Model 1 is trained on observations 101-500
   - Performance is evaluated on observations 1-100

2. **Second iteration**: Observations 101-200 are used for validation, the rest for training.
   - Model 2 is trained on observations 1-100 and 201-500
   - Performance is evaluated on observations 101-200

3. **This process repeats** until we have trained 5 different models and obtained 5 different performance results.

After completing all iterations, the performance metrics from each fold are averaged to produce a final performance estimate. For example, if the R² values for the five folds are 0.25, 0.28, 0.30, 0.27, and 0.25, the average performance would be 0.27.

It's important to note that in practice, data isn't taken in sequence as shown in this example. Instead, observations are typically randomly assigned to folds to ensure that the target variable is properly distributed across all folds.

#### Stratified K-Fold Cross-Validation

This variation of k-fold CV ensures that each fold maintains the same class distribution as the overall dataset, which is particularly important for imbalanced classification problems.

#### 10-Fold Cross-Validation

Ten-fold cross-validation is commonly used as it offers a good compromise between computational effort and model performance assessment. Each fold contains 10% of the data.

#### Leave-One-Out Cross-Validation (LOOCV)

In LOOCV, we create as many folds as there are data points. Each model is trained on all data except one observation, which is used for validation.

**Pros**: Provides the most stable and reliable results  
**Cons**: Computationally very expensive, especially with large datasets

### Time Series Cross-Validation

For time series data, specialized cross-validation techniques preserve the temporal order:

- **Forward chaining**: Train on a fixed window, predict the next time period, then expand the window
- **Rolling-origin cross-validation**: Use a fixed-size window that moves forward through time

### Nested Cross-Validation

When tuning hyperparameters and evaluating model performance, nested CV uses an inner loop for hyperparameter tuning and an outer loop for model evaluation.

### Advantages of Cross-Validation

1. Provides more stable and reliable performance estimates
2. Helps compare different machine learning algorithms effectively
3. Useful for assessing the impact of hyperparameter tuning
4. Utilizes data more efficiently than a simple train-test split

### Disadvantages of Cross-Validation

1. Computationally expensive, especially with large datasets or complex models
2. May still not fully represent real-world performance if the data distribution shifts
3. Can be challenging to implement correctly for time series or hierarchical data

## 4. Resampling Techniques for Imbalanced Data

When dealing with imbalanced datasets (where some classes are much more frequent than others), special techniques are needed to ensure models don't simply predict the majority class.

### The Problem of Imbalanced Data

When one class significantly outnumbers others, models tend to favor the majority class, potentially ignoring the minority class entirely. This is problematic when the minority class is often what we're most interested in detecting (e.g., fraud in financial transactions, rare diseases in medical diagnosis).

### Common Approaches:
- **Undersampling**: Reducing the number of observations from the majority class
- **Oversampling**: Increasing the number of observations from the minority class
- **SMOTE** (Synthetic Minority Over-sampling Technique): Creating synthetic examples of the minority class
- **Ensemble methods**: Combining multiple models trained on differently resampled datasets

### Undersampling Techniques

1. **Random Undersampling**: Randomly remove samples from the majority class
2. **Tomek Links**: Remove majority samples that form Tomek links with minority samples (pairs from different classes that are each other's nearest neighbor)
3. **Edited Nearest Neighbors (ENN)**: Remove samples whose class differs from the majority of their k nearest neighbors
4. **Neighborhood Cleaning Rule (NCR)**: Combines ENN with selective undersampling

### Oversampling Techniques

1. **Random Oversampling**: Randomly duplicate minority class samples
2. **SMOTE (Synthetic Minority Oversampling Technique)**: Create synthetic samples along the line segments connecting minority class neighbors
3. **ADASYN (Adaptive Synthetic Sampling)**: Similar to SMOTE but focuses on harder-to-learn examples
4. **Borderline-SMOTE**: Creates synthetic samples near the decision boundary rather than throughout the feature space

### Hybrid Methods

1. **SMOTE + Tomek**: Apply SMOTE oversampling followed by Tomek links undersampling
2. **SMOTE + ENN**: SMOTE followed by Edited Nearest Neighbors cleaning

### Algorithm-Level Approaches

1. **Cost-sensitive learning**: Assign higher misclassification costs to the minority class
2. **Class weights**: Adjust the weight of classes in the learning algorithm's objective function
3. **Specialized algorithms**: Some algorithms like XGBoost have built-in mechanisms to handle imbalanced data

### Evaluation Metrics for Imbalanced Data

Standard accuracy is misleading for imbalanced datasets. Instead, use:
- **Precision and Recall**: Focus on the minority class performance
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the receiver operating characteristic curve
- **Precision-Recall AUC**: Better for highly imbalanced datasets
- **Balanced Accuracy**: The average of recall obtained on each class

### Implementation Considerations

When applying resampling techniques:
- Be careful not to introduce bias through the resampling process
- Apply resampling only to training data, never to validation or test data
- Consider the domain context when choosing between undersampling and oversampling
- Evaluate models using metrics appropriate for imbalanced datasets (F1-score, precision-recall AUC, etc.)
- Monitor for potential overfitting, especially with oversampling techniques

> **Supplementary Reading**: For comprehensive information on handling imbalanced datasets, see [Resampling Techniques Guide](ResamplingTechniques_101.pdf)

## 5. Practical Implementation in PyTorch

### Detecting Overfitting

```python
def plot_learning_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Learning Curves')
    plt.show()
    
    # Overfitting detection
    divergence_point = np.argmin(val_losses)
    if divergence_point < len(val_losses) - 1:
        print(f"Potential overfitting detected after epoch {divergence_point}")
        print(f"Consider early stopping at epoch {divergence_point}")
    else:
        print("No clear overfitting detected")
```

### Train-Validation-Test Split with PyTorch

```python
from sklearn.model_selection import train_test_split
from torch.utils.data import SubsetRandomSampler, DataLoader

def create_data_loaders(dataset, batch_size=32, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """Create train, validation and test data loaders from a single dataset."""
    
    # Generate indices for the split
    indices = list(range(len(dataset)))
    
    # First split: temporary train and test
    temp_train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    
    # Second split: train and validation from the temporary train
    train_idx, val_idx = train_test_split(
        temp_train_idx, test_size=val_size/(train_size+val_size), random_state=random_state
    )
    
    # Create data samplers
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    
    # Create the data loaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    
    return train_loader, val_loader, test_loader
```

### K-Fold Cross-Validation with PyTorch

```python
from sklearn.model_selection import KFold
import numpy as np
import torch

def k_fold_cross_validation(dataset, model_class, k=5, batch_size=32, epochs=10, lr=0.001):
    """Perform k-fold cross-validation with PyTorch models."""
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_performances = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Training fold {fold+1}/{k}...")
        
        # Create data loaders for this fold
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
        
        # Initialize model and optimizer
        model = model_class()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Train the model
        for epoch in range(epochs):
            # Training code here
            # ...
        
        # Evaluate on validation set
        model.eval()
        val_performance = evaluate_model(model, val_loader)
        fold_performances.append(val_performance)
        
    # Calculate average performance across folds
    avg_performance = np.mean(fold_performances)
    std_performance = np.std(fold_performances)
    
    print(f"Cross-validation performance: {avg_performance:.4f} ± {std_performance:.4f}")
    return fold_performances
```

### Handling Imbalanced Data with PyTorch

```python
from torch.utils.data import WeightedRandomSampler

def create_balanced_sampler(dataset):
    """Create a weighted sampler to handle imbalanced datasets."""
    
    # Assuming dataset.targets contains the labels
    targets = dataset.targets
    class_counts = torch.bincount(targets)
    class_weights = 1. / class_counts.float()
    
    # Assign weights to each sample
    sample_weights = class_weights[targets]
    
    # Create a weighted sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler

# Usage
balanced_sampler = create_balanced_sampler(dataset)
balanced_loader = DataLoader(
    dataset, 
    batch_size=32, 
    sampler=balanced_sampler
)
```

## Summary

Proper model evaluation is essential for developing reliable and robust machine learning systems. By understanding concepts like overfitting, using appropriate data splitting techniques, implementing cross-validation, and handling data imbalances, you can build models that perform well not just on training data but in real-world applications as well.

## Additional Resources

- The PDF guides provided with this module offer more detailed explanations and visual examples
- Practice implementing these techniques with the code examples in the accompanying notebooks
- Consider how these evaluation methods might be applied to your specific machine learning projects