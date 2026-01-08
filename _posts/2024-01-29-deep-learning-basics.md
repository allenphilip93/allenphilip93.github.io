---
layout: post
title: "Deep Learning Basics"
date: 2024-01-29
author: "Allen Philip J"
description: "A practical introduction to deep learning concepts, from neural networks to CNNs, with PyTorch examples."
tags: [Deep Learning, PyTorch, ML]
katex: true
---

Deep learning is a class of machine learning algorithms that uses *multiple stacked layers of processing units* to learn high-level representations from *unstructured* data.

Deep learning differs from traditional ML models in the sense that it can learn how to build high-level information by itself from unstructured data. While ML models like XGBoost, logistic regression, etc. rely on the input features to be informative and not spatially dependent like pixels in an image.

## Deep Neural Networks

### What is a Neural Network?

A neural network consists of a series of stacked *layers*. Each layer contains *units* that are connected to the previous layer's units through a set of *weights*. Neural networks where all adjacent layers are fully connected are called *multilayer perceptrons* (MLPs).

The input (e.g., an image) is transformed by each layer in turn, in what is known as a *forward pass* through the network, until it reaches the output layer. Specifically, each unit applies a nonlinear transformation to a weighted sum of its inputs and passes the output through to the subsequent layer.

The magic of deep neural networks lies in finding the set of weights for each layer that results in the most accurate predictions.

During the training process, batches of images are passed through the network and the predicted outputs are compared to the ground truth. The error in the prediction is then propagated backward through the network, adjusting each set of weights a small amount in the direction that improves the prediction most significantly. This process is called *backpropagation*.

Gradually, each unit becomes skilled at identifying a particular feature that ultimately helps the network make better predictions.

### Learning High-Level Features

The critical property that makes neural networks so powerful is their ability to learn features from the input data, without human guidance. We do not need to do any feature engineering, which is why neural networks are so useful! We can let the model decide how it wants to arrange its weights, guided only by its desire to minimize the error in its predictions.

## MLP on CIFAR-10 Dataset

### Multi Layer Perceptron (MLP)

It is a type of feedforward artificial neural network (ANN) composed of multiple layers of nodes (neurons), organized in a sequence of interconnected layers: an input layer, one or more hidden layers, and an output layer.

Key characteristics of MLPs:

1. **Feedforward Network**: Information flows in one direction, from the input layer through the hidden layers to the output layer. No cycles or loops in the connections.

2. **Fully Connected Layers**: Each node in one layer is connected to every node in the subsequent layer, allowing MLPs to capture complex relationships in the data.

3. **Non-linear Activation Functions**: Each node typically applies a non-linear activation function, enabling the network to learn complex patterns.

4. **Training with Backpropagation**: MLPs are trained using backpropagation, calculating gradients of the loss function with respect to the weights and adjusting using gradient descent.

5. **Universal Function Approximators**: MLPs with a single hidden layer can theoretically approximate any continuous function to arbitrary accuracy, given sufficiently many neurons.

### CIFAR-10 Dataset

The CIFAR-10 dataset is a widely used benchmark in computer vision. It consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

### Model Definition

A sample MLP can be defined in PyTorch as follows:

```python
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
```

Breaking down the model:
- The first layer flattens the 3-D input to a 1-D input
- This is passed to a dense/linear layer with neurons—each arrow has a weight, and each neuron's output is the weighted sum of inputs multiplied by weights
- The output passes through an activation layer which adds non-linearity, allowing the model to learn complex patterns[^1]

[^1]: Without non-linear activation layers, the neural network could only model linear relationships. It has been mathematically proven that neural networks with non-linear activation functions are universal function approximators.

We use ReLU and softmax for activation:
- **ReLU** (rectified linear unit) is defined to be 0 if the input is negative and is otherwise equal to the input
- ReLU units can sometimes die if they always output 0, because of a large bias toward negative values. LeakyReLU fixes this by always ensuring the gradient is nonzero
- **Sigmoid** activation scales the output between 0 and 1

### Loss Functions

The *loss function* is used by the neural network to compare its predicted output to the ground truth. It returns a single number for each observation; the greater this number, the worse the network has performed.

**Mean Squared Error** (for regression):

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Categorical Cross-Entropy** (for multi-class classification):

$$\text{CCE} = -\sum_{i} y_i \log(\hat{y}_i)$$

**Binary Cross-Entropy** (for binary/multilabel classification):

$$\text{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left( y_i \cdot \log(\hat{y}_i) + (1 - y_i) \cdot \log(1 - \hat{y}_i) \right)$$

### Optimizers

The *optimizer* is the algorithm used to update the weights based on the gradient of the loss function. One of the most commonly used is **Adam** (Adaptive Moment Estimation).

In most cases, you shouldn't need to tweak the default parameters of Adam except the *learning rate*. The greater the learning rate, the larger the change in weights at each training step. While training is initially faster with a large learning rate, it may result in less stable training and may not find the global minimum.

### Training Code

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 32 * 32 * 3  # CIFAR-10 image size is 32x32x3
hidden_size = 100
num_classes = 10
learning_rate = 0.001
num_epochs = 5
batch_size = 100

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = MLP(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Key training steps:

- `optimizer.zero_grad()` — Clears gradients from the previous minibatch. In PyTorch, gradients accumulate by default.
- `loss.backward()` — Computes gradients of the loss with respect to all parameters that have `requires_grad=True`.
- `optimizer.step()` — Updates parameters in the direction that minimizes the loss.

### Evaluation

Final evaluation of the trained MLP on the test dataset shows an accuracy of **49.96%**. Let's see if we can do better with a CNN.

## CNN on CIFAR-10 Dataset

One of the reasons our network isn't yet performing well is because there isn't anything in the network that takes into account the spatial structure of the input images. In fact, our first step is to flatten the image into a single vector!

### Convolutional Layers

A convolutional layer uses *filters* (or *kernels*) that slide across the image. The convolution is performed by multiplying the filter pixelwise with the portion of the image, and summing the results. The output is more positive when the portion of the image closely matches the filter.

If we move the filter across the entire image from left to right and top to bottom, recording the convolutional output as we go, we obtain a new array that picks out a particular feature of the input.

A convolutional layer is simply a collection of filters, where the values stored in the filters are the weights learned by the neural network through training. Initially these are random, but gradually the filters adapt their weights to start picking out interesting features such as edges or particular color combinations.

```python
nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
```

- **in_channels**: Number of input channels (3 for RGB)
- **out_channels**: Number of filters applied
- **kernel_size**: Size of the convolutional filter (3x3)
- **stride**: Number of pixels the filter moves (1 = one pixel at a time)
- **padding**: Pixels added to preserve spatial dimensions

### Batch Normalization

One common problem when training a deep neural network is ensuring that the weights remain within a reasonable range—if they start to become too large, this is a sign of the *exploding gradient* problem.

One of the reasons for scaling input data is to ensure a stable start to training. Because the input is scaled, it's natural to expect the activations from all future layers to be well scaled too. But as the network trains, this assumption can break down. This phenomenon is known as *covariate shift*.

During training, a batch normalization layer calculates the mean and standard deviation of each of its input channels across the batch and normalizes by subtracting the mean and dividing by the standard deviation.

```python
nn.BatchNorm2d(32, momentum=0.9)
```

### Dropout

Dropout layers are a common regularization technique. During training, each dropout layer chooses a random set of units from the preceding layer and sets their output to 0.

Dropout layers are used most commonly after dense layers since these are the most prone to overfitting due to the higher number of weights.

### CNN Model Definition

```python
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.LeakyReLU()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.relu5 = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.flatten(x)
        x = self.relu5(self.bn5(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN(num_classes=10)
```

### Evaluation

Final evaluation of the trained CNN on the test dataset shows an accuracy of **75.52%**.

This improvement has been achieved simply by changing the architecture to include convolutional, batch normalization, and dropout layers. Notice that the number of parameters is actually fewer in the CNN than the MLP, even though the number of layers is far greater.

This demonstrates the importance of being experimental with your model design and being comfortable with how different layer types can be used to your advantage. When building generative models, it becomes even more important to understand the inner workings of your model since it is the middle layers of your network that capture the high-level features that you are most interested in.
