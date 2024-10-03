"""
Script: xor
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the XOR inputs and targets
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Convert numpy arrays to PyTorch tensors
X = torch.from_numpy(X)
y = torch.from_numpy(y)


# Define a simple feedforward neural network model
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # Two input features, two hidden units
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation as its own layer
        self.fc2 = nn.Linear(2, 1)  # One output (for binary classification)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# Instantiate the model, define loss function and optimizer
model = XORModel()
criterion = nn.BCELoss()  # Binary cross-entropy loss for binary classification
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(10000):  # Training for 10,000 epochs
    optimizer.zero_grad()  # Zero the gradient buffers
    outputs = model(X)  # Forward pass
    loss = criterion(outputs, y)  # Compute the loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch [{epoch+1}/10000], Loss: {loss.item():.4f}")

# Test the model
with torch.no_grad():
    predicted = model(X)
    predicted = (predicted > 0.5).float()  # Convert probabilities to binary output
    print("Predicted:\n", predicted)
    print("Actual:\n", y)
