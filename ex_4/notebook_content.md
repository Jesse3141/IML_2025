# Notebook Content: `ex_4/dev_ex4.ipynb`

## Cell 1 [Markdown]
```markdown
# Exercise 4: Machine Learning Methods

In this exercise, we will experiment with Multi-Layer Perceptron (MLP) and Convolutional Neural Network (CNN) models.
```

## Cell 2 [Code]
```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add current directory to path to import helpers if needed
sys.path.append(os.getcwd())
try:
    from helpers import *
except ImportError:
    print("helpers.py not found or error importing")
```

## Cell 3 [Markdown]
```markdown
## 2. Seeding
```

## Cell 4 [Code]
```python
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)
```

## Cell 5 [Markdown]
```markdown
# 5. Data
## 5.1 European Countries
The dataset consists of train.csv, validation.csv, test.csv. 
Columns: longitude (long), latitude (lat), country (label).
```

## Cell 6 [Code]
```python
class CountryDataset(Dataset):
    def __init__(self, split='train'):
        filename = f"{split}.csv"
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found. Make sure you are in the correct directory.")
            self.X = torch.empty(0, 2)
            self.y = torch.empty(0, dtype=torch.long)
            return
            
        self.data = pd.read_csv(filename)
        # Features: long, lat
        self.X = torch.tensor(self.data[['long', 'lat']].values, dtype=torch.float32)
        # Labels: country
        self.y = torch.tensor(self.data['country'].values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

## Cell 7 [Markdown]
```markdown
# 6. Multi-Layer Perceptrons
## 6.1 Optimization of an MLP
### 6.1.1 Task
Implement a training pipeline from scratch. 
Model: 6 layers (includes input but not output -> 7 nn.Linear instances). Activation: ReLU. Batch Norm before activation.
Default parameters provided in code.
```

## Cell 8 [Code]
```python
class MLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=5, hidden_dim=16, num_layers=6, use_batch_norm=False):
        super(MLP, self).__init__()
        layers = []
        
        # According to instructions: "6 layers, it includes the input layer but not the output layer (therefore, you'll initialize 7 nn.Linear instances in total)"
        # So we have:
        # Linear(input, hidden)
        # Linear(hidden, hidden) x (num_layers - 1)
        # Linear(hidden, output)
        # Total linear layers = 1 + (num_layers - 1) + 1 = num_layers + 1. 
        # If num_layers=6, then 7 linear layers.
        
        dims = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            # Add activation and BN for all but the last layer
            if i < len(dims) - 2:
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i+1]))
                layers.append(nn.ReLU())
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
```

## Cell 9 [Code]
```python
class MLPTrainer:
    def __init__(self, 
                 learning_rate=1e-2, 
                 num_epochs=50, 
                 batch_size=32, 
                 use_batch_norm=False,
                 hidden_dim=16,
                 num_layers=6,
                 seed=42):
        
        set_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Data
        self.train_ds = CountryDataset('train')
        self.val_ds = CountryDataset('validation')
        self.test_ds = CountryDataset('test')
        
        self.batch_size = batch_size
        self.train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_ds, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_ds, batch_size=batch_size, shuffle=False)
        
        # Model
        # Infer output dim from data labels (assuming 0 to N-1 labels)
        output_dim = len(torch.unique(self.train_ds.y))
        self.model = MLP(input_dim=2, output_dim=output_dim, 
                         hidden_dim=hidden_dim, num_layers=num_layers, 
                         use_batch_norm=use_batch_norm).to(self.device)
        
        # Optimization
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        
        self.num_epochs = num_epochs
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * X.size(0)
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            
            avg_train_loss = train_loss / total
            train_acc = correct / total
            
            # Validation
            val_loss, val_acc = self.evaluate(self.val_loader)
            
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # print(f"Epoch {epoch+1}/{self.num_epochs}: Train Loss {avg_train_loss:.4f}, Val Loss {val_loss:.4f}")
            
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                total_loss += loss.item() * X.size(0)
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        return total_loss / total, correct / total

    def plot_results(self):
        epochs = range(1, self.num_epochs + 1)
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_loss'], label='Train')
        plt.plot(epochs, self.history['val_loss'], label='Val')
        plt.title('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history['train_acc'], label='Train')
        plt.plot(epochs, self.history['val_acc'], label='Val')
        plt.title('Accuracy')
        plt.legend()
        plt.show()
```

## Cell 10 [Markdown]
```markdown
### 6.1.2 Questions
1. **Learning Rate**: Train with 1, 0.01, 0.001, 0.00001. Plot validation loss.
2. **Epochs**: Train for 100 epochs. Plot loss.
3. **Batch Norm**: Add batch norm. Compare.
4. **Batch Size**: 1, 16, 128, 1024.
```

## Cell 11 [Code]
```python
# Q1 Learning Rate
lrs = [1.0, 0.01, 0.001, 0.00001]
histories = {}
for lr in lrs:
    print(f"Training with LR={lr}")
    trainer = MLPTrainer(learning_rate=lr, num_epochs=50)
    trainer.train()
    histories[lr] = trainer.history['val_loss']

plt.figure()
for lr, loss in histories.items():
    plt.plot(loss, label=f'LR={lr}')
plt.legend()
plt.title('Validation Loss per LR')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
```

## Cell 12 [Markdown]
```markdown
## 6.2 Evaluating MLPs Performance
Train 6 classifiers for combinations of depth and width.
```

## Cell 13 [Code]
```python
# Combinations from Table 2
configs = [
    {'depth': 1, 'width': 16},
    {'depth': 2, 'width': 16},
    {'depth': 6, 'width': 16},
    {'depth': 10, 'width': 16},
    {'depth': 6, 'width': 8},
    {'depth': 6, 'width': 32},
    {'depth': 6, 'width': 64}
]

# Implement loop to train these models and store results
```

## Cell 14 [Markdown]
```markdown
# 7. Convolutional Neural Networks
## 7.4 Task
1. XGBoost
2. Training from Scratch (ResNet18)
3. Linear Probing
4. Sklearn Probing
5. Fine-tuning
```

## Cell 15 [Code]
```python
# Placeholders for CNN tasks
# You might need to import from cnn.py if you implement models there
# from cnn import *
```
