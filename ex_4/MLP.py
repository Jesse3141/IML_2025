import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from helpers import *
import pandas as pd

class EuropeDataset(Dataset):
    def __init__(self, split='train'):
        """
        Args:
            split (string): 'train', 'validation', or 'test'
        """
        #### YOUR CODE HERE ####
        # Load the data into a tensors
        # The features shape is (n,d)
        # The labels shape is (n)
        # The feature dtype is float
        # THe labels dtype is long
        csv_file = f"{split}.csv"
        data = pd.read_csv(csv_file)
        self.X = torch.tensor(data[['long', 'lat']].values, dtype=torch.float)
        self.y = torch.tensor(data['country'].values, dtype=torch.long)
        # Aliases for skeleton compatibility
        self.features = self.X
        self.labels = self.y
        #### END OF YOUR CODE ####

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        #### YOUR CODE HERE ####
        return len(self.X)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data row

        Returns:
            dictionary or list corresponding to a feature tensor and it's corresponding label tensor
        """
        #### YOUR CODE HERE ####
        return self.X[idx], self.y[idx]
    

class MLP(nn.Module):
    def __init__(self, num_hidden_layers, hidden_dim, output_dim, input_dim=2, use_batch_norm=False):
        super(MLP, self).__init__()
        """
        Args:
            num_hidden_layers (int): The number of hidden layers, in total you'll have an extra layer at the end, from hidden_dim to output_dim
            hidden_dim (int): The hidden layer dimension
            output_dim (int): The output dimension, should match the number of classes in the dataset
            input_dim (int): The input dimension (default 2 for lon/lat)
            use_batch_norm (bool): Whether to use batch normalization
        """
        #### YOUR CODE HERE ####
        layers = []
        dims = [input_dim] + [hidden_dim] * num_hidden_layers + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation/BN after final layer
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        #### YOUR CODE HERE ####
        return self.model(x)


class MLPTrainer:
    """Trainer for MLP experiments."""
    def __init__(self,
                 train_dataset=None,
                 val_dataset=None,
                 test_dataset=None,
                 model=None,
                 lr=0.001,
                 epochs=50,
                 batch_size=256,
                 use_batch_norm=False,
                 hidden_dim=16,
                 num_layers=6,
                 seed=42,
                 grad_monitor_layers=None,
                 verbose=True):

        set_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose

        # Data - use provided datasets or load defaults
        self.train_ds = train_dataset if train_dataset is not None else EuropeDataset('train')
        self.val_ds = val_dataset if val_dataset is not None else EuropeDataset('validation')
        self.test_ds = test_dataset if test_dataset is not None else EuropeDataset('test')

        self.batch_size = batch_size
        self.train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(self.val_ds, batch_size=1024, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(self.test_ds, batch_size=1024, shuffle=False, num_workers=0)

        # Model - use provided model or create default
        if model is not None:
            self.model = model.to(self.device)
        else:
            output_dim = len(torch.unique(self.train_ds.y))
            self.model = MLP(num_hidden_layers=num_layers, hidden_dim=hidden_dim, output_dim=output_dim,
                             input_dim=2, use_batch_norm=use_batch_norm).to(self.device)

        # Optimization
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

        self.num_epochs = epochs

        # History
        self.history = {
            'train_loss': [], 'val_loss': [], 'test_loss': [],
            'train_acc': [], 'val_acc': [], 'test_acc': [],
            'batch_losses': [],
            'grad_norms': {}
        }

        # Gradient Monitoring Setup
        self.grad_monitor_layers = grad_monitor_layers
        self.monitored_linear_layers = {}

        if self.grad_monitor_layers is not None:
            for layer_idx in self.grad_monitor_layers:
                self.history['grad_norms'][layer_idx] = []

            all_linear_layers = [m for m in self.model.model if isinstance(m, nn.Linear)]

            for idx in self.grad_monitor_layers:
                if 0 <= idx < len(all_linear_layers):
                    self.monitored_linear_layers[idx] = all_linear_layers[idx]
                else:
                    print(f"Warning: Gradient monitor requested layer {idx}, "
                          f"but model only has {len(all_linear_layers)} linear layers.")

    def _maybe_update_grad_history(self, epoch_grad_accumulators):
        if not self.grad_monitor_layers:
            return

        for idx, layer in self.monitored_linear_layers.items():
            if layer.weight.grad is not None:
                grad_mag = layer.weight.grad.norm(2).item() ** 2
                epoch_grad_accumulators[idx] += grad_mag

    def train(self):
        """
        Train the model and return results in skeleton-compatible format.

        Returns:
            tuple: (model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses)
        """
        for epoch in tqdm(range(self.num_epochs), desc="Epochs", leave=False, disable=not self.verbose):
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            epoch_grad_accumulators = {idx: 0.0 for idx in self.grad_monitor_layers} if self.grad_monitor_layers else None
            num_batches = 0

            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                loss.backward()

                if self.grad_monitor_layers:
                    self._maybe_update_grad_history(epoch_grad_accumulators)
                    num_batches += 1

                self.optimizer.step()

                batch_loss = loss.item()
                self.history['batch_losses'].append(batch_loss)
                train_loss += batch_loss * X.size(0)
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

            if self.grad_monitor_layers and num_batches > 0:
                for idx in self.grad_monitor_layers:
                    avg_mag = epoch_grad_accumulators[idx] / num_batches
                    self.history['grad_norms'][idx].append(avg_mag)

            avg_train_loss = train_loss / total
            train_acc = correct / total

            val_loss, val_acc = self.evaluate(self.val_loader)
            test_loss, test_acc = self.evaluate(self.test_loader)

            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)

            if self.verbose:
                print('Epoch {:}, Train Acc: {:.3f}, Val Acc: {:.3f}, Test Acc: {:.3f}'.format(
                    epoch, train_acc, val_acc, test_acc))

        # Return in skeleton-compatible format
        return (self.model,
                self.history['train_acc'],
                self.history['val_acc'],
                self.history['test_acc'],
                self.history['train_loss'],
                self.history['val_loss'],
                self.history['test_loss'])

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

        if total == 0:
            return 0.0, 0.0
        return total_loss / total, correct / total


def train(train_dataset, val_dataset, test_dataset, model, lr=0.001, epochs=50, batch_size=256, verbose=True):
    """
    Wrapper function matching original skeleton signature.

    Returns:
        tuple: (model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses)
    """
    trainer = MLPTrainer(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        model=model,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )
    return trainer.train()



if __name__ == '__main__':
    # seed for reproducibility
    torch.manual_seed(0)    

    train_dataset = EuropeDataset('train.csv')
    val_dataset = EuropeDataset('validation.csv')
    test_dataset = EuropeDataset('test.csv')

    #### YOUR CODE HERE #####
    # Find the number of classes, e.g.:
    # output_dim = len(train_dataset.labels.unique()) 
    model = MLP(6, 16, output_dim)
    


    model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = train(train_dataset, val_dataset, test_dataset, model, lr=0.001, epochs=50, batch_size=256)

    plt.figure()
    plt.plot(train_losses, label='Train', color='red')
    plt.plot(val_losses, label='Val', color='blue')
    plt.plot(test_losses, label='Test', color='green')
    plt.title('Losses')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_accs, label='Train', color='red')
    plt.plot(val_accs, label='Val', color='blue')
    plt.plot(test_accs, label='Test', color='green')
    plt.title('Accs.')
    plt.legend()
    plt.show()



    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')
    plot_decision_boundaries(model, test_data[['long', 'lat']].values, test_data['country'].values, 'Decision Boundaries', implicit_repr=False)