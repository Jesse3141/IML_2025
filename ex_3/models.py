import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import pdb

class Ridge_Regression:

    def __init__(self, lambd):
        self.lambd = lambd

    def fit(self, X, Y):

        """
        Fit the ridge regression model to the provided data.
        :param X: The training features.
        :param Y: The training labels.
        """

        Y = 2 * (Y - 0.5) # transform the labels to -1 and 1, instead of 0 and 1.

        ########## YOUR CODE HERE ##########
        # add a feature of 1's for bias
        N = X.shape[1]
        X = np.vstack([X, np.ones((1, N))])
        # compute the ridge regression weights using the formula from class / exercise.
        # you may not use np.linalg.solve, but you may use np.linalg.inv
        feat_dim, N = X.shape # feat_dim is d+1 (includes bias)

        # Term 1: (XX^T / N + lambda * I)^-1
        XXT = X @ X.T
        I = np.eye(feat_dim)
        term1 = np.linalg.inv((XXT / N) + self.lambd * I)
        
        # Term 2: (XY^T / N)
        term2 = (X @ Y.T) / N
        self.W = term1 @ term2

        ####################################
        pass

    def predict(self, X):
        """
        Predict the output for the provided data.
        :param X: The data to predict. 
        :return: The predicted output. 
        we want to calculate W @ X, where X = [n_feats + 1, N]
        since W.shape = 3,1, 
        we can take its transpose and multiply by X to get ouput in [1,N]
        """
        N = X.shape[1]
        X =  np.vstack([X, np.ones((1, N))])
        preds = self.W.T @ X
        preds = np.where(preds < 0, 0,1)


        return preds
        
class Logistic_Regression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim,output_dim)



    def forward(self, x):
        """
        Computes the output of the linear operator.
        :param x: The input to the linear operator.
        :return: The transformed input.
        """
        logits = self.linear(x)
        
        return logits

    def predict(self, x):
        """
        THIS FUNCTION IS NOT NEEDED FOR PYTORCH. JUST FOR OUR VISUALIZATION
        """
        with torch.no_grad():
            x = torch.from_numpy(x).float().to(self.linear.weight.device)
            logits = self.forward(x)
            probs = nn.functional.softmax(logits, dim=1)
            return torch.argmax(probs, dim=1).detach().cpu().numpy()


#### cutsom code relating to models

##### pytorch helpers
def get_loaders(train_df, val_df, test_df, batch_size=32):
    def df_to_tensor(df):
        # Assumes format: [long, lat, country_id]
        x = torch.tensor(df.iloc[:, :2].values, dtype=torch.float32)
        y = torch.tensor(df.iloc[:, 2].values, dtype=torch.long)
        return TensorDataset(x, y)
    eval_batch_size = 1024 
    train_loader = DataLoader(df_to_tensor(train_df), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(df_to_tensor(val_df), batch_size=eval_batch_size, shuffle=False)
    test_loader = DataLoader(df_to_tensor(test_df), batch_size=eval_batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

def train_experiment(train_df, val_df, test_df, 
                     param_values, param_name='lr', 
                     epochs=10, batch_size=32,
                     step_size=None, gamma=0.1, 
                     fixed_lr=0.01,verbose=False,binary=False):
    """
    Generic training loop. 
    - Can iterate over 'lr' (Section 6.3/6.4) or 'lambda' (Section 6.4 Bonus).
    - Handles Scheduler (Section 6.4) via step_size.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_loaders(train_df, val_df, test_df, batch_size)
    n_classes = len(train_df.iloc[:, 2].unique())
    
    results = {}
    if verbose: print(f"Starting Training: {len(param_values)} configs over {epochs} epochs...")

    for val in param_values:
        # Resolve hyperparameters
        lr = val if param_name == 'lr' else fixed_lr
        wd = val if param_name == 'lambda' else 0.0 # Weight Decay = Lambda

        # Init Model & Optimizer
        model_cls = Logistic_Regression
        model = model_cls(input_dim=2, output_dim=n_classes).to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.CrossEntropyLoss()
        
        # Init Scheduler if step_size is provided (Section 6.4)
        scheduler = None
        if step_size:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        
        # Stats Storage
        stats = {k: [] for k in ['train_loss', 'val_loss', 'test_loss', 
                                 'train_acc', 'val_acc', 'test_acc']}

        # Epoch Loop
        for epoch in range(epochs):
            model.train()
            ep_loss, ep_correct, ep_total = 0, 0, 0
            
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(X)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                
                ep_loss += loss.item() * X.size(0)
                _, pred = torch.max(out, 1)
                ep_correct += (pred == y).sum().item()
                ep_total += X.size(0)
            
            if scheduler: scheduler.step()

            # Record Epoch Metrics
            stats['train_loss'].append(ep_loss / ep_total)
            stats['train_acc'].append(ep_correct / ep_total)
            
            v_loss, v_acc = evaluate(model, val_loader, criterion, device)
            t_loss, t_acc = evaluate(model, test_loader, criterion, device)
            
            stats['val_loss'].append(v_loss); stats['val_acc'].append(v_acc)
            stats['test_loss'].append(t_loss); stats['test_acc'].append(t_acc)

        results[val] = {'model': model, 'stats': stats}
        if verbose: print(f"Finished {param_name}={val} | Final Val Acc: {stats['val_acc'][-1]:.4f}")

    return results


###### ridge regression helpers
def get_accuracy(model,x,y):
    preds = model.predict(x)
    return (preds == y).mean()



class TransposeAdapter:
    """
    A simple wrapper to adapt the Ridge_Regression model (which expects 2xN)
    to the helper function (which sends Nx2).
    """
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        # X comes from the helper as (N_samples, 2)
        # We transpose it to (2, N_samples) for our model
        return self.model.predict(X.T)
