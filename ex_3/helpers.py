import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import torch

# Sklearn imports
from sklearn.tree import DecisionTreeClassifier

# Import your models
from models import *

# ==========================================
# General Utilities
# ==========================================

def plot_decision_boundaries(model, X, y, title='Decision Boundaries', save=False):
    """
    Plots decision boundaries of a classifier.
    """
    y_map = {v: i for i, v in enumerate(np.unique(y))}
    enum_y = np.array([y_map[v] for v in y]).astype(int)

    h_x = (np.max(X[:, 0]) - np.min(X[:, 0])) / 200
    h_y = (np.max(X[:, 1]) - np.min(X[:, 1])) / 200

    added_margin_x = h_x * 20
    added_margin_y = h_y * 20
    x_min, x_max = X[:, 0].min() - added_margin_x, X[:, 0].max() + added_margin_x
    y_min, y_max = X[:, 1].min() - added_margin_y, X[:, 1].max() + added_margin_y
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x), np.arange(y_min, y_max, h_y))

    # Make predictions
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array([y_map.get(v, 0) for v in Z]) # .get safe handling
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=enum_y, cmap=plt.cm.Paired, edgecolors='k', s=40, alpha=0.7)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)

    if save:
        filename = title.replace(" ", "_").replace(".", "").replace("$", "") + ".png"
        plt.savefig(filename)
    
    plt.show()

def load_data(binary: bool = True, return_splits=False):
    """
    Loads data from csv files.
    """
    data_path = Path(os.getcwd())
    suffix = '' if binary else '_multiclass'
    
    train_df = pd.read_csv(data_path / f'train{suffix}.csv')
    val_df = pd.read_csv(data_path / f'validation{suffix}.csv')
    test_df = pd.read_csv(data_path / f'test{suffix}.csv')

    if return_splits:
        X_train = train_df[['long', 'lat']].values
        y_train = train_df['country'].values
        X_val = val_df[['long', 'lat']].values
        y_val = val_df['country'].values
        X_test = test_df[['long', 'lat']].values
        y_test = test_df['country'].values
        return X_train, y_train, X_val, y_val, X_test, y_test

    return train_df, val_df, test_df


def plot_training_results(entry, title, plot_acc=True):
    """
    Plots training history and optionally prints model parameters.
    
    Parameters:
    - entry: The dictionary containing {'model': ..., 'stats': ...}
    - title: String title for the plots
    - plot_acc: Boolean, if True plots (1x2) Loss & Accuracy, else just Loss.
    """
    stats = entry['stats']
    model = entry['model']
    epochs = range(1, len(stats['train_loss']) + 1)
    
    # Setup Figure: 1 or 2 subplots
    if plot_acc:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        ax2 = None

    # --- Plot Loss (Always) ---
    ax1.plot(epochs, stats['train_loss'], 'o-', label='Train')
    ax1.plot(epochs, stats['val_loss'], 's-', label='Val')
    ax1.plot(epochs, stats['test_loss'], '^-', label='Test')
    ax1.set_title(f'{title} - Loss Curves')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Plot Accuracy (Optional) ---
    if plot_acc and ax2 is not None:
        ax2.plot(epochs, stats['train_acc'], 'o-', label='Train')
        ax2.plot(epochs, stats['val_acc'], 's-', label='Val')
        ax2.plot(epochs, stats['test_acc'], '^-', label='Test')
        ax2.set_title(f'{title} - Accuracy Curves')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()





# ==========================================
# Section 3: Ridge Regression Helpers
# ==========================================

class TransposeAdapter:
    def __init__(self, model):
        self.model = model
    def predict(self, X):
        return self.model.predict(X.T)

def run_ridge_grid_search(lambdas):
    """
    Runs the training for Ridge Regression and returns results + data.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    # Load Data
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(binary=True, return_splits=True)
    x_train, x_val, x_test = x_train.T,  x_val.T, x_test.T
    records = []
    trained_models = {}

    for l in lambdas:
        model = Ridge_Regression(l)
        model.fit(x_train, y_train)
        
        t_acc = get_accuracy(model, x_train, y_train)
        v_acc = get_accuracy(model, x_val, y_val)
        test_acc = get_accuracy(model, x_test, y_test)
        
        records.append({
            'Lambda': l,
            'Train Accuracy': t_acc,
            'Validation Accuracy': v_acc,
            'Test Accuracy': test_acc
        })
        trained_models[l] = model

    df_results = pd.DataFrame(records)
    return df_results, trained_models, (x_test, y_test)

def ridge_q1_plot_accuracies(df_results):
    """Plots Train/Val/Test Accuracy vs Lambda"""
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['Lambda'], df_results['Train Accuracy'], marker='o', label='Train')
    plt.plot(df_results['Lambda'], df_results['Validation Accuracy'], marker='o', label='Validation')
    plt.plot(df_results['Lambda'], df_results['Test Accuracy'], marker='o', label='Test')
    plt.xlabel('Lambda')
    plt.ylabel('Accuracy')
    plt.title('Ridge Regression Accuracy vs Lambda')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    best_row = df_results.loc[df_results['Validation Accuracy'].idxmax()]
    print(f"Best Model (Validation) Lambda: {best_row['Lambda']}")
    print(f"Test Accuracy of Best Model: {best_row['Test Accuracy']:.4f}")

def ridge_q2_plot_boundaries(df_results, models_dict, x_test, y_test):
    """Plots Best and Worst Lambda Decision Boundaries"""
    best_lambda = df_results.loc[df_results['Validation Accuracy'].idxmax()]['Lambda']
    worst_lambda = df_results.loc[df_results['Validation Accuracy'].idxmin()]['Lambda']

    # X needs to be (N, 2) for the plotter, which expects sklearn style
    # But our Ridge model expects (2, N). The TransposeAdapter handles this.
    X_plot = x_test.T 

    print(f"Plotting Worst Model (Lambda={worst_lambda})...")
    plot_decision_boundaries(TransposeAdapter(models_dict[worst_lambda]), X_plot, y_test, 
                             title=rf'Ridge Boundary (Worst $\lambda={worst_lambda}$)')

    print(f"Plotting Best Model (Lambda={best_lambda})...")
    plot_decision_boundaries(TransposeAdapter(models_dict[best_lambda]), X_plot, y_test, 
                             title=rf'Ridge Boundary (Best $\lambda={best_lambda}$)')


# ==========================================
# Section 4: NumPy Gradient Descent
# ==========================================

def run_numpy_gd_experiment():
    np.random.seed(42)
    x, y = 0.0, 0.0
    lr = 0.1
    iterations = 1000
    history_x = [x]
    history_y = [y]

    for i in range(iterations):
        grad_x = 2 * (x - 3)
        grad_y = 2 * (y - 5)
        x = x - lr * grad_x
        y = y - lr * grad_y
        history_x.append(x)
        history_y.append(y)

    print(f"Final point reached: ({x:.4f}, {y:.4f})")
    
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(history_x, history_y, c=range(len(history_x)), cmap='viridis', s=15)
    plt.colorbar(sc, label='Iteration')
    plt.scatter(history_x[0], history_y[0], c='red', marker='x', s=100, label='Start')
    plt.scatter(history_x[-1], history_y[-1], c='red', marker='*', s=100, label='End')
    plt.title(r'GD Optimization of $f(x, y) = (x - 3)^2 + (y - 5)^2$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ==========================================
# Section 6.3: Binary Logistic Regression
# ==========================================

def train_binary_logistic_models():
    """
    Trains the binary logistic regression models for grid search.
    Returns: results dictionary, test_data (df)
    """
    np.random.seed(42)
    torch.manual_seed(42)
    train, val, test = load_data(binary=True)
    lrs = [0.1, 0.01, 0.001]
    
    print(f"Training Binary Models with LRs: {lrs}...")
    results = train_experiment(train, val, test, 
                               param_values=lrs, 
                               param_name='lr', 
                               epochs=10, 
                               step_size=None,
                               binary=True,
                               verbose=False)
    return results, test

def section_6_3_q1(results, test_df):
    """
    Q1: Choose best val accuracy model and visualize test predictions.
    """
    # Find best LR based on final validation accuracy
    best_lr = max(results, key=lambda k: results[k]['stats']['val_acc'][-1])
    best_entry = results[best_lr]
    
    print(f"Best Binary Model LR: {best_lr}")
    print(f"Validation Acc: {best_entry['stats']['val_acc'][-1]:.4f}")
    print(f"Test Acc: {best_entry['stats']['test_acc'][-1]:.4f}")
    
    X_test = test_df[['long', 'lat']].values
    y_test = test_df['country'].values
    
    plot_decision_boundaries(best_entry['model'], X_test, y_test, 
                             title=f'Binary LogReg Boundary (Best LR={best_lr})')
    
    return best_lr

def section_6_3_q2(results, best_lr):
    """
    Q2: Plot Training, Val, Test LOSS curves for the best model.
    """
    best_entry = results[best_lr]
    plot_training_results(best_entry, 
                          title=f"Binary Model (LR={best_lr})", 
                          plot_acc=False) # Only Loss is requested in Q2 text, but usually acc is good too

# ==========================================
# Section 6.4: Multi-Class Logistic Regression
# ==========================================

def train_multiclass_logistic_models():
    """
    Trains multi-class models.
    Returns: results dict, test_data (df)
    """
    np.random.seed(42)
    torch.manual_seed(42)
    train, val, test = load_data(binary=False)
    lrs = [0.01, 0.001, 0.0003]
    
    print(f"Training Multi-class Models with LRs: {lrs}...")
    results = train_experiment(train, val, test, 
                               param_values=lrs, 
                               param_name='lr', 
                               epochs=30, 
                               step_size=5,  
                               gamma=0.3,
                               binary=False,
                               verbose=False)
    return results, test

def section_6_4_q1(results):
    """
    Q1: Plot Test & Val Accuracy vs Initial LR.
    Report test accuracy of the best model (by val).
    """
    lrs = sorted(results.keys())
    val_peaks = []
    test_at_peak = []
    
    # Extract metrics
    for lr in lrs:
        stats = results[lr]['stats']
        # "Best" means best validation accuracy epoch
        best_epoch_idx = int(np.argmax(stats['val_acc']))
        val_peaks.append(stats['val_acc'][best_epoch_idx])
        test_at_peak.append(stats['test_acc'][best_epoch_idx])

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(lrs, val_peaks, 'o-', label='Max Validation Acc')
    plt.plot(lrs, test_at_peak, 's-', label='Test Acc (at Max Val)')
    plt.xscale('log')
    plt.xlabel('Initial Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('Multi-Class: Accuracy vs Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # Report
    best_idx = np.argmax(val_peaks)
    best_lr = lrs[best_idx]
    print(f"Best LR (by Validation): {best_lr}")
    print(f"Validation Acc: {val_peaks[best_idx]:.4f}")
    print(f"Test Acc: {test_at_peak[best_idx]:.4f}")
    
    return best_lr

def section_6_4_q2(results, best_lr):
    """
    Q2: Choose best model. Plot Train/Val/Test LOSSES and ACCURACIES.
    """
    best_entry = results[best_lr]
    
    # We use plot_acc=True to get both Loss (Left) and Accuracy (Right) plots
    # as requested: "In addition, plot the training, validation & test accuracies"
    plot_training_results(best_entry, 
                          title=f"Multi-Class Model (LR={best_lr})", 
                          plot_acc=True)

def section_6_4_q3_tree_d2():
    """
    Q3: Decision Tree with Depth 2.
    """
    np.random.seed(42)
    train_df, _, test_df = load_data(binary=False)
    X_train = train_df[['long', 'lat']].values
    y_train = train_df['country'].values
    X_test = test_df[['long', 'lat']].values
    y_test = test_df['country'].values

    clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    clf.fit(X_train, y_train)
    
    acc = clf.score(X_test, y_test)
    print(f"Decision Tree (Depth=2) Test Accuracy: {acc:.4f}")
    
    plot_decision_boundaries(clf, X_test, y_test, title=f'Decision Tree (Depth=2) - Acc: {acc:.2f}')

def section_6_4_q4_tree_d10():
    """
    Q4: Decision Tree with Depth 10.
    """
    np.random.seed(42)
    train_df, _, test_df = load_data(binary=False)
    X_train = train_df[['long', 'lat']].values
    y_train = train_df['country'].values
    X_test = test_df[['long', 'lat']].values
    y_test = test_df['country'].values

    clf = DecisionTreeClassifier(max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    
    acc = clf.score(X_test, y_test)
    print(f"Decision Tree (Depth=10) Test Accuracy: {acc:.4f}")
    
    plot_decision_boundaries(clf, X_test, y_test, title=f'Decision Tree (Depth=10) - Acc: {acc:.2f}')

def section_6_4_q5_bonus_ridge():
    """
    Q5 Bonus: Logistic Regression with Ridge (Weight Decay).
    """
    train, val, test = load_data(binary=False)
    # Using LRs from Section 3.1 logic (0, 2, 4...) is for Ridge *Regression*.
    # For Logistic Ridge, we usually use small lambdas. 
    # The PDF says "Use the same lambda values as in Sec 3.1".
    # Note: In Pytorch, 'weight_decay' is roughly 2*lambda or lambda depending on definition.
    # We will pass these directly as weight_decay.
    
    lambdas = [0., 2., 4., 6., 8., 10.]
    
    # Fixed LR as per PDF
    fixed_lr = 0.01 
    
    print(f"Training Bonus Ridge Logistic with Lambdas: {lambdas}...")
    results = train_experiment(train, val, test, 
                               param_values=lambdas, 
                               param_name='lambda', # This switches logic to iterate WD
                               epochs=30, 
                               fixed_lr=fixed_lr,
                               step_size=5, 
                               gamma=0.3,
                               binary=False)
    
    # Find best lambda
    best_lam = max(results, key=lambda k: results[k]['stats']['val_acc'][-1])
    best_entry = results[best_lam]
    
    print(f"Best Ridge Lambda: {best_lam}")
    print(f"Test Acc: {best_entry['stats']['test_acc'][-1]:.4f}")
    
    # Visualize
    X_test = test[['long', 'lat']].values
    y_test = test['country'].values
    plot_decision_boundaries(best_entry['model'], X_test, y_test, 
                             title=f'Logistic Ridge (Lambda={best_lam})')