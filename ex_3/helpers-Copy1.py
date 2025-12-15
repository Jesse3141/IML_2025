import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Sklearn imports for the tree section
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import torch
np.random.seed(42)
torch.manual_seed(42)
# Import your models
from models import *
def plot_decision_boundaries(model, X, y, title='Decision Boundaries', save=False):
    """
    Plots decision boundaries of a classifier and colors the space by the prediction of each point.

    Parameters:
    - model: The trained classifier (sklearn model).
    - X: Numpy Feature matrix.
    - y: Numpy array of Labels.
    - title: Title for the plot.
    """
    # h = .02  # Step size in the mesh

    # enumerate y
    y_map = {v: i for i, v in enumerate(np.unique(y))}
    enum_y = np.array([y_map[v] for v in y]).astype(int)

    h_x = (np.max(X[:, 0]) - np.min(X[:, 0])) / 200
    h_y = (np.max(X[:, 1]) - np.min(X[:, 1])) / 200

    # Plot the decision boundary.
    added_margin_x = h_x * 20
    added_margin_y = h_y * 20
    x_min, x_max = X[:, 0].min() - added_margin_x, X[:, 0].max() + added_margin_x
    y_min, y_max = X[:, 1].min() - added_margin_y, X[:, 1].max() + added_margin_y
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x), np.arange(y_min, y_max, h_y))

    # Make predictions on the meshgrid points.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    print(Z.shape)
    Z = np.array([y_map[v] for v in Z])
    Z = Z.reshape(xx.shape)
    vmin = np.min([np.min(enum_y), np.min(Z)])
    vmax = np.min([np.max(enum_y), np.max(Z)])

    # Plot the decision boundary.
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8, vmin=vmin, vmax=vmax)

    # Scatter plot of the data points with matching colors.
    plt.scatter(X[:, 0], X[:, 1], c=enum_y, cmap=plt.cm.Paired, edgecolors='k', s=40, alpha=0.7, vmin=vmin, vmax=vmax)

    plt.title("Decision Boundaries")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)

    if save:
        # Create a filename from the title (replace spaces with underscores)
        filename = title.replace(" ", "_").replace(".", "").replace("$", "") + ".png"
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
    
    plt.show()


def read_data_demo(filename='train.csv'):
    """
    Read the data from the csv file and return the features and labels as numpy arrays.
    """

    # the data in pandas dataframe format
    df = pd.read_csv(filename)

    # extract the column names
    col_names = list(df.columns)

    # the data in numpy array format
    data_numpy = df.values

    return data_numpy, col_names

###### custom code to implement excercise

##### loading data

def load_data(binary: bool = True, return_splits=False):
    """
    Loads data from csv files.
    
    Args:
        binary (bool): Load binary or multiclass datasets.
        return_splits (bool): If True, returns (X_train, y_train, X_val, y_val, X_test, y_test) as numpy arrays.
                              If False, returns (train_df, val_df, test_df) as DataFrames.
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


    


##### ridge regression


def plot_best_and_worst_lambda(df_results, models_dict, x_test, y_test, save=False):
    """
    Identifies the best and worst lambdas based on validation accuracy
    and plots their decision boundaries using the test set.

    :param df_results: DataFrame containing 'Lambda' and 'Validation Accuracy'
    :param models_dict: Dictionary {lambda: model_object}
    :param x_test: Test features (Shape: 2 x N)
    :param y_test: Test labels
    """
    
    # 1. Identify Best and Worst Lambdas
    best_row = df_results.loc[df_results['Validation Accuracy'].idxmax()]
    worst_row = df_results.loc[df_results['Validation Accuracy'].idxmin()]
    
    best_lambda = best_row['Lambda']
    worst_lambda = worst_row['Lambda']
    
    print(f"Best Lambda: {best_lambda} (Val Acc: {best_row['Validation Accuracy']:.4f})")
    print(f"Worst Lambda: {worst_lambda} (Val Acc: {worst_row['Validation Accuracy']:.4f})")

    # 2. Prepare Data for Helper (Helper expects N x 2, our x_test is 2 x N)
    X_plot = x_test.T 
    
    # 3. Plot Worst Model
    print(f"\nPlotting Worst Model (Lambda={worst_lambda})...")
    worst_model = models_dict[worst_lambda]
    plot_decision_boundaries(
        TransposeAdapter(worst_model), 
        X_plot, 
        y_test, 
        title=rf'Decision Boundary (Worst $\lambda={worst_lambda}$)',
         save=save
    )

    # 4. Plot Best Model
    print(f"Plotting Best Model (Lambda={best_lambda})...")
    best_model = models_dict[best_lambda]
    plot_decision_boundaries(
        TransposeAdapter(best_model), 
        X_plot, 
        y_test, 
        title=rf'Decision Boundary (Best $\lambda={best_lambda}$)',
        save=save
    )
    
def run_ridge_experiments(x_train, y_train, x_val, y_val, x_test, y_test, lambdas, plot = False):
    """
    Runs Ridge Regression for a list of lambdas, returns a DataFrame of results
    and the dictionary of trained models.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    records = []
    trained_models = {}

    for l in lambdas:
        # Train
        model = Ridge_Regression(l)
        model.fit(x_train, y_train)
        
        # Evaluate
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

    # Create DataFrame
    df_results = pd.DataFrame(records)

    # Display Table
    print("Ridge Regression Results:")
    print(df_results)
    
    # Extract best model details
    best_row = df_results.loc[df_results['Validation Accuracy'].idxmax()]
    print(f"\nBest Lambda according to Validation: {best_row['Lambda']}")
    print(f"Test Accuracy of Best Model: {best_row['Test Accuracy']}")
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(df_results['Lambda'], df_results['Train Accuracy'], marker='o', label='Train')
        plt.plot(df_results['Lambda'], df_results['Validation Accuracy'], marker='o', label='Validation')
        plt.plot(df_results['Lambda'], df_results['Test Accuracy'], marker='o', label='Test')
        plt.xlabel('Lambda')
        plt.ylabel('Accuracy')
        plt.title('Ridge Regression Accuracy vs Lambda')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    
    
    return df_results, trained_models


###### gradiant descent in numpy


def run_gradient_descent():
    np.random.seed(42)
    torch.manual_seed(42)
    # 1. Initialization
    x, y = 0.0, 0.0
    lr = 0.1
    iterations = 1000
    
    # Store history for plotting
    history_x = [x]
    history_y = [y]

    # 2. Optimization Loop
    for i in range(iterations):
        # Derivatives of f(x, y) = (x - 3)^2 + (y - 5)^2
        # df/dx = 2*(x - 3)
        # df/dy = 2*(y - 5)
        grad_x = 2 * (x - 3)
        grad_y = 2 * (y - 5)
        
        # Update step
        x = x - lr * grad_x
        y = y - lr * grad_y
        
        history_x.append(x)
        history_y.append(y)

    final_point = (x, y)
    print(f"Final point reached: ({x:.4f}, {y:.4f})")

    # 3. Plotting
    plt.figure(figsize=(8, 6))
    
    # Scatter plot with color mapping to iterations
    sc = plt.scatter(history_x, history_y, c=range(len(history_x)), cmap='viridis', s=10)
    plt.colorbar(sc, label='Iteration')
    
    # Mark the start and end
    plt.scatter(history_x[0], history_y[0], color='red', label='Start (0,0)', marker='x', s=100)
    plt.scatter(history_x[-1], history_y[-1], color='red', label='End', marker='*', s=100)

    plt.title(r'Gradient Descent Optimization of $f(x, y) = (x - 3)^2 + (y - 5)^2$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()



#### pytorch ridge regression

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

    # --- Print Parameters ---
    print(f"\n[{title}] Best Model Parameters:")
    # Detach to numpy for clean printing
    print('Weight:', model.linear.weight.detach().cpu().numpy())
    print('Bias:', model.linear.bias.detach().cpu().numpy())


def run_section_6_3():
    np.random.seed(42)
    torch.manual_seed(42)
    print("\n=== Section 6.3: Binary Logistic Regression ===")
    
    # 1. Load Data
    train, val, test = load_data(binary=True)
    
    # 2. Train Grid
    lrs = [0.1, 0.01, 0.001]
    results = train_experiment(train, val, test, 
                               param_values=lrs, 
                               param_name='lr', 
                               epochs=10, 
                               step_size=None,
                              binary=True,
                              ) # No decay 

    # 3. Select Best Model (Q1)
    best_lr = max(results, key=lambda k: results[k]['stats']['val_acc'][-1])
    best_entry = results[best_lr]
    print(f"\nBest Binary Model LR: {best_lr} (Test Acc: {best_entry['stats']['test_acc'][-1]:.4f})")
    
    # Q1 Visualization
    X_test = test[['long', 'lat']].values
    y_test = test['country'].values
    plot_decision_boundaries(best_entry['model'], X_test, y_test, 
                             title=f'Binary LogReg (LR={best_lr})')

    # 4. Plot Loss Curves (Q2)
    plot_training_results(best_entry, 
                      title=f"Binary Case (LR={best_lr})", 
                      plot_acc=False)




def run_section_6_4(num_epochs: int = 30):
    np.random.seed(42)
    torch.manual_seed(42)
    print("\n=== Section 6.4: Multi-Class Logistic Regression ===")
    
    # 1. Load Data
    train, val, test = load_data(binary=False)
    
    # 2. Train Grid
    lrs = [0.01, 0.001, 0.0003]
    results = train_experiment(train, val, test, 
                               param_values=lrs, 
                               param_name='lr', 
                               epochs=num_epochs, 
                               step_size=5,  # Decay every 5 epochs
                               gamma=0.3,
                              binary=False,
                              verbose=True)    # By factor of 0.3

    # Q1 (required): plot BEST val/test accuracy achieved vs initial LR
    # For each LR, pick the epoch with max val_acc and take test_acc from same epoch.
    val_best = []
    test_at_val_best = []
    best_epoch_per_lr = []
    for lr in lrs:
        stats = results[lr]['stats']
        e_best = int(np.argmax(stats['val_acc']))
        best_epoch_per_lr.append(e_best)
        val_best.append(stats['val_acc'][e_best])
        test_at_val_best.append(stats['test_acc'][e_best])

    plt.figure(figsize=(8, 5))
    plt.plot(lrs, val_best, marker='o', label='Validation (best over epochs)')
    plt.plot(lrs, test_at_val_best, marker='o', label='Test (at best-val epoch)')
    plt.xscale('log')
    plt.xlabel('Initial learning rate')
    plt.ylabel('Accuracy')
    plt.title('Multi-Class Logistic Regression: Accuracy vs Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # Report: test accuracy of the best model according to validation set
    i_best = int(np.argmax(val_best))
    best_lr = lrs[i_best]
    best_epoch = best_epoch_per_lr[i_best]
    best_entry = results[best_lr]
    print(f"\nBest LR by validation: {best_lr} (best epoch: {best_epoch+1})")
    print(f"Validation Acc: {val_best[i_best]:.4f}")
    print(f"Test Acc (same epoch): {test_at_val_best[i_best]:.4f}")



    # Visualization
    X_test = test[['long', 'lat']].values
    y_test = test['country'].values
    plot_decision_boundaries(best_entry['model'], X_test, y_test, 
                             title=f'Multi-Class LogReg (LR={best_lr})')


###### trees


def run_tree_experiments():
    np.random.seed(42)
    print("\n=== Section 6.4 Q3 & Q4: Decision Trees ===")

    # 1. Load Data (Multi-class)
    # Reusing the load_data function from your previous code
    train_df, _, test_df = load_data(binary=False)

    # Convert to Numpy arrays for sklearn
    X_train = train_df[['long', 'lat']].values
    y_train = train_df['country'].values
    X_test = test_df[['long', 'lat']].values
    y_test = test_df['country'].values

    # 2. Run Experiments for Depth 2 and 10
    depths = [2, 10]

    for depth in depths:
        # Initialize and Train
        # random_state=42 ensures reproducibility as requested in Sec 2.1
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)

        # Evaluate
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        print(f"\nDecision Tree (max_depth={depth}):")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy:  {test_acc:.4f}")

        # Visualize
        plot_title = f'Decision Tree (max_depth={depth}) - Test Acc: {test_acc:.2f}'
        plot_decision_boundaries(clf, X_test, y_test, title=plot_title)
