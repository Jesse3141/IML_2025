import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.image as mpimg
import matplotlib.patches as patches
from knn import KNNClassifier

np.random.seed(0)



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


######### KNN ###############################

def get_knn_grid():
    # 1. Define paths
    train_path = 'train.csv'
    test_path = 'test.csv'
    
    # 2. Load Data 
    train_data, train_cols = read_data_demo(train_path)
    test_data, test_cols = read_data_demo(test_path)
    
    # 3. Prepare Train/Test arrays
    X_train = train_data[:, :2]
    Y_train = train_data[:, 2]
    X_test  = test_data[:, :2]
    Y_test  = test_data[:, 2]
    
    # 4. Grid Search Parameters
    knn_size = [1, 10, 100, 1000, 3000]
    metric = ['l1', 'l2']
    
    results = []
    
    # 5. Run Loop
    for k in knn_size:
        for m in metric:
            # Cast k to int to prevent Faiss TypeError
            knn = KNNClassifier(k=int(k), distance_metric=m)
            
            knn.fit(X_train, Y_train)
            preds = knn.predict(X_test)
            
            acc = (preds == Y_test).mean()
            results.append({'k': k, 'metric': m, 'accuracy': acc})
    
    # 6. Display Results
    df = pd.DataFrame(results)
    return df


def knn_examples(X_train, Y_train, X_test, Y_test):
    """
    Notice the similarity to the decision tree demo above.
    This is the sklearn standard format for models.
    """

    # Initialize the KNNClassifier with k=5 and L2 distance metric
    knn_classifier = KNNClassifier(k=5, distance_metric='l2')

    # Train the classifier
    knn_classifier.fit(X_train, Y_train)

    # Predict the labels for the test set
    y_pred = knn_classifier.predict(X_test)

    # Calculate the accuracy of the classifier
    accuracy = np.mean(y_pred == Y_test)


######### trees #############################
def decision_tree_demo():
    # Create random data
    np.random.seed(42)
    X = np.random.rand(100, 2)  # Feature matrix with 100 samples and 2 features
    y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Binary labels based on a simple condition

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Decision Tree classifier
    tree_classifier = DecisionTreeClassifier(random_state=42)

    # Train the Decision Tree on the training data
    tree_classifier.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = tree_classifier.predict(X_test)

    # Compute the accuracy of the predictions
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy}")

def loading_random_forest():
    model = RandomForestClassifier(n_estimators=300, max_depth=6, n_jobs=4)
    return model


def loading_xgboost():
    from xgboost import XGBClassifier
    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=4)
    return model


# 1. Load Data
def get_24_trees():
    # Assuming files are in the local directory as per your 'ls' command
    train_data, _ = read_data_demo('train.csv')
    val_data, _ = read_data_demo('validation.csv')
    test_data, _ = read_data_demo('test.csv')
    
    X_train, Y_train = train_data[:, :2], train_data[:, 2]
    X_val, Y_val = val_data[:, :2], val_data[:, 2]
    X_test, Y_test = test_data[:, :2], test_data[:, 2]
    
    # 2. Define Hyperparameters
    depths = [1, 2, 4, 6, 10, 20, 50, 100]
    leaves = [50, 100, 1000]
    
    results = []
    saved_models = {} # Dictionary to store { (depth, leaf) : model_object }
    
    print("Starting Grid Search...")
    
    for d in depths:
        for l in leaves:
            # Initialize and Train
            dt = DecisionTreeClassifier(max_depth=d, max_leaf_nodes=l)
            dt.fit(X_train, Y_train)
            
            # Calculate Accuracies
            train_acc = dt.score(X_train, Y_train)
            val_acc = dt.score(X_val, Y_val)
            test_acc = dt.score(X_test, Y_test)
            
            # Store results
            results.append({
                'max_depth': d,
                'max_leaf_nodes': l,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'test_acc': test_acc
            })
            
            # Save the actual model for plotting later
            saved_models[(d, l)] = dt
    
    # Create DataFrame for analysis
    df_trees = pd.DataFrame(results)
    return saved_models, df_trees



def get_fitted_xgb():
    # 1. Prepare Labels for XGBoost (XGB requires 0-indexed integers)
    train_data, _ = read_data_demo('train.csv')
    test_data, _ = read_data_demo('test.csv')
    
    X_train, Y_train = train_data[:, :2], train_data[:, 2]
    X_test, Y_test = test_data[:, :2], test_data[:, 2]
    le = LabelEncoder()
    Y_train_enc = le.fit_transform(Y_train)
    Y_test_enc = le.transform(Y_test)
    
    # 2. Initialize and Train
    # params: n_estimators=300, max_depth=6, learning_rate=0.1
    xgb_model = loading_xgboost()
    
    print("Training XGBoost...")
    xgb_model.fit(X_train, Y_train_enc)
    return xgb_model,Y_test_enc,X_test


#### visualistaions #########################



def plot_decision_boundaries(model, X, y, title='Decision Boundaries', save=False,):
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
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x), np.arange(y_min, y_max, h_y))

    # Make predictions on the meshgrid points.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
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
        # Create a safe filename from the title
        filename = title.lower().replace(" ", "_") + ".png"
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
    
    plt.show()

def visualize_leaf_box(model, X, Y, sample_index, feature_names=['Longitude', 'Latitude'], save=False, title = ''):
    """
    Visualizes the decision path (rectangle) for a specific sample point.
    """
    # 1. Get the path for this specific sample
    sample = X[sample_index].reshape(1, -1)
    
    # node_indicator is a sparse matrix indicating which nodes the sample passed through
    node_indicator = model.decision_path(sample)
    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
    
    # 2. Initialize Bounds with the limits of the entire dataset
    # We start with the whole world
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Access internal tree structure
    tree = model.tree_
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold

    print(f"Tracing path for sample #{sample_index} (Class: {Y[sample_index]})...")

    # 3. Traverse the path and tighten the bounds
    for node_id in node_index:
        # If it's a leaf, stop
        if children_left[node_id] == children_right[node_id]:
            continue

        # Get the feature and threshold for the current node
        feat_idx = feature[node_id]
        thresh = threshold[node_id]
        
        # Check if the path goes left or right at this node
        # The next node in our path tells us which way we went
        next_node_in_path = node_index[list(node_index).index(node_id) + 1]

        if next_node_in_path == children_left[node_id]:
            # We went LEFT, meaning Value <= Threshold
            # So the upper bound for this feature becomes the threshold
            if feat_idx == 0: x_max = min(x_max, thresh)
            else:             y_max = min(y_max, thresh)
            print(f"  Node {node_id}: {feature_names[feat_idx]} <= {thresh:.2f}")
            
        else: # next_node == children_right[node_id]
            # We went RIGHT, meaning Value > Threshold
            # So the lower bound for this feature becomes the threshold
            if feat_idx == 0: x_min = max(x_min, thresh)
            else:             y_min = max(y_min, thresh)
            print(f"  Node {node_id}: {feature_names[feat_idx]} > {thresh:.2f}")

    # 4. Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot all data points faintly
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='tab20', alpha=0.2, s=10, label='All Data')
    
    # Plot the specific sample point strictly
    plt.scatter(sample[0, 0], sample[0, 1], c='red', s=100, edgecolors='black', marker='*', zorder=5, label='Sample Point')

    # Draw the Rectangle
    width = x_max - x_min
    height = y_max - y_min
    
    rect = patches.Rectangle((x_min, y_min), width, height, 
                             linewidth=3, edgecolor='black', facecolor='none', 
                             label='Leaf Region')
    
    plt.gca().add_patch(rect)
    
    plt.title(f"Leaf Region for Sample #{sample_index}\nLeaf Bounds: X[{x_min:.2f}, {x_max:.2f}], Y[{y_min:.2f}, {y_max:.2f}]")
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend()

    if save:
        # Create a safe filename from the title
        filename = title.lower().replace(" ", "_") + ".png"
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
        
    plt.show()


def show_comparison(img_path1, title1, img_path2, title2):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Image 1
    img1 = mpimg.imread(img_path1)
    axes[0].imshow(img1)
    axes[0].set_title(title1, fontsize=14)
    axes[0].axis('off') # Turn off axis numbers
    
    # Image 2
    img2 = mpimg.imread(img_path2)
    axes[1].imshow(img2)
    axes[1].set_title(title2, fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()



def plot_anomalies():
    train_path = Path('train.csv')
    ad_test_path = Path('AD_test.csv')
    train_data, train_cols = read_data_demo(train_path)
    ad_test_data, ad_test_cols = read_data_demo(ad_test_path)
    # 1. Load the data
    # Assuming train_data is already loaded from previous steps. 
    # If not, uncomment: train_data, _ = read_data_demo('train.csv')
    X_train = train_data[:, :2] # Use only features (columns 0 and 1)
    # Load AD_test.csv
    X_ad = ad_test_data[:, :2] # Features only
    
    # 2. Initialize and Fit KNN
    # We use k=5 and L2 distance as requested
    knn_ad = KNNClassifier(k=5, distance_metric='l2')
    # We fit on the training data (labels don't matter here, so we pass dummy labels or the original ones)
    knn_ad.fit(X_train, train_data[:, 2]) 
    # 3. Calculate Distances
    # The knn_distance method in your class returns the distances
    distances = knn_ad.knn_distance(X_ad)
    
    # 4. Calculate Anomaly Score (Sum of 5 distances)
    anomaly_scores = np.sum(distances, axis=1)
    
    # 5. Identify Top 50 Anomalies
    
    top_50_indices = np.argsort(anomaly_scores)[-50:]
    
    # Create a mask or boolean array for plotting
    is_anomaly = np.zeros(len(X_ad), dtype=bool)
    is_anomaly[top_50_indices] = True
    
    # 6. Visualization
    plt.figure(figsize=(10, 8))
    
    # Plot Train data (Background density) - indesx 0 is latitude, index 1 is longitude
    plt.scatter(X_train[:, 0], X_train[:, 1], c='black', s=10, alpha=0.01, label='Training Data')
    
    # Plot Normal AD_test points 
    plt.scatter(X_ad[~is_anomaly, 0], X_ad[~is_anomaly, 1], c='blue', s=20, label='Normal (AD Test)')
    
    # Plot Anomaly AD_test points
    plt.scatter(X_ad[is_anomaly, 0], X_ad[is_anomaly, 1], c='red', s=40, label='Anomaly (AD Test)')
    
    plt.title('Anomaly Detection: k=5, L2 Distance')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    decision_tree_demo()
