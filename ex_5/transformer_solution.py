"""Transformer solution functions for plotting and comparison."""
import matplotlib.pyplot as plt
from transformer import train_model, generate_text


# ============ Training Wrapper ============
def run_training(train_path='train_shakespeare.txt', test_path='test_shakespeare.txt',
                 epochs=10, verbose=True, **kwargs):
    """Train transformer, return (model, data_handler, history)."""
    return train_model(train_path, test_path, epochs=epochs,
                       verbose=verbose, return_history=True, **kwargs)


# ============ Plotting Functions ============
def plot_loss_curves(history, figsize=(10, 5)):
    """Q2: Plot train/test loss vs epoch."""
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=figsize)
    plt.plot(epochs, history['train_loss'], 'o-', label='Train')
    plt.plot(epochs, history['test_loss'], 's-', label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title('Transformer: Loss vs Epoch')
    plt.show()


def plot_accuracy_curves(history, figsize=(10, 5)):
    """Q2: Plot train/test accuracy vs epoch."""
    epochs = range(1, len(history['train_acc']) + 1)
    plt.figure(figsize=figsize)
    plt.plot(epochs, history['train_acc'], 'o-', label='Train')
    plt.plot(epochs, history['test_acc'], 's-', label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title('Transformer: Accuracy vs Epoch')
    plt.show()


def plot_training_results(history, figsize=(15, 5)):
    """Q2: Plot loss and accuracy curves side by side."""
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'o-', label='Train')
    ax1.plot(epochs, history['test_loss'], 's-', label='Test')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs Epoch')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'o-', label='Train')
    ax2.plot(epochs, history['test_acc'], 's-', label='Test')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Epoch')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def display_generated_sentences(history):
    """Q3: Print generated sentences per epoch (Standard vs Top-k)."""
    print("=" * 80)
    print("Generated Sentences per Epoch: Standard vs Top-k")
    print("=" * 80)
    
    # Zip the standard and top-k lists. Assuming they are aligned by epoch.
    # history['generated_sentences'] is [(1, [s1, s2, s3]), (2, [...]), ...]
    # history['generated_sentences_tpk'] should be similar.
    
    std_hist = history.get('generated_sentences', [])
    tpk_hist = history.get('generated_sentences_tpk', [])
    
    # If tpk is missing (e.g. running old code), just print standard
    if not tpk_hist:
        print("Warning: Top-k history not found. Printing standard only.")
        for epoch, sents in std_hist:
            print(f"\nEpoch {epoch}:")
            for i, s in enumerate(sents, 1):
                print(f"  {i}. {s}")
        return

    for (epoch, std_sents), (_, tpk_sents) in zip(std_hist, tpk_hist):
        print(f"\nEpoch {epoch}:")
        print(f"  {'Standard Sampling':<40} | {'Top-k Sampling':<40}")
        print("-" * 85)
        for s_std, s_tpk in zip(std_sents, tpk_sents):
            # Truncate slightly if too long for side-by-side
            print(f"  {s_std.replace('\n', ' '):<40} | {s_tpk.replace('\n', ' '):<40}")



def compare_generations(standard_sentences, topk_sentences, k=5):
    """Q4: Compare standard vs top-k sampling."""
    print("=" * 50)
    print("Standard vs Top-k Sampling Comparison")
    print("=" * 50)
    print("\nStandard Sampling:")
    for i, s in enumerate(standard_sentences, 1):
        print(f"  {i}. {s}")
    print(f"\nTop-k (k={k}):")
    for i, s in enumerate(topk_sentences, 1):
        print(f"  {i}. {s}")
    print("\nAnalysis: Top-k restricts to most likely tokens -> more coherent, less diverse.")
