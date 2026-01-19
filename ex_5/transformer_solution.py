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


def display_generated_sentences(history):
    """Q3: Print generated sentences per epoch."""
    print("=" * 50)
    print("Generated Sentences per Epoch (Standard Sampling)")
    print("=" * 50)
    for epoch, sents in history['generated_sentences']:
        print(f"\nEpoch {epoch}:")
        for i, s in enumerate(sents, 1):
            print(f"  {i}. {s}")


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
