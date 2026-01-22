import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from dataset import DataHandler

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))




class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_head, n_embd, block_size):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size

        # Combined Q, K, V projection (more efficient than 3 separate layers)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # Dimension per head
        self.head_dim = n_embd // n_head
        # Causal mask buffer (lower triangular)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer('mask', mask.view(1, 1, block_size, block_size))
        

    def forward(self, x):
        B, T, C = x.size()

        # Q, K, V projections from combined linear layer
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head: (B, T, C) -> (B, n_head, T, head_dim)
        # split C dim into n heads, then transpose head and token - head dim outside matmul!
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention: (Q @ K.T) / sqrt(d_k).
        # note, K.T only on last two dims.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # Apply causal mask (prevent attending to future tokens)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        


        # Softmax and weighted sum with values
        att = torch.nn.functional.softmax(att, dim=-1)
        y = att @ v  # (B, n_head, T, head_dim)

        # Re-assemble heads: (B, n_head, T, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.c_proj(y)
        return y
        

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_head, n_embd, block_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_head, n_embd, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = NewGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPT(nn.Module):
    """ GPT Language Model """


    def __init__(self, n_layer, n_head, n_embd, vocab_size, block_size):
        super().__init__()

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, self.n_embd),
            wpe = nn.Embedding(block_size, self.n_embd),            
            h = nn.ModuleList([Block(n_head, n_embd, block_size) for _ in range(self.n_layer)]),
            ln_f = nn.LayerNorm(self.n_embd),
        ))
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)



    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits


def generate_text(model, data_handler, prompt, max_chars, block_size, device, top_k=None):
    """
    Generate text from prompt using autoregressive sampling.

    Args:
        model: GPT model
        data_handler: DataHandler with encoder/decoder
        prompt: Starting text string
        max_chars: Number of characters to generate
        block_size: Context window size
        device: torch device
        top_k: If None, standard multinomial. If int, restrict to top-k tokens.

    Returns:
        Generated text string (prompt + generated characters)
    """
    was_training = model.training  # Save state
    model.eval()
    current = prompt

    with torch.no_grad():
        for _ in range(max_chars):
            # Use sliding window for context
            context = current[-block_size:] if len(current) >= block_size else current
            tokens = torch.tensor(data_handler.encoder(context), dtype=torch.long, device=device)[None]

            logits = model(tokens)
            logits_last = logits[0, -1, :]  # Get logits for last position

            if top_k is not None:
                # Top-k sampling: restrict to k most probable tokens
                topk_logits, topk_indices = torch.topk(logits_last, top_k)
                probs = torch.softmax(topk_logits, dim=-1)
                idx = torch.multinomial(probs, 1).item()
                next_token = topk_indices[idx].item()
            else:
                # Standard multinomial sampling
                probs = torch.softmax(logits_last, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

            current += data_handler.decoder([next_token])

    model.train(was_training)  # Restore state
    return current


def _evaluate_loader(model, loader, criterion, device, verbose=False):
    """Run model on loader, return (avg_loss, accuracy)."""
    total_loss, correct, total = 0.0, 0, 0
    for batch in tqdm(loader, disable=not verbose):
        x, y = batch
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item() * x.size(0)
        preds = logits[:, -1, :].argmax(dim=-1)
        correct += (preds == y[:, -1]).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def train_model(
        train_path,
        test_path=None,
        model=None,
        block_size=10,
        n_layer=3,
        n_head=3,
        n_embd=48,
        learning_rate=3e-4,
        batch_size=64,
        epochs=10,
        verbose=True,
        return_history=False,
        seed=42
):
    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Initialize history tracking
    history = {
        'train_loss': [], 'test_loss': [],
        'train_acc': [], 'test_acc': [],
        'generated_sentences': [],
        'generated_sentences_tpk': []
    }

    data_handler = DataHandler(train_path, test_path, block_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = data_handler.get_vocab_size()
    if model is None:
        model = GPT(n_layer, n_head, n_embd, vocab_size, block_size)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()
    if verbose:
        print('Using device:', device)


    trainset = data_handler.get_dataset('train')
    testset = data_handler.get_dataset('test')
    
    # setup the dataloader
    train_loader = DataLoader(
        trainset,
        sampler=torch.utils.data.RandomSampler(trainset, replacement=True, num_samples=int(1e5)),
        shuffle=False,
        pin_memory=True,
        batch_size=batch_size,        
    )     
    if testset:       
        test_loader = DataLoader(
            testset,
            sampler=torch.utils.data.RandomSampler(testset, replacement=False, num_samples=int(1e4)),
            shuffle=False,
            pin_memory=True,
            batch_size=batch_size,            
        )

    for ep in range(epochs):
        # Training pass (with gradients)
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch in tqdm(train_loader, disable=not verbose):
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            preds = logits[:, -1, :].argmax(dim=-1)
            train_correct += (preds == y[:, -1]).sum().item()
            train_total += x.size(0)

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Test pass (no gradients)
        model.eval()
        with torch.no_grad():
            avg_test_loss, test_acc = _evaluate_loader(model, test_loader, criterion, device, verbose)

        # Track history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(avg_test_loss)
        history['test_acc'].append(test_acc)

        if verbose:
            print(f'Epoch {ep+1}: Train Loss={avg_train_loss:.4f}, Acc={train_acc:.4f}, '
                  f'Test Loss={avg_test_loss:.4f}, Acc={test_acc:.4f}')

        # Text generation (standard sampling)
        prompt = "the "
        epoch_sentences = []
        epoch_sentences_top_k = []
        for _ in range(3):
            sent = generate_text(model, data_handler, prompt, 30, block_size, device)
            tpk = generate_text(model, data_handler, prompt, 30, block_size, device,5)
            epoch_sentences.append(sent)
            epoch_sentences_top_k.append(tpk)
            if verbose:
                print(f'  Generated: {sent} \n, and with top k {tpk}')
        history['generated_sentences'].append((ep + 1, epoch_sentences))
        history['generated_sentences_tpk'].append((ep + 1, epoch_sentences_top_k))

    if return_history:
        return model, data_handler, history
    return model


if __name__ == "__main__":
    torch.manual_seed(42)
    train_model('train_shakespeare.txt', 'test_shakespeare.txt')
    

