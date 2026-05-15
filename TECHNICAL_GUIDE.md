# Technical Deep Dive: Transformers, PyTorch, and Layer Duplication

**Target Audience**: You have strong CS fundamentals, linear algebra, calculus, and conceptual LLM knowledge, but limited hands-on PyTorch/Transformer implementation experience.

**Goal**: Build mastery of the codebase by understanding the mathematical foundations, PyTorch implementations, and architectural details.

---

## Table of Contents

1. [Transformer Architecture](#1-transformer-architecture)
2. [PyTorch Fundamentals](#2-pytorch-fundamentals)
3. [HuggingFace Model Internals](#3-huggingface-model-internals)
4. [Layer Duplication Theory](#4-layer-duplication-theory)
5. [Implementation Deep Dive](#5-implementation-deep-dive)

---

## 1. Transformer Architecture

### 1.1 High-Level Overview

A transformer language model is fundamentally a **sequence-to-sequence function**:

```
f: ℤ^S → ℝ^(S×V)
```

Where:
- Input: Sequence of S token IDs (integers)
- Output: S × V matrix of logits (V = vocabulary size)
- logits[i, j] = unnormalized log-probability that token j follows position i

The model learns this function through layers of transformations.

### 1.2 Mathematical Flow

For an input sequence of token IDs `x = [x₁, x₂, ..., xₛ]`:

#### Step 1: Embedding Lookup
```
E ∈ ℝ^(V×H)  (embedding matrix, learned)
h₀ = E[x]    (lookup: for each xᵢ, get row E[xᵢ])
h₀ ∈ ℝ^(S×H)
```

Each token gets mapped to an H-dimensional vector (H = hidden dimension, typically 768-4096).

#### Step 2: Positional Information

Transformers have no inherent notion of position. Two approaches:

**Sinusoidal (original Transformer):**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/H))
PE(pos, 2i+1) = cos(pos / 10000^(2i/H))
```

**Learned (GPT-style):**
```
P ∈ ℝ^(max_len×H)  (learned parameter matrix)
h₀ = E[x] + P[0:S]
```

**Rotary (RoPE, Llama/Qwen):**
Applied in attention, not to embeddings. Rotates query/key vectors by position-dependent angles.

#### Step 3: Transformer Layers

Each layer ℓ applies:
```
h_ℓ = TransformerLayer(h_(ℓ-1))
```

A **TransformerLayer** consists of two sub-blocks:

##### Sub-block 1: Multi-Head Self-Attention

Purpose: Mix information across sequence positions.

```
# 1. Project to queries, keys, values
Q = h W_Q,  K = h W_K,  V = h W_V
  where W_Q, W_K, W_V ∈ ℝ^(H×H)

# 2. Split into multiple heads (h = num_heads)
Q = split(Q, h) → (B, h, S, d_k) where d_k = H/h
K = split(K, h) → (B, h, S, d_k)
V = split(V, h) → (B, h, S, d_v) where d_v = H/h

# 3. Scaled dot-product attention (per head)
scores = (Q @ K^T) / √d_k          # (B, h, S, S)
scores = scores + mask             # Causal mask: mask[i,j] = -∞ for j>i
attn_weights = softmax(scores)     # (B, h, S, S)
attn_output = attn_weights @ V     # (B, h, S, d_v)

# 4. Concatenate heads and project
attn_output = concat(heads)        # (B, S, H)
output = attn_output @ W_O         # (B, S, H)
```

**Why multi-head?** Different heads can learn different types of relationships:
- Head 1: Syntactic dependencies (subject-verb)
- Head 2: Semantic similarity
- Head 3: Coreference resolution
- etc.

**Causal Mask**: For autoregressive generation, token i cannot attend to tokens j > i (future tokens). Implemented by adding -∞ to attention scores:

```
mask = [[0,   -∞,  -∞,  -∞],    # Token 0 only sees itself
        [0,    0,  -∞,  -∞],    # Token 1 sees 0 and 1
        [0,    0,   0,  -∞],    # Token 2 sees 0, 1, 2
        [0,    0,   0,   0]]    # Token 3 sees all tokens
```

After softmax, -∞ becomes 0 probability.

##### Sub-block 2: Feed-Forward Network (FFN)

Purpose: Position-wise transformation (each token processed independently).

Modern FFN (SwiGLU, used in Llama):
```
# Expand to 4H dimensions (or more)
gate = σ(h W_gate)    where σ = sigmoid, W_gate ∈ ℝ^(H×4H)
up = h W_up           where W_up ∈ ℝ^(H×4H)
intermediate = gate ⊙ up    # Element-wise product
output = intermediate W_down   where W_down ∈ ℝ^(4H×H)
```

Original FFN (GPT-2):
```
intermediate = GELU(h W₁)    where W₁ ∈ ℝ^(H×4H)
output = intermediate W₂     where W₂ ∈ ℝ^(4H×H)
```

**Why expand?** The bottleneck (H) → expansion (4H) → bottleneck (H) structure allows the model to compute complex nonlinear functions while keeping the residual stream at dimension H.

##### Full Layer with Residuals

Modern transformer layer (pre-norm, used in Llama):

```python
def TransformerLayer(h):
    # Attention block
    h_norm = LayerNorm(h)
    attn_out = MultiHeadAttention(h_norm)
    h = h + attn_out  # Residual connection
    
    # FFN block
    h_norm = LayerNorm(h)
    ffn_out = FeedForward(h_norm)
    h = h + ffn_out   # Residual connection
    
    return h
```

Original Transformer (post-norm):
```python
def TransformerLayer(h):
    # Attention block
    attn_out = MultiHeadAttention(h)
    h = LayerNorm(h + attn_out)  # Residual then norm
    
    # FFN block
    ffn_out = FeedForward(h)
    h = LayerNorm(h + ffn_out)   # Residual then norm
    
    return h
```

**Pre-norm vs Post-norm**: Pre-norm (modern) is more stable for deep networks because gradients flow more directly through residual connections.

#### Step 4: Output Projection

After L layers:
```
h_L = TransformerLayer_L(...TransformerLayer_1(h₀))
h_L = LayerNorm(h_L)           # Final normalization
logits = h_L @ W_output         # W_output ∈ ℝ^(H×V)
```

logits[i, j] = score for token j at position i.

#### Step 5: Generation (Inference)

To generate text:
```python
for position in range(max_length):
    logits = model(input_ids)         # (B, S, V)
    next_token_logits = logits[:, -1, :]  # (B, V) - last position
    
    # Sampling strategies:
    # 1. Greedy (deterministic)
    next_token = argmax(next_token_logits)
    
    # 2. Temperature sampling
    probs = softmax(next_token_logits / temperature)
    next_token = sample(probs)
    
    # 3. Top-k sampling
    top_k_probs, top_k_indices = topk(probs, k)
    next_token = sample(top_k_probs) from top_k_indices
    
    input_ids = concat([input_ids, next_token])
    
    if next_token == EOS:
        break
```

### 1.3 Parameter Count

For a model with:
- V = vocabulary size (e.g., 32,000)
- H = hidden dimension (e.g., 4096)
- L = number of layers (e.g., 32)
- h = number of attention heads (e.g., 32)
- F = FFN intermediate size (typically 4H)

Parameters per layer:
```
Attention:
  Q, K, V projections: 3 × H × H = 3H²
  Output projection:   H × H = H²
  Total attention:     4H²

FFN:
  Up projection:       H × F = 4H²
  Down projection:     F × H = 4H²
  Gate projection:     H × F = 4H² (if SwiGLU)
  Total FFN:           8H² or 12H² (SwiGLU)

Layer norms:
  2 × 2H ≈ 0 (negligible)

Total per layer: ~12H² to 16H²
```

Embedding + output:
```
Embedding matrix:    V × H
Output projection:   H × V (often tied to embedding)
```

**Total parameters**: 
```
P ≈ 2VH + L × 12H²
```

For Llama-7B:
```
V = 32,000
H = 4,096
L = 32

P ≈ 2(32k)(4k) + 32 × 12 × (4k)²
  ≈ 256M + 6,442M
  ≈ 6.7B parameters
```

### 1.4 Computational Complexity

For sequence length S, hidden dimension H:

**Attention**: O(S² H)
- Computing Q @ K^T: S × S matrix with S × H work per element → O(S² H)
- This is the bottleneck for long sequences

**FFN**: O(S H²)
- Linear transformations: S positions × H² per position

**Total per layer**: O(S² H + S H²)

For typical values:
- Short sequences (S < 2048): FFN dominates → O(S H²)
- Long sequences (S > 2048): Attention dominates → O(S² H)

This is why long-context models use techniques like:
- Flash Attention (algorithmic optimization)
- Sparse attention (attention to subset of tokens)
- KV cache (reuse previous computations during generation)

---

## 2. PyTorch Fundamentals

### 2.1 Tensors: The Core Data Structure

A `torch.Tensor` is an n-dimensional array with:
- **Data**: Numerical values
- **Shape**: Dimensions (e.g., (2, 3, 4) = 2×3×4 tensor)
- **Dtype**: Data type (float32, float16, int64, etc.)
- **Device**: CPU or GPU (cuda)
- **Gradient**: Optional, for backpropagation

#### Creating Tensors

```python
import torch

# From Python lists
x = torch.tensor([1, 2, 3])  # Shape: (3,)

# With specific dtype
x = torch.tensor([1.0, 2.0], dtype=torch.float16)

# Initialized tensors
zeros = torch.zeros(2, 3)        # 2×3 matrix of zeros
ones = torch.ones(2, 3)          # 2×3 matrix of ones
random = torch.randn(2, 3)       # Normal(0, 1) random
empty = torch.empty(2, 3)        # Uninitialized (fast)

# Like another tensor (same shape/dtype/device)
y = torch.zeros_like(x)
```

#### Tensor Operations

```python
# Shape queries
x.shape          # torch.Size([2, 3])
x.size()         # Same as .shape
x.dim()          # Number of dimensions: 2
x.numel()        # Total elements: 6

# Indexing (like NumPy)
x[0, 1]          # Element at row 0, col 1
x[0, :]          # First row
x[:, 1]          # Second column

# Reshaping
x = torch.randn(6)
y = x.view(2, 3)         # Reshape to 2×3 (must have compatible size)
z = x.reshape(2, 3)      # Same, but copies if necessary
w = x.unsqueeze(0)       # Add dimension: (6,) → (1, 6)
v = w.squeeze()          # Remove dimensions of size 1: (1, 6) → (6,)

# Transposition
x = torch.randn(2, 3)
y = x.t()                # Transpose: (2, 3) → (3, 2)
z = x.transpose(0, 1)    # Swap dimensions 0 and 1
w = x.permute(1, 0)      # Reorder dimensions
```

#### Matrix Operations

```python
# Element-wise
a + b            # Addition (broadcasting supported)
a * b            # Element-wise multiplication
a / b            # Element-wise division
a ** 2           # Element-wise power

# Matrix multiplication
A @ B            # Matrix multiply (preferred)
torch.matmul(A, B)       # Same
torch.mm(A, B)           # Same, but requires 2D tensors

# For batched operations:
# A: (B, N, M), B: (B, M, P) → (B, N, P)
torch.bmm(A, B)          # Batch matrix multiply

# Einstein summation (powerful notation)
torch.einsum('ijk,ikl->ijl', A, B)  # Batched matmul
torch.einsum('ij,jk->ik', A, B)     # Regular matmul
torch.einsum('ij,ij->i', A, B)      # Batched dot product
```

### 2.2 Autograd: Automatic Differentiation

PyTorch tracks operations to compute gradients automatically.

```python
# Enable gradient tracking
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x
y.backward()     # Compute gradients
print(x.grad)    # dy/dx = 2x + 3 = 4 + 3 = 7

# Gradient accumulation
x.grad.zero_()   # Clear gradients (they accumulate by default!)

# Detaching from graph (no gradients)
z = y.detach()   # z is a tensor, but gradients won't flow through it

# Disable gradients (inference mode)
with torch.no_grad():
    y = x ** 2
    # No computation graph built, saves memory
```

**Why accumulation?** In training, we often accumulate gradients over multiple batches:

```python
optimizer.zero_grad()
for batch in batches:
    loss = model(batch)
    loss.backward()  # Accumulates gradients
optimizer.step()     # Update weights using accumulated gradients
```

### 2.3 Device Management (CPU/GPU)

```python
# Check GPU availability
torch.cuda.is_available()  # True if CUDA GPU available

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0')  # Specific GPU

# Move tensors to device
x = torch.randn(2, 3)
x = x.to(device)                # Move to GPU
x = x.to('cpu')                 # Move back to CPU

# Create tensors directly on device
x = torch.randn(2, 3, device=device)

# Move models to device
model.to(device)                # Moves all parameters to GPU

# CRITICAL: Tensors in operations must be on same device!
# This will error if x is on GPU and y is on CPU:
# z = x + y  # RuntimeError: Expected all tensors to be on the same device
```

**Memory management**:
```python
# Check GPU memory
torch.cuda.memory_allocated()        # Bytes currently allocated
torch.cuda.max_memory_allocated()    # Peak memory

# Clear cache
torch.cuda.empty_cache()             # Release unused cached memory

# Delete tensors to free memory
del x
torch.cuda.empty_cache()
```

### 2.4 Neural Network Modules (nn.Module)

`nn.Module` is the base class for all neural network components.

```python
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()  # MUST call parent constructor
        
        # Define layers
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # Define forward pass
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

# Instantiate and use
model = SimpleModel(10, 20, 5)
x = torch.randn(32, 10)  # Batch of 32 samples
output = model(x)        # Calls forward() automatically
```

**Key methods**:

```python
# Access parameters
for param in model.parameters():
    print(param.shape)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())

# Save/load
torch.save(model.state_dict(), 'model.pt')
model.load_state_dict(torch.load('model.pt'))

# Training/eval modes
model.train()   # Enable dropout, batch norm in training mode
model.eval()    # Disable dropout, batch norm in eval mode
```

### 2.5 Common Layers

```python
# Linear (fully connected)
# y = xW^T + b
linear = nn.Linear(in_features=10, out_features=5)
# Weight shape: (5, 10), bias shape: (5,)

# Embedding (lookup table)
# Input: (B, S) of integers 0 to vocab_size-1
# Output: (B, S, embedding_dim)
embedding = nn.Embedding(num_embeddings=1000, embedding_dim=128)

# Layer normalization
# Normalizes across feature dimension
# y = (x - mean) / std * γ + β
layer_norm = nn.LayerNorm(hidden_dim)

# RMS normalization (used in Llama)
# y = x / rms(x) * γ
# Not in base PyTorch, implemented in HuggingFace models

# Dropout
# Randomly zeros elements with probability p during training
dropout = nn.Dropout(p=0.1)

# Activation functions
relu = nn.ReLU()
gelu = nn.GELU()
silu = nn.SiLU()  # Also called Swish
```

---

## 3. HuggingFace Model Internals

### 3.1 Model Structure

HuggingFace uses a hierarchical structure:

```
AutoModelForCausalLM (wrapper class)
├── config (LlamaConfig)
│   ├── vocab_size
│   ├── hidden_size
│   ├── num_hidden_layers
│   ├── num_attention_heads
│   └── ...
│
├── model (LlamaModel) - the core transformer
│   ├── embed_tokens (Embedding)
│   ├── layers (nn.ModuleList of LlamaDecoderLayer)
│   │   ├── layers[0]
│   │   │   ├── self_attn (LlamaAttention)
│   │   │   │   ├── q_proj (Linear)
│   │   │   │   ├── k_proj (Linear)
│   │   │   │   ├── v_proj (Linear)
│   │   │   │   ├── o_proj (Linear)
│   │   │   │   └── rotary_emb (LlamaRotaryEmbedding)
│   │   │   ├── mlp (LlamaMLP)
│   │   │   │   ├── gate_proj (Linear)
│   │   │   │   ├── up_proj (Linear)
│   │   │   │   ├── down_proj (Linear)
│   │   │   │   └── act_fn (SiLU)
│   │   │   ├── input_layernorm (LlamaRMSNorm)
│   │   │   └── post_attention_layernorm (LlamaRMSNorm)
│   │   └── ... (more layers)
│   └── norm (LlamaRMSNorm) - final layer norm
│
└── lm_head (Linear) - output projection
```

### 3.2 Forward Pass Signature

```python
def forward(
    self,
    input_ids: torch.Tensor,              # (B, S) of token IDs
    attention_mask: Optional[torch.Tensor] = None,  # (B, S) of 0/1
    position_ids: Optional[torch.Tensor] = None,    # (B, S) of positions
    past_key_values: Optional[List[Tuple]] = None,  # KV cache for generation
    use_cache: bool = False,               # Whether to return KV cache
    output_attentions: bool = False,       # Return attention weights
    output_hidden_states: bool = False,    # Return all layer outputs
    return_dict: bool = True,              # Return dataclass vs tuple
) -> Union[Tuple, CausalLMOutputWithPast]:
    ...
```

**Key parameters**:

- `input_ids`: The token IDs to process
- `attention_mask`: 1 = attend, 0 = ignore (for padding)
- `past_key_values`: KV cache from previous generation step
- `use_cache`: Enable KV caching for faster generation
- `output_attentions`: Return attention weights for analysis
- `return_dict`: If True, returns dataclass; if False, returns tuple

### 3.3 KV Cache (For Generation)

During generation, we compute attention for the same prefix many times:

```
Step 1: Process "Hello"
Step 2: Process "Hello world"
Step 3: Process "Hello world!"
```

The KV cache stores key/value projections to avoid recomputation:

```python
# Without cache (slow)
for token in generated_tokens:
    logits = model(all_tokens_so_far)  # Reprocess everything
    next_token = argmax(logits[:, -1, :])
    all_tokens_so_far = concat([all_tokens_so_far, next_token])

# With cache (fast)
past_key_values = None
for token in generated_tokens:
    logits, past_key_values = model(
        last_token_only,  # Only process new token
        past_key_values=past_key_values,  # Reuse cached K, V
        use_cache=True
    )
    next_token = argmax(logits[:, -1, :])
```

Speeds up generation by ~10x for long sequences.

### 3.4 Accessing Internal Components

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("model-name")

# Get embeddings
embeddings = model.model.embed_tokens  # or model.transformer.wte

# Get specific layer
layer_5 = model.model.layers[5]  # or model.transformer.h[5]

# Get attention in layer 5
attention = model.model.layers[5].self_attn

# Get MLP in layer 5
mlp = model.model.layers[5].mlp

# Get final norm
final_norm = model.model.norm  # or model.transformer.ln_f

# Get output projection
lm_head = model.lm_head
```

---

## 4. Layer Duplication Theory

### 4.1 The RYS Hypothesis

**Core Idea**: Transformer layers don't operate independently. Instead, groups of consecutive layers form "functional circuits" that implement complete cognitive operations.

Examples of hypothetical circuits:
- Layers 5-7: Arithmetic reasoning
- Layers 12-15: Factual recall
- Layers 20-24: Language generation

If this is true, **duplicating these circuits should improve performance** on tasks that use that circuit.

### 4.2 Why Duplication Works (Hypothesis)

Think of each circuit as an iterative algorithm. For example, arithmetic might be:

```
Layer 5: Parse number representations
Layer 6: Align operands
Layer 7: Perform operation
```

Running this circuit twice effectively does:
```
Iteration 1: Rough answer
Iteration 2: Refined answer
```

Similar to iterative refinement in numerical methods or EM algorithms.

### 4.3 Mathematical Formulation

Normal forward pass:
```
h₀ = Embed(x)
h₁ = Layer₁(h₀)
h₂ = Layer₂(h₁)
...
h_L = Layer_L(h_(L-1))
```

With duplication of layers [i, j):
```
h₀ = Embed(x)
...
h_(i-1) = Layer_(i-1)(h_(i-2))

# First pass through duplicated layers
h_i = Layer_i(h_(i-1))
...
h_(j-1) = Layer_(j-1)(h_(j-2))

# Second pass through SAME layers
h'_i = Layer_i(h_(j-1))      # Note: input is h_(j-1), not h_(i-1)
...
h'_(j-1) = Layer_(j-1)(h'_(j-2))

# Continue with rest of model
h_j = Layer_j(h'_(j-1))
...
h_L = Layer_L(h_(L-1))
```

The duplicated section processes its output as input (forming a recurrent loop).

### 4.4 Experimental Predictions

If the circuit hypothesis is correct:

1. **Single-layer duplication should not work** - incomplete circuit
2. **Multi-layer blocks should work** - complete circuits
3. **Optimal span should be consistent** - circuits have natural boundaries
4. **Heatmap should show clusters** - distinct functional regions

The RYS paper found exactly these patterns!

---

## 5. Implementation Deep Dive

### 5.1 Challenge: Modifying Forward Pass

HuggingFace models are complex. We can't easily subclass and override `forward()` because:
- Model-specific code (Llama vs GPT-2 vs BLOOM)
- Many internal methods called
- Distributed training complications

**Solution**: Manually execute forward pass with explicit layer control.

### 5.2 Manual Forward Implementation

```python
def manual_forward(model, input_ids, dup_start, dup_end):
    # Step 1: Get model components
    if hasattr(model, 'model'):  # Llama-style
        embed = model.model.embed_tokens
        layers = model.model.layers
        norm = model.model.norm
        lm_head = model.lm_head
    else:  # GPT-2 style
        embed = model.transformer.wte
        layers = model.transformer.h
        norm = model.transformer.ln_f
        lm_head = model.lm_head
    
    # Step 2: Embed tokens
    hidden_states = embed(input_ids)  # (B, S) → (B, S, H)
    
    # Step 3: Pass through layers
    for i, layer in enumerate(layers):
        hidden_states = layer(hidden_states)[0]  # Get just hidden_states from output
        
        # If we just finished duplication range, run it again
        if i == dup_end - 1 and dup_start < dup_end:
            for j in range(dup_start, dup_end):
                hidden_states = layers[j](hidden_states)[0]
    
    # Step 4: Final norm and projection
    hidden_states = norm(hidden_states)
    logits = lm_head(hidden_states)
    
    return logits
```

### 5.3 Evaluation Strategy

The RYS paper used **direct answer evaluation**: prompt for numerical answer, check if correct.

**Why not chain-of-thought?** We want to isolate the model's internal reasoning ability, not its ability to verbalize reasoning.

```python
# Good prompt (direct)
"What is 15 + 27? Answer with just the number."

# Bad prompt (chain-of-thought)
"What is 15 + 27? Let's think step by step."
```

### 5.4 Heatmap Interpretation

The heatmap shows Δaccuracy for each (start, end) configuration:

```
    End Layer →
  ┌─────────────────┐
S │ \               │  Upper triangle invalid (start >= end)
t │   \             │
a │     Green       │  Green = improvement
r │       blocks    │  Red = degradation
t │         \       │  White = neutral
  │           \     │
L │             \   │
a │               \ │
y │                 │
↓ └─────────────────┘
```

**What to look for**:
- **Rectangular green regions**: Coherent circuits (e.g., layers 5-8)
- **Scattered pixels**: Noise (measurement error)
- **Dark red**: Disrupting important computations
- **Best configuration**: Brightest green cell

---

## 6. Running Experiments

### 6.1 Quick Start

```bash
# Install dependencies
pip install torch transformers numpy matplotlib seaborn tqdm

# Run annotated version (has inline explanations)
python layer_duplication_annotated.py
```

### 6.2 Experiment Progression

1. **Baseline evaluation** (5 minutes)
   - Verify model loads and runs
   - Get baseline accuracy

2. **Single configuration test** (10 minutes)
   - Test one duplication config (e.g., middle layers)
   - Verify duplication mechanism works

3. **Coarse scan** (1-2 hours)
   - `step=3`, `max_span=6`
   - Find promising regions

4. **Fine scan** (3-6 hours)
   - `step=1` in promising regions
   - Get high-resolution heatmap

5. **Validation** (30 minutes)
   - Re-test best config on different dataset
   - Verify improvement is real

### 6.3 Debugging Tips

**Model won't load:**
- Check HuggingFace model name
- Ensure enough RAM/VRAM (use smaller model)
- Try `device_map="auto"` for multi-GPU

**OOM (Out of Memory):**
- Reduce batch size
- Use `torch.float16`
- Use 8-bit quantization: `load_in_8bit=True`

**Slow evaluation:**
- Reduce dataset size
- Increase `step` parameter
- Use smaller model for testing

**No improvement from duplication:**
- Model might be too small (< 350M params)
- Evaluation task might be too easy/hard
- Try different layer ranges

---

## 7. Further Reading

**Transformers:**
- "Attention Is All You Need" (Vaswani et al., 2017)
- "The Illustrated Transformer" (Jay Alammar blog)

**LLM Architecture:**
- "Language Models are Few-Shot Learners" (GPT-3 paper)
- "LLaMA: Open and Efficient Foundation Language Models"

**PyTorch:**
- Official PyTorch tutorials: pytorch.org/tutorials
- "Deep Learning with PyTorch" (Stevens et al.)

**Layer Analysis:**
- "Analyzing Hidden Representations in LLMs" (various)
- "Interpretability in the Wild" (Anthropic)

---

## Glossary

- **Autoregressive**: Generating one token at a time, conditioned on previous tokens
- **Causal Mask**: Prevents attention to future tokens
- **Embedding**: Mapping discrete tokens to continuous vectors
- **FFN**: Feed-Forward Network, the MLP in each transformer layer
- **Head**: One attention mechanism in multi-head attention
- **Hidden State**: Intermediate representation (vector per token)
- **Logits**: Unnormalized log-probabilities
- **Pre-norm**: LayerNorm before attention/FFN (modern design)
- **Residual**: Skip connection that adds input to output
- **Token**: Atomic unit of text (word, subword, or character)

---

**You now have the foundation to deeply understand the code!** Start by running `layer_duplication_annotated.py` and reading the comments while the code executes. The comments map directly to the concepts explained here.
