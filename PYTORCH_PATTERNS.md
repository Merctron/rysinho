# PyTorch Patterns: A Practical Reference

**Purpose**: Quick reference for common PyTorch patterns you'll see in the code.

---

## 1. Shape Manipulation Patterns

### Broadcasting

PyTorch automatically broadcasts tensors during operations:

```python
# Shape rules: Dimensions are aligned from the right
# Compatible if: dimensions are equal OR one is 1 OR one doesn't exist

x = torch.randn(8, 1, 6, 1)  # Shape: (8, 1, 6, 1)
y = torch.randn(   7, 1, 5)  # Shape: (   7, 1, 5)
z = x + y                     # Shape: (8, 7, 6, 5)

# Common patterns:
batch = torch.randn(32, 10)   # (B, features)
bias = torch.randn(10)        # (features,)
result = batch + bias         # Broadcasting: (32, 10) + (10,) → (32, 10)

# Attention mask broadcasting:
mask = torch.randn(1, 1, seq_len, seq_len)  # (1, 1, S, S)
scores = torch.randn(batch, heads, seq_len, seq_len)  # (B, H, S, S)
masked_scores = scores + mask  # Broadcasts to (B, H, S, S)
```

### Dimension Manipulation Cheat Sheet

```python
x = torch.randn(2, 3, 4)  # Start with (2, 3, 4)

# Add dimension
x.unsqueeze(0)            # (1, 2, 3, 4) - add at position 0
x.unsqueeze(1)            # (2, 1, 3, 4) - add at position 1
x.unsqueeze(-1)           # (2, 3, 4, 1) - add at end
x[None, :, :, :]          # (1, 2, 3, 4) - equivalent to unsqueeze(0)

# Remove dimension
x.squeeze()               # Remove all dimensions of size 1
x.squeeze(1)              # Remove dimension 1 if it's size 1

# Reshape
x.view(2, 12)             # (2, 12) - must have contiguous memory
x.reshape(2, 12)          # (2, 12) - copies if needed
x.view(-1)                # (24,) - flatten, -1 infers size
x.view(2, -1)             # (2, 12) - -1 infers second dimension

# Transpose
x.transpose(0, 1)         # Swap dims 0 and 1: (2, 3, 4) → (3, 2, 4)
x.permute(2, 0, 1)        # Reorder dims: (2, 3, 4) → (4, 2, 3)
x.t()                     # Transpose (2D only)

# Repeat
x.repeat(2, 1, 1)         # (4, 3, 4) - repeat along dim 0
x.expand(5, 2, 3, 4)      # (5, 2, 3, 4) - doesn't copy data (view only)

# Concatenate
torch.cat([x, x], dim=0)  # (4, 3, 4) - concat along dim 0
torch.stack([x, x], dim=0)  # (2, 2, 3, 4) - create new dim

# Split
torch.chunk(x, 2, dim=0)  # List of 2 tensors, each (1, 3, 4)
torch.split(x, [1, 1], dim=0)  # Split with specific sizes
```

### Practical Example: Multi-Head Attention Reshaping

```python
# Start with projections: (B, S, H) where H = num_heads * head_dim
B, S, H = 32, 128, 768
num_heads = 12
head_dim = H // num_heads  # 64

Q = torch.randn(B, S, H)  # Query projection

# Split into multiple heads
# Method 1: reshape + transpose
Q = Q.view(B, S, num_heads, head_dim)  # (32, 128, 12, 64)
Q = Q.transpose(1, 2)                   # (32, 12, 128, 64)
# Now: (batch, heads, seq_len, head_dim)

# Method 2: einops-style (clearer but requires understanding the pattern)
Q = Q.view(B, S, num_heads, head_dim).permute(0, 2, 1, 3)

# After attention, merge heads back
# attn_output shape: (32, 12, 128, 64)
attn_output = attn_output.transpose(1, 2)  # (32, 128, 12, 64)
attn_output = attn_output.contiguous()     # Make contiguous for view
attn_output = attn_output.view(B, S, H)    # (32, 128, 768)
```

---

## 2. Indexing Patterns

### Basic Indexing

```python
x = torch.randn(4, 5, 6)

# Integer indexing
x[0]           # First element along dim 0: shape (5, 6)
x[0, 1]        # Shape (6,)
x[0, 1, 2]     # Scalar (0-d tensor)

# Slice indexing
x[:, 0, :]     # All of dim 0, first of dim 1, all of dim 2: (4, 6)
x[1:3, :, :]   # Rows 1-2: (2, 5, 6)
x[::2, :, :]   # Every other row: (2, 5, 6)

# Negative indexing
x[-1]          # Last element along dim 0
x[:, :, -1]    # Last element along dim 2: (4, 5)

# Ellipsis (...)
x[..., 0]      # Same as x[:, :, 0]: (4, 5)
x[0, ...]      # Same as x[0, :, :]: (5, 6)
```

### Advanced Indexing

```python
# Boolean masking
x = torch.randn(4, 5)
mask = x > 0            # Boolean tensor
positive = x[mask]      # 1D tensor of positive values

# Applying mask along specific dimension
# Example: mask padding tokens
tokens = torch.tensor([[1, 2, 3, 0, 0],
                       [4, 5, 0, 0, 0]])
mask = tokens != 0      # (2, 5) boolean
# To compute mean ignoring padding:
mean = (tokens * mask).sum(dim=1) / mask.sum(dim=1)

# Gather (select specific indices)
x = torch.randn(3, 5)
indices = torch.tensor([[0, 2, 4],
                        [1, 3, 4],
                        [0, 1, 2]])
gathered = torch.gather(x, dim=1, index=indices)  # (3, 3)
# gathered[i, j] = x[i, indices[i, j]]

# Index_select (select along dimension)
indices = torch.tensor([0, 2])
selected = torch.index_select(x, dim=0, index=indices)  # (2, 5)
# Equivalent to: x[[0, 2], :]
```

### Practical Example: Extracting Last Non-Padded Token

```python
# Common pattern: Get last real token (before padding)
# input_ids: (B, S) with padding at end
# attention_mask: (B, S) with 1 for real tokens, 0 for padding

input_ids = torch.tensor([[1, 2, 3, 0, 0],      # Real length: 3
                          [4, 5, 6, 7, 0]])      # Real length: 4
attention_mask = torch.tensor([[1, 1, 1, 0, 0],
                               [1, 1, 1, 1, 0]])

# Find length of each sequence
lengths = attention_mask.sum(dim=1)  # [3, 4]

# Get indices of last real token
last_indices = lengths - 1           # [2, 3]

# Get last tokens
batch_indices = torch.arange(input_ids.size(0))  # [0, 1]
last_tokens = input_ids[batch_indices, last_indices]  # [3, 7]

# For hidden states: (B, S, H)
hidden_states = torch.randn(2, 5, 768)
last_hidden = hidden_states[batch_indices, last_indices, :]  # (2, 768)
```

---

## 3. Matrix Operations

### Matrix Multiplication Variants

```python
# @ operator (preferred)
A @ B                   # Works for 2D, batched, and higher-dim tensors

# Explicit functions
torch.matmul(A, B)      # Same as @
torch.mm(A, B)          # Only for 2D tensors
torch.bmm(A, B)         # Batched matmul (3D tensors)

# Examples:
# 2D matmul
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = A @ B               # (3, 5)

# Batched matmul
A = torch.randn(10, 3, 4)  # 10 matrices of shape (3, 4)
B = torch.randn(10, 4, 5)  # 10 matrices of shape (4, 5)
C = A @ B                  # (10, 3, 5) - batched multiplication

# Broadcasting in matmul
A = torch.randn(10, 3, 4)
B = torch.randn(4, 5)      # No batch dimension
C = A @ B                  # (10, 3, 5) - broadcasts B across batch
```

### Einstein Summation (einsum)

Super powerful notation for complex tensor operations:

```python
# Format: 'input_indices,other_input_indices->output_indices'

# Matrix multiply: C[i,k] = sum_j A[i,j] * B[j,k]
C = torch.einsum('ij,jk->ik', A, B)

# Batched matmul: C[b,i,k] = sum_j A[b,i,j] * B[b,j,k]
C = torch.einsum('bij,bjk->bik', A, B)

# Batched dot product: c[b] = sum_{i} a[b,i] * b[b,i]
c = torch.einsum('bi,bi->b', A, B)

# Attention scores: scores[b,h,i,j] = sum_d Q[b,h,i,d] * K[b,h,j,d]
scores = torch.einsum('bhid,bhjd->bhij', Q, K)

# Trace (sum of diagonal): trace = sum_i A[i,i]
trace = torch.einsum('ii->', A)

# Outer product: C[i,j] = a[i] * b[j]
C = torch.einsum('i,j->ij', a, b)

# Element-wise multiply and sum: scalar = sum_{i,j} A[i,j] * B[i,j]
result = torch.einsum('ij,ij->', A, B)
```

---

## 4. Memory and Performance Patterns

### Contiguous Memory

```python
# Some operations create non-contiguous tensors (views into memory)
x = torch.randn(3, 4)
y = x.transpose(0, 1)  # y is non-contiguous

# Check contiguity
y.is_contiguous()      # False

# Make contiguous (creates copy)
y = y.contiguous()     # Now contiguous

# Why it matters: view() requires contiguous memory
# y.view(-1)           # Would error if y not contiguous
y = y.contiguous().view(-1)  # Safe
```

### In-Place Operations

```python
# Operations ending with _ modify tensor in-place
x = torch.randn(3, 4)

x.add_(1)              # x = x + 1 (in-place)
x.mul_(2)              # x = x * 2 (in-place)
x.zero_()              # Set all elements to 0
x.fill_(5)             # Set all elements to 5

# Gradient methods are often in-place
x.grad.zero_()         # Clear gradients

# Benefits: No memory allocation, faster
# Drawbacks: Can't undo, breaks autograd if x requires grad
```

### Memory-Efficient Patterns

```python
# 1. Use no_grad() for inference
with torch.no_grad():
    output = model(input)  # No computation graph, saves memory

# 2. Use inplace operations when safe
# Instead of: x = x + 1
x.add_(1)

# 3. Delete large tensors
del large_tensor
torch.cuda.empty_cache()  # Release cached GPU memory

# 4. Process in chunks
results = []
for chunk in data.chunk(10):
    with torch.no_grad():
        result = model(chunk)
    results.append(result.cpu())  # Move to CPU to free GPU memory
    del result

# 5. Use gradient checkpointing (for training large models)
from torch.utils.checkpoint import checkpoint
output = checkpoint(model.layer, input)  # Saves memory at cost of recomputation
```

---

## 5. Device Movement Patterns

### Moving Between CPU and GPU

```python
# Check device
x.device               # device(type='cuda', index=0) or device(type='cpu')

# Move to device
x = x.to('cuda')       # Move to default GPU
x = x.to('cuda:1')     # Move to GPU 1
x = x.to('cpu')        # Move to CPU

# Copy to device
y = x.cuda()           # Copy to GPU (deprecated, use .to())
y = x.cpu()            # Copy to CPU

# Create on device directly
x = torch.randn(3, 4, device='cuda')

# Common pattern: flexible device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)
```

### Handling Mixed Device Tensors

```python
# All tensors in operation must be on same device!
x = torch.randn(3, 4, device='cuda')
y = torch.randn(3, 4, device='cpu')

# This will error:
# z = x + y  # RuntimeError!

# Solution 1: Move one to match the other
z = x + y.to(x.device)

# Solution 2: Helper function
def to_device(tensor, device):
    if tensor.device != device:
        return tensor.to(device)
    return tensor

# For models with inputs
def forward(self, input_ids):
    # Ensure input is on same device as model
    input_ids = input_ids.to(self.device)
    ...
```

---

## 6. Debugging Patterns

### Shape Debugging

```python
# Quick shape check
print(f"x shape: {x.shape}")

# Detailed tensor info
print(x)  # Shows shape, dtype, device, and values (truncated if large)

# Check shapes at each step
def debug_forward(x):
    print(f"Input: {x.shape}")
    x = layer1(x)
    print(f"After layer1: {x.shape}")
    x = layer2(x)
    print(f"After layer2: {x.shape}")
    return x

# Assertions for expected shapes
assert x.shape == (batch_size, seq_len, hidden_dim), f"Unexpected shape: {x.shape}"

# Named shapes (for clarity)
batch_size, seq_len, hidden_dim = x.shape
assert hidden_dim == 768, "Hidden dimension mismatch"
```

### NaN and Inf Detection

```python
# Check for NaN
torch.isnan(x).any()   # True if any NaN
torch.isfinite(x).all()  # True if all finite

# Find where NaN occurs
nan_mask = torch.isnan(x)
print(f"NaN at positions: {torch.where(nan_mask)}")

# Register hook to catch NaN during forward pass
def nan_hook(module, input, output):
    if isinstance(output, torch.Tensor):
        if torch.isnan(output).any():
            print(f"NaN detected in {module.__class__.__name__}")
            raise RuntimeError("NaN in forward pass")

for module in model.modules():
    module.register_forward_hook(nan_hook)
```

### Gradient Debugging

```python
# Check if gradients are computed
x.requires_grad        # True if gradients will be computed

# Check gradient values
print(x.grad)          # None if backward() not called yet

# Find layers without gradients
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"No gradient for {name}")

# Check gradient magnitudes (for vanishing/exploding gradients)
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"{name}: grad norm = {grad_norm:.4f}")
```

---

## 7. Common Patterns in Transformers

### Attention Mask Construction

```python
# Causal mask (for autoregressive models)
seq_len = 5
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
# [[False,  True,  True,  True,  True],
#  [False, False,  True,  True,  True],
#  [False, False, False,  True,  True],
#  [False, False, False, False,  True],
#  [False, False, False, False, False]]

# Convert to attention scores format (additive mask)
mask = mask.float() * -1e9  # False→0, True→-1e9
# After adding to scores and softmax, -1e9 → 0 probability

# Padding mask (ignore padding tokens)
input_ids = torch.tensor([[1, 2, 3, 0, 0]])  # 0 = padding
padding_mask = (input_ids == 0)               # True where padded
# Invert for attention mask (1 = attend, 0 = ignore)
attention_mask = (~padding_mask).long()       # [1, 1, 1, 0, 0]
```

### Rotary Positional Encoding (RoPE)

```python
# Conceptual implementation (actual is more optimized)
def apply_rotary_emb(x, cos, sin):
    """
    x: (batch, seq_len, dim) or (batch, heads, seq_len, head_dim)
    cos, sin: precomputed rotation matrices
    """
    # Split x into even and odd indices
    x1 = x[..., ::2]   # Even indices
    x2 = x[..., 1::2]  # Odd indices
    
    # Apply rotation
    # [cos  -sin] [x1]
    # [sin   cos] [x2]
    rotated = torch.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)
    
    # Flatten back
    return rotated.flatten(-2)

# In practice, cos/sin are precomputed and cached
```

### KV Cache Pattern

```python
# Without cache (slow generation)
for step in range(max_length):
    # Process full sequence every time
    logits = model(input_ids)  # input_ids grows each step
    next_token = logits[:, -1, :].argmax(dim=-1)
    input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=1)

# With cache (fast generation)
past_key_values = None
for step in range(max_length):
    # Only process new token
    input_for_step = input_ids if step == 0 else next_token.unsqueeze(-1)
    outputs = model(
        input_for_step,
        past_key_values=past_key_values,
        use_cache=True
    )
    logits = outputs.logits
    past_key_values = outputs.past_key_values  # Reuse in next iteration
    
    next_token = logits[:, -1, :].argmax(dim=-1)
    input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=1)
```

---

## 8. Type Hints and Annotations

### Common Type Hints in PyTorch Code

```python
from typing import Optional, List, Tuple, Union
import torch
from torch import Tensor

def forward(
    self,
    input_ids: Tensor,                          # Required tensor
    attention_mask: Optional[Tensor] = None,    # Optional tensor
    return_dict: bool = True,                   # Boolean flag
) -> Union[Tensor, Tuple[Tensor, ...]]:        # Return type
    """
    Type hints help with:
    - IDE autocomplete
    - Type checking (mypy, pyright)
    - Documentation
    """
    pass

# Common patterns:
# Optional[T]: T or None
# Union[T, U]: T or U
# List[T]: List of T
# Tuple[T, ...]: Tuple of variable length containing T
# Tuple[T, U, V]: Tuple of exactly 3 elements
```

---

## 9. Quick Reference: Common Gotchas

### Issue: "RuntimeError: Expected all tensors to be on the same device"
```python
# Problem
x = torch.randn(3, 4, device='cuda')
y = torch.randn(3, 4, device='cpu')
z = x + y  # Error!

# Solution
z = x + y.to(x.device)
```

### Issue: "RuntimeError: shape mismatch in matmul"
```python
# Problem
A = torch.randn(3, 4)
B = torch.randn(3, 5)
C = A @ B  # Error! Inner dimensions don't match

# Solution: Check shapes
print(f"A: {A.shape}, B: {B.shape}")
# Need A: (3, 4) @ B: (4, 5) for valid matmul
B = torch.randn(4, 5)
C = A @ B  # OK
```

### Issue: "RuntimeError: view size is not compatible with input tensor's size"
```python
# Problem
x = torch.randn(3, 4)
y = x.transpose(0, 1)
z = y.view(-1)  # Error! y is not contiguous

# Solution
z = y.contiguous().view(-1)
# Or use reshape (automatically handles contiguity)
z = y.reshape(-1)
```

### Issue: Gradients not flowing
```python
# Problem
x = torch.randn(3, 4, requires_grad=True)
y = x.detach()  # Detaches from computation graph!
z = y.sum()
z.backward()    # x.grad will be None

# Solution: Don't detach if you need gradients
y = x
z = y.sum()
z.backward()    # x.grad computed correctly
```

### Issue: Gradient accumulation
```python
# Problem
for epoch in range(10):
    loss = model(data)
    loss.backward()
    # Gradients accumulate! Second iteration has 2x gradients

# Solution: Zero gradients
for epoch in range(10):
    optimizer.zero_grad()  # Clear old gradients
    loss = model(data)
    loss.backward()
    optimizer.step()
```

---

## 10. Performance Tips

### DO:
- ✅ Use `torch.no_grad()` for inference
- ✅ Use `.to(device)` once, not repeatedly
- ✅ Batch operations when possible
- ✅ Use `torch.compile()` (PyTorch 2.0+) for faster execution
- ✅ Use appropriate dtype (float16 for inference on GPU)
- ✅ Reuse tensors with in-place operations when safe

### DON'T:
- ❌ Move tensors between devices unnecessarily
- ❌ Use `.item()` in tight loops (synchronizes GPU)
- ❌ Create new tensors in loops (use preallocated tensors)
- ❌ Use Python loops for tensor operations (use vectorized ops)
- ❌ Forget to free GPU memory (use `del` and `empty_cache()`)

---

This reference should help you understand the PyTorch patterns in the layer duplication code. Keep it open while reading through `layer_duplication_annotated.py`!
