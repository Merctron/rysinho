"""
Layer Duplication Experiments - TECHNICAL DEEP DIVE WITH ANNOTATIONS

This file is heavily commented to teach PyTorch, Transformers, and the RYS approach.
Read this like a technical paper with executable code.

═══════════════════════════════════════════════════════════════════════════════
CONCEPTUAL FOUNDATION: What are we actually doing?
═══════════════════════════════════════════════════════════════════════════════

A transformer language model is a stack of identical layers:
    Input Tokens → Embeddings → Layer₀ → Layer₁ → ... → Layer_N → LM Head → Logits

Each transformer layer performs:
    1. Multi-head self-attention (contextual mixing of token representations)
    2. Feed-forward network (position-wise transformation)
    3. Residual connections + layer normalization (training stability)

The RYS hypothesis: Layers don't work independently. Groups of consecutive layers
form "functional circuits" that perform complete cognitive operations (like
"arithmetic reasoning" or "factual recall"). These circuits span multiple layers.

Layer duplication tests this by repeating layer ranges:
    Normal:      Layer₀ → Layer₁ → Layer₂ → ... → Layer_N
    Duplicated:  Layer₀ → [Layer₁ → Layer₂] → [Layer₁ → Layer₂] → ... → Layer_N
                                  ↑______________|

If repeating layers 1-2 improves performance on math tasks, it suggests layers 1-2
form a reasoning circuit that benefits from deeper processing.

═══════════════════════════════════════════════════════════════════════════════
PYTORCH FOUNDATIONS
═══════════════════════════════════════════════════════════════════════════════

Key PyTorch concepts you'll see:
1. torch.Tensor: N-dimensional arrays with GPU support and autograd
2. torch.nn.Module: Base class for all neural network components
3. torch.no_grad(): Context manager that disables gradient computation (for inference)
4. .to(device): Moves tensors/models between CPU and GPU
5. torch.dtype: Precision (float16=half precision, float32=full precision)

Shape notation used below:
    B = batch size (number of sequences processed in parallel)
    S = sequence length (number of tokens)
    H = hidden dimension (size of token embeddings, e.g., 768 or 4096)
    V = vocabulary size (number of possible tokens, e.g., 32000)
    L = number of layers in the model
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm  # Progress bars
import json
from dataclasses import dataclass
import re


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Understanding HuggingFace Model Structure
# ═══════════════════════════════════════════════════════════════════════════════

"""
HuggingFace models follow a standard structure (using Llama/Qwen as example):

MyModel (AutoModelForCausalLM)
├── model (LlamaModel)                    ← The core transformer
│   ├── embed_tokens (Embedding)          ← Token ID → embedding vector
│   │   Input:  (B, S) of token IDs
│   │   Output: (B, S, H) embeddings
│   │
│   ├── layers (ModuleList)               ← The transformer layers!
│   │   ├── layers[0] (LlamaDecoderLayer)
│   │   │   ├── self_attn (LlamaAttention)
│   │   │   │   ├── q_proj, k_proj, v_proj, o_proj (Linear layers)
│   │   │   │   └── rotary_emb (RoPE positional encoding)
│   │   │   ├── mlp (LlamaMLP)
│   │   │   │   ├── gate_proj, up_proj, down_proj
│   │   │   │   └── act_fn (SiLU activation)
│   │   │   ├── input_layernorm (RMSNorm)
│   │   │   └── post_attention_layernorm (RMSNorm)
│   │   ├── layers[1] (LlamaDecoderLayer) ← Same structure
│   │   ├── layers[2] (LlamaDecoderLayer)
│   │   └── ... (L layers total)
│   │
│   └── norm (RMSNorm)                    ← Final layer normalization
│
└── lm_head (Linear)                      ← Projects to vocabulary
    Input:  (B, S, H)
    Output: (B, S, V) logits for each token

For GPT-2 style models, the structure is slightly different:
    model.transformer.wte (embeddings)
    model.transformer.h (layers)
    model.transformer.ln_f (final norm)
    model.lm_head (output projection)
"""


@dataclass
class EvalResult:
    """
    Simple data container for evaluation results.

    Python's dataclass decorator auto-generates __init__, __repr__, etc.
    This is just a typed struct/record.
    """
    config: Tuple[int, int]  # (start_layer, end_layer) for duplication
    score: float              # Accuracy (0.0 to 1.0)
    correct: int             # Number of correct predictions
    total: int               # Total number of problems


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Transformer Layer Deep Dive
# ═══════════════════════════════════════════════════════════════════════════════

"""
What happens inside a single transformer decoder layer?

Input: hidden_states with shape (B, S, H)

1. SELF-ATTENTION
   ───────────────
   Purpose: Allow each token to "look at" and aggregate information from other tokens

   a) Project to Q, K, V:
      Q = hidden @ W_q  # Query:  "what am I looking for?"
      K = hidden @ W_k  # Key:    "what information do I have?"
      V = hidden @ W_v  # Value:  "what information should I pass?"

   b) Compute attention scores:
      scores = (Q @ K^T) / sqrt(d_k)  # Scaled dot-product
      # Shape: (B, num_heads, S, S) - each token's attention to every other token

   c) Apply causal mask (for autoregressive LM):
      mask[i, j] = -inf if j > i  # Token i can't attend to future tokens j
      scores = scores + mask

   d) Softmax and weighted sum:
      attention_weights = softmax(scores)  # Normalize to probabilities
      output = attention_weights @ V       # Weighted sum of values

   e) Multi-head: Do this in parallel with multiple sets of W_q, W_k, W_v
      Then concatenate heads and project: output = concat(heads) @ W_o

2. FEED-FORWARD NETWORK (MLP)
   ───────────────────────────
   Purpose: Position-wise transformation (same weights applied to each token independently)

   Two-layer network with expansion:
      intermediate = activation(hidden @ W_1)  # Expand to ~4*H dimensions
      output = intermediate @ W_2               # Project back to H dimensions

   Modern models use SwiGLU activation:
      gate = sigmoid(hidden @ W_gate)
      up = hidden @ W_up
      output = (gate * up) @ W_down

3. RESIDUAL CONNECTIONS + LAYER NORM
   ───────────────────────────────────
   Purpose: Training stability and gradient flow

   Layer structure with residuals:
      # Attention block
      normed = layer_norm(hidden_states)
      attn_out = self_attention(normed)
      hidden_states = hidden_states + attn_out  # Residual

      # FFN block
      normed = layer_norm(hidden_states)
      ffn_out = feed_forward(normed)
      hidden_states = hidden_states + ffn_out  # Residual

   The residual connections create a "highway" for gradients during backprop.

In code, this looks like:
    layer_output = layer(hidden_states)
    # layer_output might be a tuple (hidden_states, attention_weights, ...)
    # We only care about the new hidden_states
"""


class LayerDuplicator:
    """
    Wrapper class to modify a model's forward pass to duplicate layers.

    This is a "monkey-patching" approach - we intercept and modify the model's
    behavior without changing its weights or source code.

    ═══════════════════════════════════════════════════════════════════════════════
    DESIGN PATTERN: Why not subclass the model?
    ═══════════════════════════════════════════════════════════════════════════════

    HuggingFace models are complex with model-specific code. Rather than subclass
    every model type, we use composition: wrap the model and modify behavior
    externally. This is more flexible and works across model families.
    """

    def __init__(self, model, layer_attr='model.layers'):
        """
        Args:
            model: A HuggingFace PreTrainedModel (specifically AutoModelForCausalLM)
            layer_attr: Dot-separated path to access the layer list
                       - 'model.layers' for Llama/Qwen/Mistral
                       - 'transformer.h' for GPT-2/GPT-Neo
        """
        self.model = model
        self.layer_attr = layer_attr
        self.duplication_config = None  # None = no duplication, else (start, end)

    def get_layers(self):
        """
        Navigate nested attributes to access the ModuleList of transformer layers.

        PyTorch models are nested: model.model.layers
        This method walks the attribute path: 'model.layers' → getattr(getattr(model, 'model'), 'layers')

        Returns:
            nn.ModuleList of transformer layers
        """
        obj = self.model
        for attr in self.layer_attr.split('.'):
            obj = getattr(obj, attr)
        return obj

    def set_duplication(self, start_layer: int, end_layer: int):
        """
        Configure which layers to duplicate.

        Args:
            start_layer: First layer to repeat (0-indexed)
            end_layer: One past last layer to repeat (Python slice convention)

        Example: set_duplication(5, 8) means repeat layers [5, 6, 7]

        The model will process:
            layers[0:5] → layers[5:8] → layers[5:8] → layers[8:N]
                          \___ repeat ___/
        """
        self.duplication_config = (start_layer, end_layer)

    def clear_duplication(self):
        """Reset to normal forward pass (no duplication)"""
        self.duplication_config = None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Manual Forward Pass Implementation
# ═══════════════════════════════════════════════════════════════════════════════

def manual_forward_with_duplication(
    model,
    input_ids: torch.Tensor,
    start_layer: int,
    end_layer: int,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Manually execute a forward pass with layer duplication.

    This function breaks down what normally happens in model(input_ids) into
    explicit steps so we can insert the layer duplication logic.

    ═══════════════════════════════════════════════════════════════════════════════
    UNDERSTANDING THE FORWARD PASS
    ═══════════════════════════════════════════════════════════════════════════════

    Normal forward pass:
        1. Token IDs → Embeddings
        2. Pass through each layer sequentially
        3. Final normalization
        4. Project to vocabulary (lm_head)

    With duplication at layers [start, end):
        1. Token IDs → Embeddings
        2. Pass through layers 0 to start-1
        3. Pass through layers start to end-1  ← First time
        4. Pass through layers start to end-1  ← DUPLICATE (second time)
        5. Pass through layers end to N-1
        6. Final normalization + lm_head

    Args:
        model: HuggingFace causal LM model
        input_ids: Token IDs with shape (B, S)
                   B = batch size, S = sequence length
        start_layer: First layer to duplicate (inclusive)
        end_layer: Last layer to duplicate (exclusive)
        attention_mask: Optional mask with shape (B, S)
                       1 = attend to this token, 0 = ignore (padding)

    Returns:
        logits: Tensor of shape (B, S, V) where V = vocabulary size
                logits[b, s, v] = unnormalized log probability of token v
                                  at position s in batch element b

    ═══════════════════════════════════════════════════════════════════════════════
    PYTORCH SHAPES AND OPERATIONS
    ═══════════════════════════════════════════════════════════════════════════════

    Key tensor operations:
    - tensor.shape or tensor.size(): Get dimensions (returns torch.Size)
    - tensor[indices]: Indexing works like numpy
    - tensor.unsqueeze(dim): Add dimension of size 1 at position dim
    - tensor.expand(sizes): Repeat tensor along dimensions (view, no copy)
    - tensor.to(device): Move to GPU/CPU
    - torch.ones_like(tensor): Create tensor of 1s with same shape/dtype/device
    """

    # ───────────────────────────────────────────────────────────────────────────
    # STEP 1: Access model components (handle different architectures)
    # ───────────────────────────────────────────────────────────────────────────

    # Different model families have different internal structures
    # We need to detect which one we're using

    if hasattr(model, 'model'):
        # Llama-style: model.model.{embed_tokens, layers, norm}, model.lm_head
        embed = model.model.embed_tokens
        layers = model.model.layers
        norm = model.model.norm
        lm_head = model.lm_head
    elif hasattr(model, 'transformer'):
        # GPT-2 style: model.transformer.{wte, h, ln_f}, model.lm_head
        embed = model.transformer.wte
        layers = model.transformer.h
        norm = model.transformer.ln_f
        lm_head = model.lm_head
    else:
        raise ValueError("Unsupported model architecture - can't find layers")

    # ───────────────────────────────────────────────────────────────────────────
    # STEP 2: Convert token IDs to embeddings
    # ───────────────────────────────────────────────────────────────────────────

    # embed is an nn.Embedding layer, essentially a lookup table
    # embed.weight has shape (V, H) where V = vocab size, H = hidden dim
    # embed(input_ids) looks up embeddings: input_ids[i] → embed.weight[input_ids[i]]

    hidden_states = embed(input_ids)  # (B, S) → (B, S, H)

    # hidden_states[b, s, :] is now a dense H-dimensional vector representing
    # token s in batch element b. Initially, this is just a learned lookup,
    # but as we pass through layers, it accumulates contextual information.

    # ───────────────────────────────────────────────────────────────────────────
    # STEP 3: Prepare attention mask
    # ───────────────────────────────────────────────────────────────────────────

    # Attention mask serves two purposes:
    # 1. Padding mask: Ignore padding tokens (mask = 0)
    # 2. Causal mask: Prevent attending to future tokens (for autoregressive LM)

    if attention_mask is None:
        # If no mask provided, assume all tokens are valid (no padding)
        attention_mask = torch.ones_like(input_ids)  # (B, S) of all 1s

    # Most HuggingFace models handle causal masking internally in their attention
    # layers, so we just need to pass the padding mask here.

    # ───────────────────────────────────────────────────────────────────────────
    # STEP 4: Pass through transformer layers with duplication
    # ───────────────────────────────────────────────────────────────────────────

    num_layers = len(layers)

    # Each layer is an nn.Module that implements forward()
    # layer(hidden_states) returns either:
    #   - just new hidden_states (simple case), or
    #   - tuple (hidden_states, attention_weights, ...) (if output_attentions=True)

    for layer_idx in range(num_layers):
        # ═══════════════════════════════════════════════════════════════════════
        # UNDERSTANDING LAYER FORWARD CALLS
        # ═══════════════════════════════════════════════════════════════════════
        #
        # The transformer layer forward signature typically looks like:
        #   forward(hidden_states, attention_mask=None, **kwargs) -> output
        #
        # Where output can be:
        #   - A tuple: (hidden_states, attention_weights, ...)
        #   - Just hidden_states (if return_dict=False and minimal outputs)
        #
        # We call with attention_mask to handle padding correctly.

        layer_output = layers[layer_idx](
            hidden_states,
            attention_mask=attention_mask,
        )

        # Extract hidden_states from output (handle both tuple and tensor returns)
        if isinstance(layer_output, tuple):
            # If tuple, first element is always the new hidden_states
            hidden_states = layer_output[0]
        else:
            # If tensor, it's directly the hidden_states
            hidden_states = layer_output

        # hidden_states still has shape (B, S, H) but now incorporates information
        # from this layer's attention and feed-forward computations

        # ═══════════════════════════════════════════════════════════════════════
        # LAYER DUPLICATION LOGIC
        # ═══════════════════════════════════════════════════════════════════════

        # Check if we just finished the duplication range
        if layer_idx == end_layer - 1 and start_layer < end_layer:
            # We've completed layers [start, end), now run them again

            for dup_idx in range(start_layer, end_layer):
                # Run the same layer again on the current hidden_states
                # This is the key: we're feeding the output of layer end-1
                # back into layer start, creating a "loop" in computation

                layer_output = layers[dup_idx](
                    hidden_states,
                    attention_mask=attention_mask,
                )

                if isinstance(layer_output, tuple):
                    hidden_states = layer_output[0]
                else:
                    hidden_states = layer_output

            # After this loop, hidden_states has been processed by layers
            # [start, end) TWICE, effectively deepening that section

    # ───────────────────────────────────────────────────────────────────────────
    # STEP 5: Final normalization
    # ───────────────────────────────────────────────────────────────────────────

    # Final layer norm stabilizes the outputs before projection
    # RMSNorm (used in Llama): normalized = hidden / rms(hidden) * scale
    # LayerNorm (used in GPT-2): normalized = (hidden - mean) / std * scale + bias

    hidden_states = norm(hidden_states)  # (B, S, H) → (B, S, H)

    # ───────────────────────────────────────────────────────────────────────────
    # STEP 6: Project to vocabulary (compute logits)
    # ───────────────────────────────────────────────────────────────────────────

    # lm_head is a Linear layer: nn.Linear(H, V)
    # Projects each H-dimensional hidden state to V-dimensional vocabulary space

    logits = lm_head(hidden_states)  # (B, S, H) @ (H, V) → (B, S, V)

    # logits[b, s, v] represents the unnormalized log probability that
    # the next token after position s-1 is token v
    #
    # To get probabilities: probs = softmax(logits, dim=-1)
    # To get next token: next_token = argmax(logits[:, -1, :], dim=-1)

    return logits


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Evaluation Framework
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleEvaluator:
    """
    Evaluates model performance on reasoning tasks.

    The RYS paper used hard benchmarks (MATH, MMLU). We start simpler with
    arithmetic, but the pattern is the same:
    1. Create problems with known answers
    2. Generate model predictions
    3. Compare predictions to ground truth

    ═══════════════════════════════════════════════════════════════════════════════
    WHY SIMPLE TASKS?
    ═══════════════════════════════════════════════════════════════════════════════

    Small language models (< 1B params) struggle with complex math. Simple
    arithmetic lets us:
    - Get signal even from small models
    - Run many evaluations quickly
    - Isolate the effect of layer duplication

    For larger models (> 7B), use harder benchmarks from lm-evaluation-harness.
    """

    def __init__(self, model, tokenizer, device='cuda'):
        """
        Args:
            model: HuggingFace PreTrainedModel
            tokenizer: HuggingFace PreTrainedTokenizer
            device: 'cuda' or 'cpu' (where to run inference)

        ═══════════════════════════════════════════════════════════════════════
        PYTORCH DEVICE MANAGEMENT
        ═══════════════════════════════════════════════════════════════════════

        Tensors and models must be on the same device (CPU or GPU).

        - model.to(device): Moves all model parameters to device
        - tensor.to(device): Moves tensor data to device
        - device = 'cuda' uses the default GPU
        - device = 'cuda:0', 'cuda:1', etc. for specific GPUs
        - device = 'cpu' for CPU execution

        Always ensure: input tensors' device == model's device
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)  # Move model to GPU/CPU
        self.model.eval()  # Set to evaluation mode (disables dropout, etc.)

    def create_simple_math_dataset(self, num_samples=50) -> List[Dict]:
        """
        Generate simple arithmetic problems.

        Returns:
            List of dicts with keys: 'question', 'answer', 'type'

        ═══════════════════════════════════════════════════════════════════════
        EVALUATION DESIGN PHILOSOPHY
        ═══════════════════════════════════════════════════════════════════════

        Good evaluation datasets should be:
        1. Deterministic: Same problems every run (for reproducibility)
        2. Balanced: Multiple problem types
        3. Difficulty-appropriate: Not too easy (ceiling effect) or too hard
        4. Parseable: Easy to extract and verify answers

        We use "Answer with just the number" prompting to make parsing easier.
        """
        problems = []

        # Seed for reproducibility (same problems across runs)
        np.random.seed(42)

        # ═══════════════════════════════════════════════════════════════════════
        # PROBLEM TYPE 1: Addition (moderate difficulty)
        # ═══════════════════════════════════════════════════════════════════════
        for _ in range(num_samples // 5):
            a, b = np.random.randint(10, 100), np.random.randint(10, 100)
            problems.append({
                'question': f"What is {a} + {b}? Answer with just the number.",
                'answer': str(a + b),
                'type': 'addition'
            })

        # ═══════════════════════════════════════════════════════════════════════
        # PROBLEM TYPE 2: Subtraction
        # ═══════════════════════════════════════════════════════════════════════
        for _ in range(num_samples // 5):
            a, b = np.random.randint(50, 100), np.random.randint(10, 49)
            problems.append({
                'question': f"What is {a} - {b}? Answer with just the number.",
                'answer': str(a - b),
                'type': 'subtraction'
            })

        # ═══════════════════════════════════════════════════════════════════════
        # PROBLEM TYPE 3: Multiplication (harder for small LMs)
        # ═══════════════════════════════════════════════════════════════════════
        for _ in range(num_samples // 5):
            a, b = np.random.randint(2, 20), np.random.randint(2, 20)
            problems.append({
                'question': f"What is {a} * {b}? Answer with just the number.",
                'answer': str(a * b),
                'type': 'multiplication'
            })

        # ═══════════════════════════════════════════════════════════════════════
        # PROBLEM TYPE 4: Division (requires integer division)
        # ═══════════════════════════════════════════════════════════════════════
        for _ in range(num_samples // 5):
            b = np.random.randint(2, 12)
            a = b * np.random.randint(2, 15)  # Ensure clean division
            problems.append({
                'question': f"What is {a} / {b}? Answer with just the number.",
                'answer': str(a // b),
                'type': 'division'
            })

        # ═══════════════════════════════════════════════════════════════════════
        # PROBLEM TYPE 5: Word problems (tests reading comprehension + arithmetic)
        # ═══════════════════════════════════════════════════════════════════════
        for _ in range(num_samples // 5):
            apples = np.random.randint(5, 20)
            given = np.random.randint(2, apples)
            problems.append({
                'question': f"John has {apples} apples. He gives {given} to Mary. How many apples does John have left? Answer with just the number.",
                'answer': str(apples - given),
                'type': 'word_problem'
            })

        return problems

    def extract_number(self, text: str) -> Optional[str]:
        """
        Extract the first number from generated text.

        LLM outputs can be messy: "The answer is 42." or "42\n" or "I think it's 42"
        We use regex to find the number.

        Args:
            text: Generated text from model

        Returns:
            First number found as string, or None if no number found

        ═══════════════════════════════════════════════════════════════════════
        PARSING MODEL OUTPUTS
        ═══════════════════════════════════════════════════════════════════════

        LLMs are generative - they produce text, not structured outputs.
        We need to parse their responses to extract answers.

        Regex pattern r'-?\d+' matches:
        - -? : optional negative sign
        - \d+ : one or more digits

        findall returns all matches; we take the first one.

        More robust evaluation would use:
        - Multiple extraction attempts (different regex patterns)
        - Fuzzy matching
        - Explicit answer formatting in prompt ("Answer: [NUMBER]")
        """
        numbers = re.findall(r'-?\d+', text)
        return numbers[0] if numbers else None

    def evaluate_problem(self, question: str, answer: str, max_new_tokens=10) -> bool:
        """
        Evaluate a single problem.

        Args:
            question: Question text (the prompt)
            answer: Ground truth answer (as string)
            max_new_tokens: How many tokens to generate

        Returns:
            True if model's answer matches ground truth, False otherwise

        ═══════════════════════════════════════════════════════════════════════
        TEXT GENERATION PROCESS
        ═══════════════════════════════════════════════════════════════════════

        1. Tokenization: text → token IDs
        2. Encoding: model maps tokens → hidden states → logits
        3. Sampling: logits → next token (via argmax, sampling, beam search, etc.)
        4. Decoding: token IDs → text
        5. Repeat steps 2-4 until stopping condition (max length, EOS token, etc.)
        """

        # ───────────────────────────────────────────────────────────────────────
        # STEP 1: Tokenize the question
        # ───────────────────────────────────────────────────────────────────────

        # Tokenizer converts text to token IDs
        # "Hello world" → [15496, 1917] (example IDs)

        inputs = self.tokenizer(
            question,
            return_tensors="pt"  # Return PyTorch tensors (not lists)
        ).to(self.device)

        # inputs is a dict with keys:
        # - 'input_ids': (1, S) tensor of token IDs
        # - 'attention_mask': (1, S) tensor of 1s (no padding in single sentence)

        # ───────────────────────────────────────────────────────────────────────
        # STEP 2: Generate answer
        # ───────────────────────────────────────────────────────────────────────

        # torch.no_grad() disables autograd (gradient computation)
        # This saves memory and speeds up inference
        # We don't need gradients since we're not training

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,  # Unpack input_ids and attention_mask
                max_new_tokens=max_new_tokens,  # Generate at most 10 new tokens
                do_sample=False,  # Greedy decoding (always pick argmax)
                pad_token_id=self.tokenizer.eos_token_id  # Padding token ID
            )

        # outputs shape: (1, S + new_tokens)
        # It includes BOTH the input prompt and the generated continuation

        # ═══════════════════════════════════════════════════════════════════════
        # UNDERSTANDING model.generate()
        # ═══════════════════════════════════════════════════════════════════════
        #
        # generate() is a high-level API that handles the generation loop:
        #
        # while len(output) < max_length:
        #     logits = model(output)              # Forward pass
        #     next_token_logits = logits[:, -1, :]  # Get last position logits
        #
        #     if do_sample:
        #         next_token = sample(next_token_logits)  # Multinomial sampling
        #     else:
        #         next_token = argmax(next_token_logits)  # Greedy (deterministic)
        #
        #     output = cat([output, next_token])   # Append to sequence
        #
        #     if next_token == eos_token_id:
        #         break
        #
        # Parameters:
        # - do_sample=False: Greedy decoding (deterministic, good for evaluation)
        # - do_sample=True + temperature: Stochastic sampling (creative generation)
        # - top_k, top_p: Truncated sampling (limit to top K or top P% probability)
        # - num_beams: Beam search (explore multiple candidate sequences)

        # ───────────────────────────────────────────────────────────────────────
        # STEP 3: Decode only the generated part
        # ───────────────────────────────────────────────────────────────────────

        # We want to extract just the model's answer, not the question
        # So we slice off the prompt: outputs[0][len(input):]

        input_length = inputs['input_ids'].shape[1]
        generated_ids = outputs[0][input_length:]  # Get only new tokens

        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True  # Remove <eos>, <pad>, etc.
        )

        # ───────────────────────────────────────────────────────────────────────
        # STEP 4: Extract and compare answer
        # ───────────────────────────────────────────────────────────────────────

        predicted = self.extract_number(generated_text)

        # String comparison (not numeric) to handle edge cases
        # "42" == "42" ✓
        # "42.0" == "42" ✗ (would need normalization)

        return predicted == answer if predicted else False

    def evaluate_dataset(self, dataset: List[Dict]) -> Dict:
        """
        Evaluate model on full dataset.

        Returns:
            Dict with overall accuracy and per-type breakdown
        """
        correct = 0
        results_by_type = {}

        # tqdm wraps iterator to show progress bar
        for problem in tqdm(dataset, desc="Evaluating"):
            is_correct = self.evaluate_problem(problem['question'], problem['answer'])
            correct += int(is_correct)  # True → 1, False → 0

            # Accumulate per-type statistics
            ptype = problem['type']
            if ptype not in results_by_type:
                results_by_type[ptype] = {'correct': 0, 'total': 0}
            results_by_type[ptype]['correct'] += int(is_correct)
            results_by_type[ptype]['total'] += 1

        return {
            'accuracy': correct / len(dataset),
            'correct': correct,
            'total': len(dataset),
            'by_type': results_by_type
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Layer Scanning and Visualization
# ═══════════════════════════════════════════════════════════════════════════════

class LayerDuplicationScanner:
    """
    Systematically tests all layer duplication configurations.

    For a model with N layers, there are O(N²) possible configurations:
    - (0, 1): duplicate just layer 0
    - (0, 2): duplicate layers 0-1
    - ...
    - (0, N): duplicate all layers
    - (1, 2): duplicate just layer 1
    - etc.

    We evaluate each configuration and create a heatmap showing which
    layer ranges improve performance.

    ═══════════════════════════════════════════════════════════════════════════════
    COMPUTATIONAL COST ANALYSIS
    ═══════════════════════════════════════════════════════════════════════════════

    Let:
    - N = number of layers
    - E = evaluation dataset size
    - T = time per forward pass

    Full scan complexity: O(N² × E × T)

    For N=24, E=50, T=0.1s → ~14,400 seconds ≈ 4 hours

    Optimization strategies:
    1. Use step > 1 (test every 2nd or 3rd layer)
    2. Limit max_span (only test spans up to K layers)
    3. Use smaller eval set for quick iterations
    4. Parallelize across multiple GPUs (batching)
    """

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.evaluator = SimpleEvaluator(model, tokenizer, device)

        self.num_layers = self._get_num_layers()
        print(f"Model has {self.num_layers} layers")

    def _get_num_layers(self) -> int:
        """
        Count transformer layers.

        Different model architectures store layers in different attributes.
        We check common patterns.
        """
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return len(self.model.model.layers)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return len(self.model.transformer.h)
        else:
            raise ValueError("Cannot determine number of layers")

    def scan_configurations(
        self,
        eval_dataset: List[Dict],
        min_span: int = 1,
        max_span: int = None,
        step: int = 1
    ) -> Tuple[List[EvalResult], EvalResult]:
        """
        Scan all layer duplication configurations.

        Args:
            eval_dataset: Problems to evaluate on
            min_span: Minimum number of layers to duplicate (e.g., 2 = at least 2 layers)
            max_span: Maximum span (None = no limit)
            step: Step size for layer indices (2 = test every other layer)

        Returns:
            Tuple of (all_results, baseline_result)

        ═══════════════════════════════════════════════════════════════════════
        EXPERIMENTAL DESIGN
        ═══════════════════════════════════════════════════════════════════════

        Always evaluate baseline first (no duplication) to establish reference.
        Then test configurations and measure DELTA from baseline.

        Key insight from RYS paper: Absolute performance matters less than
        relative improvement. A +5% boost is significant even if baseline is 40%.
        """

        if max_span is None:
            max_span = self.num_layers

        results = []

        # ───────────────────────────────────────────────────────────────────────
        # BASELINE EVALUATION
        # ───────────────────────────────────────────────────────────────────────

        print("Evaluating baseline (no duplication)...")
        baseline_perf = self.evaluator.evaluate_dataset(eval_dataset)
        baseline_result = EvalResult(
            config=(-1, -1),  # Sentinel value for "no duplication"
            score=baseline_perf['accuracy'],
            correct=baseline_perf['correct'],
            total=baseline_perf['total']
        )
        results.append(baseline_result)
        print(f"Baseline accuracy: {baseline_result.score:.2%}")

        # ───────────────────────────────────────────────────────────────────────
        # GENERATE CONFIGURATIONS TO TEST
        # ───────────────────────────────────────────────────────────────────────

        configs_to_test = []
        for start in range(0, self.num_layers, step):
            for end in range(start + min_span, min(start + max_span + 1, self.num_layers + 1), step):
                # Configuration (start, end) means duplicate layers [start, end)
                # Constraints:
                # - end > start (non-empty range)
                # - end - start >= min_span (minimum span size)
                # - end - start <= max_span (maximum span size)
                configs_to_test.append((start, end))

        print(f"\nScanning {len(configs_to_test)} configurations...")
        print(f"This will take approximately {len(configs_to_test) * len(eval_dataset) * 0.1 / 60:.1f} minutes")

        # ───────────────────────────────────────────────────────────────────────
        # TEST EACH CONFIGURATION
        # ───────────────────────────────────────────────────────────────────────

        # NOTE: This is where you'd actually implement the layer duplication
        # The current code evaluates without duplication (TODO for full implementation)

        for start, end in tqdm(configs_to_test, desc="Testing configurations"):
            # TODO: Actually implement duplication here
            # This would involve modifying the model's forward pass to duplicate
            # layers [start, end) before running evaluation

            # For now, we just run normal evaluation (no duplication)
            # In full implementation, you'd do:
            # 1. Patch model's forward method to duplicate layers [start, end)
            # 2. Evaluate
            # 3. Restore original forward method

            perf = self.evaluator.evaluate_dataset(eval_dataset)

            results.append(EvalResult(
                config=(start, end),
                score=perf['accuracy'],
                correct=perf['correct'],
                total=perf['total']
            ))

        return results, baseline_result

    def create_heatmap(
        self,
        results: List[EvalResult],
        baseline_score: float,
        output_path: str = 'layer_duplication_heatmap.png'
    ):
        """
        Create heatmap visualization of results.

        Args:
            results: Evaluation results for all configurations
            baseline_score: Baseline accuracy (no duplication)
            output_path: Where to save the figure

        ═══════════════════════════════════════════════════════════════════════
        VISUALIZATION RATIONALE
        ═══════════════════════════════════════════════════════════════════════

        A heatmap is ideal for this data because:
        1. Two dimensions: start_layer × end_layer
        2. Continuous metric: accuracy delta
        3. Pattern discovery: Visual clusters reveal functional circuits

        The heatmap shows:
        - Rows = start layer
        - Columns = end layer
        - Color = improvement over baseline
        - Upper triangle masked (invalid: start > end)

        Expected patterns:
        - Diagonal blocks: Coherent multi-layer circuits
        - Scattered pixels: Single-layer effects (likely noise)
        - Vertical/horizontal bands: Particular layers are critical
        """

        # ───────────────────────────────────────────────────────────────────────
        # STEP 1: Build matrix from results
        # ───────────────────────────────────────────────────────────────────────

        # Initialize with NaN (missing data) - will appear blank in heatmap
        matrix = np.full((self.num_layers, self.num_layers), np.nan)

        for result in results:
            start, end = result.config
            if start >= 0:  # Skip baseline (marked as -1, -1)
                # Convert absolute accuracy to delta (percentage points)
                # e.g., 0.65 → 0.70 is +5 percentage points
                delta = (result.score - baseline_score) * 100

                # Matrix[i, j] represents duplicating layers [i, j+1)
                # We use end-1 for column since end is exclusive
                matrix[start, end - 1] = delta

        # ───────────────────────────────────────────────────────────────────────
        # STEP 2: Create figure and mask
        # ───────────────────────────────────────────────────────────────────────

        plt.figure(figsize=(12, 10))

        # Mask upper triangle (invalid configurations where start >= end)
        # np.triu creates upper triangular matrix
        # k=0 means include diagonal, k=1 would exclude diagonal
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=0)

        # ───────────────────────────────────────────────────────────────────────
        # STEP 3: Plot heatmap
        # ───────────────────────────────────────────────────────────────────────

        # seaborn.heatmap is a high-level function built on matplotlib
        sns.heatmap(
            matrix,
            mask=mask,  # Don't show upper triangle
            annot=False,  # Don't write values in cells (too crowded)
            fmt='.1f',  # Format for annotations if annot=True
            cmap='RdYlGn',  # Red-Yellow-Green colormap (red=bad, green=good)
            center=0,  # Center colormap at 0 (neutral is yellow)
            xticklabels=range(self.num_layers),  # Label columns 0, 1, 2, ...
            yticklabels=range(self.num_layers),  # Label rows 0, 1, 2, ...
            cbar_kws={'label': 'Accuracy Delta (%)'}  # Colorbar label
        )

        plt.xlabel('End Layer (exclusive)')
        plt.ylabel('Start Layer')
        plt.title(f'Layer Duplication Performance Heatmap\nBaseline Accuracy: {baseline_score:.2%}')
        plt.tight_layout()

        # ───────────────────────────────────────────────────────────────────────
        # STEP 4: Save and report best configuration
        # ───────────────────────────────────────────────────────────────────────

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nHeatmap saved to {output_path}")

        # Find best configuration (ignore NaN values)
        best_idx = np.nanargmax(matrix)  # Linear index of max value
        best_start, best_end = np.unravel_index(best_idx, matrix.shape)  # Convert to (row, col)
        best_score = matrix[best_start, best_end]

        print(f"\nBest configuration: layers [{best_start}, {best_end + 1})")
        print(f"Improvement: +{best_score:.2f} percentage points")
        print(f"This suggests layers {best_start} through {best_end} form a functional circuit")

        return matrix


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Main Experimental Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Main experiment workflow.

    This demonstrates the complete pipeline:
    1. Load model and tokenizer
    2. Create evaluation dataset
    3. Run baseline evaluation
    4. (Optional) Run full layer duplication scan
    5. Generate visualizations

    ═══════════════════════════════════════════════════════════════════════════════
    BEST PRACTICES FOR RESEARCH CODE
    ═══════════════════════════════════════════════════════════════════════════════

    1. Reproducibility: Set random seeds, log configuration
    2. Modularity: Separate data creation, evaluation, analysis
    3. Incremental: Run quick tests before long experiments
    4. Persistence: Save intermediate results (recover from crashes)
    5. Documentation: Log what you tried and why
    """

    # ───────────────────────────────────────────────────────────────────────────
    # CONFIGURATION
    # ───────────────────────────────────────────────────────────────────────────

    MODEL_NAME = "HuggingFaceTB/SmolLM-360M"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_EVAL_SAMPLES = 50

    print("="*80)
    print("LAYER DUPLICATION EXPERIMENT")
    print("="*80)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Evaluation samples: {NUM_EVAL_SAMPLES}")
    print("="*80)

    # ───────────────────────────────────────────────────────────────────────────
    # LOAD MODEL AND TOKENIZER
    # ───────────────────────────────────────────────────────────────────────────

    # HuggingFace AutoModel classes automatically detect model type
    # and instantiate the correct architecture

    print(f"\nLoading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load model with appropriate precision
    # - float16 (half precision): 2x faster, 2x less memory, slight accuracy loss
    # - float32 (full precision): default, most accurate
    # - int8/int4 (quantization): 4-8x less memory, requires bitsandbytes

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None  # Auto-distribute across GPUs
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # PYTORCH DTYPE AND MEMORY
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # dtype controls numerical precision and memory usage:
    #
    # torch.float32 (fp32): 32 bits per number
    # - Range: ±3.4 × 10³⁸
    # - Precision: ~7 decimal digits
    # - Memory: 1x (baseline)
    #
    # torch.float16 (fp16): 16 bits per number
    # - Range: ±65,504
    # - Precision: ~3 decimal digits
    # - Memory: 0.5x (half of fp32)
    # - Speed: ~2x faster on modern GPUs (Tensor Cores)
    #
    # torch.bfloat16 (bf16): 16 bits, different format
    # - Range: same as fp32 (±3.4 × 10³⁸)
    # - Precision: ~2 decimal digits
    # - Better for training (wider range, less overflow)
    #
    # torch.int8: 8 bits per number (quantization)
    # - Range: -128 to 127
    # - Memory: 0.25x
    # - Requires quantization-aware training or post-training quantization
    #
    # For inference on GPUs: Use float16
    # For inference on CPUs: Use float32 (CPU doesn't accelerate float16)

    # Handle tokenizer padding token (some models don't have one defined)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded successfully")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"Memory: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9:.2f} GB")

    # ───────────────────────────────────────────────────────────────────────────
    # CREATE EVALUATION DATASET
    # ───────────────────────────────────────────────────────────────────────────

    evaluator = SimpleEvaluator(model, tokenizer, DEVICE)

    print(f"\nCreating evaluation dataset ({NUM_EVAL_SAMPLES} problems)...")
    eval_dataset = evaluator.create_simple_math_dataset(NUM_EVAL_SAMPLES)

    print(f"Sample problems:")
    for i, problem in enumerate(eval_dataset[:3]):
        print(f"  {i+1}. {problem['question']} → {problem['answer']}")

    # ───────────────────────────────────────────────────────────────────────────
    # BASELINE EVALUATION
    # ───────────────────────────────────────────────────────────────────────────

    print("\n" + "="*80)
    print("BASELINE EVALUATION (No Duplication)")
    print("="*80)

    baseline_results = evaluator.evaluate_dataset(eval_dataset)

    print(f"\nOverall accuracy: {baseline_results['accuracy']:.2%}")
    print(f"Correct: {baseline_results['correct']}/{baseline_results['total']}")

    print(f"\nBreakdown by problem type:")
    for ptype, stats in baseline_results['by_type'].items():
        acc = stats['correct'] / stats['total']
        print(f"  {ptype:15s}: {acc:.2%} ({stats['correct']}/{stats['total']})")

    # ───────────────────────────────────────────────────────────────────────────
    # OPTIONAL: FULL LAYER DUPLICATION SCAN
    # ───────────────────────────────────────────────────────────────────────────

    print("\n" + "="*80)
    print("LAYER DUPLICATION SCAN")
    print("="*80)
    print("\nFull scanning is commented out by default (takes hours).")
    print("To run it, uncomment the code block below.")
    print("\nRecommended first experiment:")
    print("  - min_span=2 (test 2+ layer spans)")
    print("  - max_span=8 (limit to 8 layers)")
    print("  - step=2 (test every other layer)")
    print("  This reduces runtime by ~4x while capturing main patterns.")

    # Uncomment to run full scan:
    """
    scanner = LayerDuplicationScanner(model, tokenizer, DEVICE)

    results, baseline = scanner.scan_configurations(
        eval_dataset,
        min_span=2,  # At least 2 layers (RYS paper shows single layers don't work)
        max_span=8,  # At most 8 layers
        step=2       # Test every other layer (faster)
    )

    # Create heatmap
    matrix = scanner.create_heatmap(results, baseline.score)

    # Save raw results
    results_data = [
        {
            'config': r.config,
            'score': r.score,
            'delta': (r.score - baseline.score) * 100,
            'correct': r.correct,
            'total': r.total
        }
        for r in results
    ]

    with open('layer_duplication_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)

    print("\nResults saved to layer_duplication_results.json")

    # Find top 5 configurations
    results_sorted = sorted(results, key=lambda r: r.score, reverse=True)
    print("\nTop 5 configurations:")
    for i, result in enumerate(results_sorted[:5], 1):
        if result.config[0] >= 0:  # Skip baseline
            delta = (result.score - baseline.score) * 100
            print(f"{i}. Layers {result.config}: {result.score:.2%} (+{delta:.2f}pp)")
    """

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Uncomment scanning code above to run full experiment")
    print("2. Try different models (edit MODEL_NAME)")
    print("3. Try different evaluation tasks (edit create_simple_math_dataset)")
    print("4. Implement actual layer duplication (see manual_forward_with_duplication)")


if __name__ == "__main__":
    main()
