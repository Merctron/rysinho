"""
Simple, concrete example of layer duplication in a small language model.

This script demonstrates the core concept with minimal code.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
import time


def duplicate_layers_forward(model, input_ids, start_layer: int, end_layer: int, attention_mask=None):
    """
    Forward pass with layer duplication.

    Runs layers [start_layer, end_layer) twice in the forward pass.

    Args:
        model: HuggingFace causal LM
        input_ids: Input token IDs
        start_layer: First layer to duplicate (inclusive)
        end_layer: Last layer to duplicate (exclusive)
        attention_mask: Optional attention mask

    Returns:
        Model output logits
    """
    # Access model components
    if hasattr(model, 'model'):  # Llama/Qwen style
        embed = model.model.embed_tokens
        layers = model.model.layers
        norm = model.model.norm
        lm_head = model.lm_head
    elif hasattr(model, 'transformer'):  # GPT-2 style
        embed = model.transformer.wte
        layers = model.transformer.h
        norm = model.transformer.ln_f
        lm_head = model.lm_head
    else:
        raise ValueError("Unsupported model architecture")

    # Get embeddings
    hidden_states = embed(input_ids)

    # Prepare attention mask if needed
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    # Create extended attention mask for transformer layers
    if attention_mask.dim() == 2:
        batch_size, seq_length = input_ids.shape
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones((seq_length, seq_length), dtype=torch.bool, device=input_ids.device),
            diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)

    # Forward pass through layers
    num_layers = len(layers)

    for layer_idx in range(num_layers):
        # Normal forward through this layer
        layer_outputs = layers[layer_idx](
            hidden_states,
            attention_mask=attention_mask,
        )

        # Extract hidden states (handle tuple outputs)
        if isinstance(layer_outputs, tuple):
            hidden_states = layer_outputs[0]
        else:
            hidden_states = layer_outputs

        # If we just finished the duplication range, run it again
        if layer_idx == end_layer - 1 and start_layer < end_layer:
            print(f"  → Duplicating layers [{start_layer}, {end_layer})")

            for dup_idx in range(start_layer, end_layer):
                layer_outputs = layers[dup_idx](
                    hidden_states,
                    attention_mask=attention_mask,
                )

                if isinstance(layer_outputs, tuple):
                    hidden_states = layer_outputs[0]
                else:
                    hidden_states = layer_outputs

    # Apply final normalization and output projection
    hidden_states = norm(hidden_states)
    logits = lm_head(hidden_states)

    return logits


def generate_with_layer_duplication(
    model,
    tokenizer,
    prompt: str,
    start_layer: int = 0,
    end_layer: int = 0,
    max_new_tokens: int = 50,
    temperature: float = 0.7
):
    """
    Generate text with layer duplication.

    Args:
        model: HuggingFace causal LM
        tokenizer: HuggingFace tokenizer
        prompt: Text prompt
        start_layer: First layer to duplicate (0 = no duplication)
        end_layer: Last layer to duplicate (0 = no duplication)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated text
    """
    device = next(model.parameters()).device

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs.get('attention_mask', None)

    # Generate tokens one at a time
    generated_ids = input_ids.clone()

    for _ in range(max_new_tokens):
        # Forward pass with layer duplication
        with torch.no_grad():
            logits = duplicate_layers_forward(
                model,
                generated_ids,
                start_layer,
                end_layer,
                attention_mask
            )

        # Get next token logits
        next_token_logits = logits[:, -1, :]

        # Apply temperature
        if temperature > 0:
            next_token_logits = next_token_logits / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Append to generated sequence
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        # Update attention mask
        if attention_mask is not None:
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((attention_mask.shape[0], 1), device=device)
            ], dim=1)

        # Stop if EOS token
        if next_token.item() == tokenizer.eos_token_id:
            break

    # Decode generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


def compare_configurations(model, tokenizer, prompt: str, configs: list):
    """
    Compare text generation across different layer duplication configurations.

    Args:
        model: HuggingFace causal LM
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt
        configs: List of (start, end) tuples to test
    """
    print(f"\n{'='*80}")
    print(f"Prompt: {prompt}")
    print(f"{'='*80}\n")

    num_layers = len(model.model.layers) if hasattr(model, 'model') else len(model.transformer.h)

    for start, end in configs:
        if start == 0 and end == 0:
            config_name = "Baseline (no duplication)"
        else:
            config_name = f"Duplicate layers [{start}, {end})"

        print(f"\n{config_name}")
        print("-" * 60)

        start_time = time.time()
        output = generate_with_layer_duplication(
            model, tokenizer, prompt,
            start_layer=start,
            end_layer=end,
            max_new_tokens=30,
            temperature=0.0  # Greedy for reproducibility
        )
        elapsed = time.time() - start_time

        print(f"Output: {output[len(prompt):]}")
        print(f"Time: {elapsed:.2f}s")


def main():
    """Run simple demonstration"""

    # Configuration
    MODEL_NAME = "HuggingFaceTB/SmolLM-360M"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {MODEL_NAME}...")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)

    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get number of layers
    num_layers = len(model.model.layers) if hasattr(model, 'model') else len(model.transformer.h)
    print(f"Model has {num_layers} layers")

    # Test prompts
    test_prompts = [
        "What is 15 + 27?",
        "The capital of France is",
        "To solve this problem, we need to",
    ]

    # Define configurations to test
    # Format: (start_layer, end_layer) where end is exclusive
    configs = [
        (0, 0),  # Baseline - no duplication
        (num_layers // 4, num_layers // 2),  # Duplicate first quarter to half
        (num_layers // 2, 3 * num_layers // 4),  # Duplicate middle section
    ]

    print(f"\nTesting configurations:")
    for start, end in configs:
        if start == 0 and end == 0:
            print(f"  - Baseline (no duplication)")
        else:
            print(f"  - Duplicate layers [{start}, {end})")

    # Run comparisons
    for prompt in test_prompts:
        compare_configurations(model, tokenizer, prompt, configs)

    print("\n" + "="*80)
    print("Experiment complete!")
    print("="*80)
    print("\nKey observations to look for:")
    print("- Does layer duplication change the output?")
    print("- Are certain layer ranges more effective?")
    print("- How does it affect generation time?")
    print("\nNext steps:")
    print("1. Try different layer ranges")
    print("2. Test on reasoning/math problems")
    print("3. Measure accuracy on benchmarks")
    print("4. Create heatmaps using layer_duplication_experiment.py")


if __name__ == "__main__":
    main()
