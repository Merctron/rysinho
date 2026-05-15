# Layer Duplication Experiments

A PyTorch implementation for exploring layer duplication in small language models, inspired by the RYS approach described in [this article](https://dnhkng.github.io/posts/rys/).

## What is Layer Duplication?

Layer duplication is a parameter-free technique to extend a model's computational depth by routing specific layers' outputs back as inputs, effectively making the model traverse the same layers multiple times during inference.

For a model with N layers:
- Configuration `(i, j)` duplicates layers `i` through `j-1`
- The model processes: layers 0→i → [i→j (repeated)] → j→N

This can reveal "functional circuits" - multi-layer units that perform complete cognitive operations.

## Recommended Small Language Models

The script is configured to work with various small models. Here are the best options for experimentation:

### Recommended for Quick Experiments (< 1B parameters)
- **SmolLM-360M** (default) - `HuggingFaceTB/SmolLM-360M`
- **Qwen2-0.5B** - `Qwen/Qwen2-0.5B`
- **Pythia-410M** - `EleutherAI/pythia-410m`

### Recommended for Better Results (1-3B parameters)
- **SmolLM-1.7B** - `HuggingFaceTB/SmolLM-1.7B`
- **TinyLlama-1.1B** - `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Qwen2-1.5B** - `Qwen/Qwen2-1.5B`
- **Phi-2** - `microsoft/phi-2`

## Installation

```bash
# Install dependencies
pip install -r requirements_layer_dup.txt

# If you have GPU support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### 1. Run Baseline Evaluation

First, test the script with baseline evaluation (no layer duplication):

```bash
python layer_duplication_experiment.py
```

This will:
- Load the model (SmolLM-360M by default)
- Create a simple math evaluation dataset
- Evaluate baseline performance
- Print accuracy results

### 2. Modify to Use Different Models

Edit the `MODEL_NAME` variable in `main()`:

```python
# For SmolLM variants
MODEL_NAME = "HuggingFaceTB/SmolLM-360M"
MODEL_NAME = "HuggingFaceTB/SmolLM-1.7B"

# For Qwen2 variants
MODEL_NAME = "Qwen/Qwen2-0.5B"
MODEL_NAME = "Qwen/Qwen2-1.5B"

# For TinyLlama
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# For Phi-2
MODEL_NAME = "microsoft/phi-2"
```

### 3. Run Full Layer Duplication Scan

Uncomment the scanning code in the `main()` function to run the full experiment:

```python
# Uncomment these lines in main():
results, baseline = scanner.scan_configurations(
    eval_dataset,
    min_span=2,  # Minimum 2 layers
    max_span=8,  # Maximum 8 layers
    step=1       # Test every layer
)

scanner.create_heatmap(results, baseline.score)
```

This will:
- Test all layer duplication configurations
- Generate a heatmap showing performance deltas
- Save results to JSON

**Warning**: Full scanning is time-consuming. For a 24-layer model, testing all configurations with step=1 means hundreds of forward passes.

## Understanding the Results

### Heatmap Interpretation

The generated heatmap shows:
- **X-axis**: End layer (exclusive)
- **Y-axis**: Start layer
- **Color**: Performance delta vs baseline
  - Green = improvement
  - Red = degradation
  - White = neutral

### Key Patterns to Look For

1. **Bright green regions**: Optimal layer combinations that improve performance
2. **Diagonal patterns**: May indicate hierarchical processing stages
3. **Clustered regions**: Suggest functional circuits that work together

## Implementation Details

### Current Limitations

The provided implementation is a **foundation** for experiments. Key limitations:

1. **Layer Duplication Mechanism**: The current `LayerDuplicator` class provides a framework but requires integration with the model's forward pass. The paper's approach modified the model architecture directly.

2. **Evaluation Tasks**: Uses simple arithmetic. The original paper used harder benchmarks (MATH, MMLU, etc.). You can extend with:
   - GSM8K for math reasoning
   - HellaSwag for common sense
   - ARC for science questions

3. **Performance**: No optimization for batch processing or caching

### Extending the Implementation

To fully implement layer duplication:

1. **Monkey-patch the model's forward method**:
```python
def modified_forward(self, hidden_states, *args, **kwargs):
    # Normal forward pass
    output = self.original_forward(hidden_states, *args, **kwargs)
    
    # If at duplication boundary, loop back
    if self.current_layer == self.dup_end:
        for layer_idx in range(self.dup_start, self.dup_end):
            hidden_states = self.layers[layer_idx](output[0])
        return hidden_states
    return output
```

2. **Use more sophisticated evaluation**:
```python
from lm_eval import evaluator
results = evaluator.simple_evaluate(
    model=model,
    tasks=["gsm8k", "arc_easy"],
    num_fewshot=0
)
```

3. **Add caching** to avoid re-computing layer activations

## Advanced Usage

### Custom Evaluation Datasets

Add your own evaluation tasks:

```python
def create_custom_dataset():
    return [
        {
            'question': 'Your question here',
            'answer': 'expected answer',
            'type': 'custom'
        },
        # ... more problems
    ]

evaluator = SimpleEvaluator(model, tokenizer)
results = evaluator.evaluate_dataset(create_custom_dataset())
```

### Focused Scanning

Instead of testing all configurations, focus on specific ranges:

```python
# Test only middle layers
results = scanner.scan_configurations(
    eval_dataset,
    min_span=3,
    max_span=6,
    step=2  # Skip every other layer for speed
)
```

### Export for Analysis

Results are saved as JSON for further analysis:

```python
import pandas as pd
import json

with open('layer_duplication_results.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
df['delta'] = (df['score'] - baseline_score) * 100

# Find top 10 configurations
top_10 = df.nlargest(10, 'delta')
print(top_10[['config', 'score', 'delta']])
```

## Research Questions to Explore

Using this framework, you can investigate:

1. **Scale Effects**: Do smaller models have different functional circuits than larger ones?
2. **Architecture Differences**: Do different model families (Qwen vs Llama vs Phi) show different patterns?
3. **Task Specificity**: Do optimal layer duplications differ for math vs reasoning vs language tasks?
4. **Training Effects**: Do base models vs instruction-tuned models have different patterns?
5. **Layer Granularity**: Are circuits always multi-layer, or can single layers be duplicated effectively?

## Expected Runtime

Approximate times on a single GPU (A100):

| Model Size | Baseline Eval | Full Scan (step=1) | Full Scan (step=2) |
|-----------|---------------|--------------------|--------------------|
| 360M      | 2-3 min       | 3-5 hours          | 45-90 min          |
| 1B        | 5-7 min       | 8-12 hours         | 2-3 hours          |
| 2-3B      | 10-15 min     | 15-24 hours        | 4-6 hours          |

**Tip**: Start with `step=2` or `step=3` to get faster initial results.

## Troubleshooting

### Out of Memory

```python
# Use smaller batch size or 8-bit quantization
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config
)
```

### Model-Specific Layer Access

Different architectures name layers differently:
- Llama/Qwen: `model.layers`
- GPT-2/GPT-Neo: `transformer.h`
- BLOOM: `transformer.h`

Adjust `layer_attr` in `LayerDuplicator` if needed.

## References

- Original article: https://dnhkng.github.io/posts/rys/
- HuggingFace Transformers: https://huggingface.co/docs/transformers
- LM Evaluation Harness: https://github.com/EleutherAI/lm-evaluation-harness

## Contributing

This is a research prototype. Improvements welcome:
- Better layer duplication implementation
- More evaluation tasks
- Batch processing for speed
- Visualization improvements
- Support for more model architectures

## License

MIT License - feel free to use and modify for your research.
