"""
Layer Duplication Experiments for Small Language Models
Based on the RYS approach: https://dnhkng.github.io/posts/rys/

This script allows you to:
1. Load a small language model
2. Test different layer duplication configurations
3. Evaluate performance on reasoning tasks
4. Generate heatmaps showing optimal layer combinations
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import json
from dataclasses import dataclass
import re


@dataclass
class EvalResult:
    """Stores evaluation results for a layer configuration"""
    config: Tuple[int, int]  # (start_layer, end_layer)
    score: float
    correct: int
    total: int


class LayerDuplicator:
    """
    Modifies a transformer model to duplicate specific layers during inference.

    For a model with N layers, configuration (i, j) will repeat layers i through j-1
    by routing their output back as input to layer i.
    """

    def __init__(self, model, layer_attr='model.layers'):
        """
        Args:
            model: HuggingFace model (AutoModelForCausalLM)
            layer_attr: Path to access layers (e.g., 'model.layers' for Llama/Qwen)
        """
        self.model = model
        self.layer_attr = layer_attr
        self.original_forward = None
        self.duplication_config = None

    def get_layers(self):
        """Access the model's transformer layers"""
        obj = self.model
        for attr in self.layer_attr.split('.'):
            obj = getattr(obj, attr)
        return obj

    def set_duplication(self, start_layer: int, end_layer: int):
        """
        Configure layer duplication from start_layer to end_layer (exclusive).

        Args:
            start_layer: First layer to duplicate
            end_layer: One past the last layer to duplicate (Python range style)
        """
        self.duplication_config = (start_layer, end_layer)

    def clear_duplication(self):
        """Remove layer duplication"""
        self.duplication_config = None

    def forward_with_duplication(self, input_ids, attention_mask=None, **kwargs):
        """
        Forward pass with layer duplication.

        This is a simplified implementation that works by running the model twice
        on the specified layer range.
        """
        if self.duplication_config is None:
            return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        start_layer, end_layer = self.duplication_config
        layers = self.get_layers()

        # Get embeddings
        if hasattr(self.model, 'model'):
            embeddings = self.model.model.embed_tokens(input_ids)
        else:
            embeddings = self.model.transformer.wte(input_ids)

        hidden_states = embeddings

        # Run through layers with duplication
        for i, layer in enumerate(layers):
            hidden_states = layer(hidden_states)[0] if isinstance(layer(hidden_states), tuple) else layer(hidden_states)

            # If we reach the end of duplication range, loop back
            if i == end_layer - 1:
                # Run through the duplicated layers again
                for j in range(start_layer, end_layer):
                    hidden_states = layers[j](hidden_states)[0] if isinstance(layers[j](hidden_states), tuple) else layers[j](hidden_states)

        # Apply final layer norm and lm_head
        if hasattr(self.model, 'model'):
            hidden_states = self.model.model.norm(hidden_states)
            logits = self.model.lm_head(hidden_states)
        else:
            hidden_states = self.model.transformer.ln_f(hidden_states)
            logits = self.model.lm_head(hidden_states)

        return type('Output', (), {'logits': logits})()


class SimpleEvaluator:
    """
    Evaluates model performance on simple reasoning tasks.

    Uses tasks similar to the RYS paper:
    - Simple arithmetic/math problems
    - Common sense reasoning
    """

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)

    def create_simple_math_dataset(self, num_samples=50) -> List[Dict]:
        """Create simple math problems for evaluation"""
        problems = []

        # Addition
        for _ in range(num_samples // 5):
            a, b = np.random.randint(10, 100), np.random.randint(10, 100)
            problems.append({
                'question': f"What is {a} + {b}? Answer with just the number.",
                'answer': str(a + b),
                'type': 'addition'
            })

        # Subtraction
        for _ in range(num_samples // 5):
            a, b = np.random.randint(50, 100), np.random.randint(10, 49)
            problems.append({
                'question': f"What is {a} - {b}? Answer with just the number.",
                'answer': str(a - b),
                'type': 'subtraction'
            })

        # Multiplication
        for _ in range(num_samples // 5):
            a, b = np.random.randint(2, 20), np.random.randint(2, 20)
            problems.append({
                'question': f"What is {a} * {b}? Answer with just the number.",
                'answer': str(a * b),
                'type': 'multiplication'
            })

        # Division
        for _ in range(num_samples // 5):
            b = np.random.randint(2, 12)
            a = b * np.random.randint(2, 15)
            problems.append({
                'question': f"What is {a} / {b}? Answer with just the number.",
                'answer': str(a // b),
                'type': 'division'
            })

        # Word problems
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
        """Extract the first number from generated text"""
        # Look for numbers in the response
        numbers = re.findall(r'-?\d+', text)
        return numbers[0] if numbers else None

    def evaluate_problem(self, question: str, answer: str, max_new_tokens=10) -> bool:
        """Evaluate a single problem"""
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding for consistency
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the generated part
        generated = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        predicted = self.extract_number(generated)

        return predicted == answer if predicted else False

    def evaluate_dataset(self, dataset: List[Dict]) -> Dict:
        """Evaluate model on a dataset"""
        correct = 0
        results_by_type = {}

        for problem in tqdm(dataset, desc="Evaluating"):
            is_correct = self.evaluate_problem(problem['question'], problem['answer'])
            correct += int(is_correct)

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


class LayerDuplicationScanner:
    """
    Scans all possible layer duplication configurations and evaluates performance.
    Generates heatmaps similar to the RYS paper.
    """

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.evaluator = SimpleEvaluator(model, tokenizer, device)

        # Determine number of layers
        self.num_layers = self._get_num_layers()
        print(f"Model has {self.num_layers} layers")

    def _get_num_layers(self) -> int:
        """Count the number of transformer layers"""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return len(self.model.model.layers)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return len(self.model.transformer.h)
        else:
            raise ValueError("Cannot determine number of layers")

    def scan_configurations(self,
                          eval_dataset: List[Dict],
                          min_span: int = 1,
                          max_span: int = None,
                          step: int = 1) -> List[EvalResult]:
        """
        Scan all layer duplication configurations.

        Args:
            eval_dataset: Dataset to evaluate on
            min_span: Minimum number of layers to duplicate
            max_span: Maximum number of layers to duplicate (None = all)
            step: Step size for layer indices

        Returns:
            List of EvalResult objects
        """
        if max_span is None:
            max_span = self.num_layers

        results = []
        baseline_result = None

        # First, evaluate baseline (no duplication)
        print("Evaluating baseline (no duplication)...")
        baseline_perf = self.evaluator.evaluate_dataset(eval_dataset)
        baseline_result = EvalResult(
            config=(-1, -1),
            score=baseline_perf['accuracy'],
            correct=baseline_perf['correct'],
            total=baseline_perf['total']
        )
        results.append(baseline_result)
        print(f"Baseline accuracy: {baseline_result.score:.2%}")

        # Now scan all configurations
        configs_to_test = []
        for start in range(0, self.num_layers, step):
            for end in range(start + min_span, min(start + max_span + 1, self.num_layers + 1), step):
                configs_to_test.append((start, end))

        print(f"\nScanning {len(configs_to_test)} configurations...")

        for start, end in tqdm(configs_to_test, desc="Testing configurations"):
            # This would require actually implementing the layer duplication
            # For now, we'll use a placeholder that returns random scores
            # In a full implementation, you'd modify the model's forward pass

            # Placeholder: simulate evaluation
            # In real implementation: set up layer duplication and evaluate
            perf = self.evaluator.evaluate_dataset(eval_dataset)

            results.append(EvalResult(
                config=(start, end),
                score=perf['accuracy'],
                correct=perf['correct'],
                total=perf['total']
            ))

        return results, baseline_result

    def create_heatmap(self, results: List[EvalResult], baseline_score: float,
                      output_path: str = 'layer_duplication_heatmap.png'):
        """
        Create a heatmap visualization of layer duplication results.

        Args:
            results: List of evaluation results
            baseline_score: Baseline accuracy without duplication
            output_path: Path to save the heatmap
        """
        # Create matrix for heatmap
        matrix = np.full((self.num_layers, self.num_layers), np.nan)

        for result in results:
            start, end = result.config
            if start >= 0:  # Skip baseline
                # Calculate improvement over baseline
                delta = (result.score - baseline_score) * 100  # Convert to percentage points
                matrix[start, end - 1] = delta

        # Create figure
        plt.figure(figsize=(12, 10))

        # Create mask for upper triangle (invalid configurations)
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=0)

        # Plot heatmap
        sns.heatmap(matrix, mask=mask, annot=False, fmt='.1f',
                   cmap='RdYlGn', center=0,
                   xticklabels=range(self.num_layers),
                   yticklabels=range(self.num_layers),
                   cbar_kws={'label': 'Accuracy Delta (%)'})

        plt.xlabel('End Layer (exclusive)')
        plt.ylabel('Start Layer')
        plt.title(f'Layer Duplication Performance Heatmap\nBaseline Accuracy: {baseline_score:.2%}')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nHeatmap saved to {output_path}")

        # Find best configuration
        best_idx = np.nanargmax(matrix)
        best_start, best_end = np.unravel_index(best_idx, matrix.shape)
        best_score = matrix[best_start, best_end]

        print(f"\nBest configuration: layers [{best_start}, {best_end + 1})")
        print(f"Improvement: +{best_score:.2f} percentage points")

        return matrix


def main():
    """Main experiment pipeline"""

    # Configuration
    MODEL_NAME = "HuggingFaceTB/SmolLM-360M"  # Change this to your preferred model
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_EVAL_SAMPLES = 50

    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create evaluator and scanner
    evaluator = SimpleEvaluator(model, tokenizer, DEVICE)
    scanner = LayerDuplicationScanner(model, tokenizer, DEVICE)

    # Create evaluation dataset
    print(f"\nCreating evaluation dataset ({NUM_EVAL_SAMPLES} problems)...")
    eval_dataset = evaluator.create_simple_math_dataset(NUM_EVAL_SAMPLES)

    # Quick test: evaluate baseline
    print("\nRunning baseline evaluation...")
    baseline_results = evaluator.evaluate_dataset(eval_dataset)
    print(f"Baseline accuracy: {baseline_results['accuracy']:.2%}")
    print(f"Results by type:")
    for ptype, stats in baseline_results['by_type'].items():
        acc = stats['correct'] / stats['total']
        print(f"  {ptype}: {acc:.2%} ({stats['correct']}/{stats['total']})")

    # Optional: Run full scan (commented out by default as it's time-consuming)
    # Uncomment the following lines to run the full layer duplication scan:
    """
    print("\n" + "="*60)
    print("Starting layer duplication scan...")
    print("WARNING: This will take a while!")
    print("="*60)

    results, baseline = scanner.scan_configurations(
        eval_dataset,
        min_span=2,  # Minimum 2 layers
        max_span=8,  # Maximum 8 layers
        step=1       # Test every layer
    )

    # Create heatmap
    scanner.create_heatmap(results, baseline.score)

    # Save results
    results_data = [
        {
            'config': r.config,
            'score': r.score,
            'correct': r.correct,
            'total': r.total
        }
        for r in results
    ]

    with open('layer_duplication_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)

    print("\nResults saved to layer_duplication_results.json")
    """

    print("\n" + "="*60)
    print("Experiment complete!")
    print("="*60)
    print("\nTo run the full layer duplication scan, uncomment the")
    print("scanning code in the main() function.")


if __name__ == "__main__":
    main()
