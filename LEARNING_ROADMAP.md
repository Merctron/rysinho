# Layer Duplication Experiments: Learning Roadmap

**Your Background**: Strong CS fundamentals, linear algebra, calculus, conceptual LLM knowledge, daily Python coding, but limited PyTorch/Transformer implementation experience.

**Goal**: Build deep mastery of the codebase through structured reading and experimentation.

---

## 📚 What You Have

1. **`layer_duplication_annotated.py`** - Production code with extensive narrative comments
2. **`TECHNICAL_GUIDE.md`** - Deep dive into transformers, PyTorch, and theory
3. **`PYTORCH_PATTERNS.md`** - Practical reference for PyTorch patterns
4. **`simple_layer_dup_example.py`** - Minimal working example (100 lines)
5. **`LAYER_DUPLICATION_README.md`** - User guide for running experiments

---

## 🗺️ Learning Path (40-60 hours to mastery)

### Phase 1: Foundation (8-12 hours)

**Objective**: Understand transformers and PyTorch basics

#### Day 1-2: Transformer Architecture (4-6 hours)

1. **Read** `TECHNICAL_GUIDE.md` Section 1 (Transformer Architecture)
   - Don't rush the math - work through it with pen and paper
   - Draw the tensor shapes at each step
   - Understand: Why attention? Why multi-head? Why residuals?

2. **Supplement** with external resources:
   - "The Illustrated Transformer" by Jay Alammar (visual explanation)
   - Watch Andrej Karpathy's "Let's build GPT" video (implementation from scratch)

3. **Exercise**: On paper, trace the full forward pass for a tiny example:
   - Input: `["Hello", "world"]` (2 tokens)
   - Vocabulary size: 10
   - Hidden dim: 4
   - 1 layer, 2 attention heads
   - Manually compute shapes at each step

#### Day 3-4: PyTorch Fundamentals (4-6 hours)

1. **Read** `PYTORCH_PATTERNS.md` sections 1-3
   - Try each code snippet in a notebook/REPL
   - Verify shapes with `print()`
   - Understand view vs reshape vs transpose

2. **Interactive exercises**:
   ```python
   import torch
   
   # Exercise 1: Shape manipulation
   x = torch.randn(32, 10, 768)  # Batch of sequences
   # Task: Reshape to (32*10, 768), then back
   # Task: Split into 12 heads of dimension 64
   # Task: Transpose to (10, 32, 768)
   
   # Exercise 2: Attention mechanism
   Q = torch.randn(1, 4, 8)  # 1 batch, 4 tokens, 8-dim
   K = torch.randn(1, 4, 8)
   V = torch.randn(1, 4, 8)
   # Task: Compute attention scores = Q @ K.T / sqrt(8)
   # Task: Apply softmax
   # Task: Compute output = scores @ V
   
   # Exercise 3: Masking
   scores = torch.randn(4, 4)  # Attention scores
   # Task: Create causal mask (upper triangular = -inf)
   # Task: Add to scores and apply softmax
   # Task: Verify future tokens have 0 probability
   ```

3. **Mini-project**: Implement a simple 2-layer neural network from scratch
   ```python
   class TinyMLP(nn.Module):
       def __init__(self):
           super().__init__()
           self.layer1 = nn.Linear(10, 20)
           self.layer2 = nn.Linear(20, 5)
       
       def forward(self, x):
           # TODO: implement with ReLU activation
           pass
   
   # Train on dummy data
   model = TinyMLP()
   optimizer = torch.optim.Adam(model.parameters())
   # ... implement training loop
   ```

---

### Phase 2: Code Reading (12-16 hours)

**Objective**: Deeply understand the layer duplication implementation

#### Day 5-6: HuggingFace Model Structure (6-8 hours)

1. **Read** `TECHNICAL_GUIDE.md` Section 3 (HuggingFace Model Internals)

2. **Hands-on exploration**:
   ```python
   from transformers import AutoModelForCausalLM
   
   model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-360M")
   
   # Exercise 1: Navigate the model structure
   print(model)  # See full architecture
   
   # Exercise 2: Access components
   embeddings = model.model.embed_tokens
   first_layer = model.model.layers[0]
   attention = first_layer.self_attn
   
   # Exercise 3: Inspect a layer's forward signature
   import inspect
   print(inspect.signature(first_layer.forward))
   
   # Exercise 4: Trace a forward pass with hooks
   def hook_fn(module, input, output):
       print(f"{module.__class__.__name__}: {output[0].shape if isinstance(output, tuple) else output.shape}")
   
   for layer in model.model.layers:
       layer.register_forward_hook(hook_fn)
   
   input_ids = torch.tensor([[1, 2, 3, 4, 5]])
   model(input_ids)  # Watch shapes propagate
   ```

3. **Checkpoint**: Answer these questions
   - Where are the embeddings stored?
   - How many parameters in one transformer layer?
   - What does `model.model.layers[5]` return?
   - What's the difference between `model.model` and `model`?
   - Where is the causal mask applied?

#### Day 7-8: Layer Duplication Implementation (6-8 hours)

1. **Read** `layer_duplication_annotated.py` top to bottom
   - Read comments first, then code
   - For each function, predict what it does before reading the implementation
   - Draw diagrams of data flow

2. **Focus areas**:
   - `manual_forward_with_duplication()`: How does layer repetition work?
   - `SimpleEvaluator.evaluate_problem()`: How is text generation implemented?
   - `LayerDuplicationScanner.scan_configurations()`: How are configs enumerated?
   - `create_heatmap()`: How is the matrix built from results?

3. **Active reading exercises**:
   - Add `print()` statements to trace execution
   - Modify `manual_forward_with_duplication()` to print shapes
   - Change duplication config and predict what happens
   - Add assertions to check invariants

4. **Checkpoint**: Explain to yourself (or write down)
   - What happens to hidden states during layer duplication?
   - Why do we extract only the number from generated text?
   - Why is baseline evaluation important?
   - How does the heatmap matrix get populated?

---

### Phase 3: Experimentation (12-16 hours)

**Objective**: Run experiments and build intuition

#### Day 9-10: Baseline Experiments (6-8 hours)

1. **Run** `python layer_duplication_annotated.py`
   - Observe baseline evaluation
   - Read the output carefully
   - Understand what each printed metric means

2. **Modify and re-run**:
   ```python
   # Experiment 1: Different models
   MODEL_NAME = "HuggingFaceTB/SmolLM-360M"  # Default
   # Try: "Qwen/Qwen2-0.5B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
   
   # Experiment 2: Different dataset sizes
   NUM_EVAL_SAMPLES = 50   # Default
   # Try: 10 (fast), 100 (thorough)
   
   # Experiment 3: Different evaluation tasks
   # Modify create_simple_math_dataset() to:
   # - Only addition (easier)
   # - Only multiplication (harder)
   # - Different number ranges
   ```

3. **Analysis questions**:
   - Which problem types are hardest for the model?
   - Does increasing dataset size change accuracy significantly?
   - How does baseline accuracy vary across models?

#### Day 11-12: Layer Duplication Experiments (6-8 hours)

1. **Implement simple duplication test**:
   ```python
   # Add to main():
   print("\n" + "="*80)
   print("TESTING SINGLE CONFIGURATION")
   print("="*80)
   
   # Test duplicating middle layers
   num_layers = len(model.model.layers)
   start, end = num_layers // 3, num_layers // 2
   
   print(f"\nDuplicating layers [{start}, {end})")
   
   # Manually evaluate with duplication
   # (You'll need to implement this - good exercise!)
   ```

2. **Run coarse scan** (uncomment scanning code, but use `step=3`):
   - Takes 1-2 hours
   - Generates heatmap
   - Find promising regions

3. **Analyze results**:
   - Which layer ranges improve performance?
   - Are improvements consistent across problem types?
   - Do you see patterns (blocks, scattered pixels)?
   - Does single-layer duplication work? (RYS predicts no)

4. **Hypothesize**:
   - Why do some layer ranges help while others hurt?
   - What might those helpful layers be doing?
   - How would you test your hypothesis?

---

### Phase 4: Deep Dive (8-12 hours)

**Objective**: Master the theory and implementation

#### Day 13-14: Layer Duplication Theory (4-6 hours)

1. **Read**:
   - `TECHNICAL_GUIDE.md` Section 4 (Layer Duplication Theory)
   - Original RYS article: https://dnhkng.github.io/posts/rys/
   - Related work on "circuits" in neural networks

2. **Critical thinking**:
   - What evidence supports the circuit hypothesis?
   - What alternative explanations could there be?
   - How would you design experiments to distinguish them?

3. **Paper reading exercise**:
   - Read the RYS article closely
   - Note their methodology: What did they test? What did they control for?
   - Compare their results to yours: Similarities? Differences?
   - What did they find about optimal layer spans?

#### Day 15-16: Advanced Implementation (4-6 hours)

1. **Implement missing pieces**:
   - Currently, `scan_configurations()` doesn't actually duplicate layers
   - Implement this properly using `manual_forward_with_duplication()`
   - Test that duplication actually changes outputs

2. **Code template**:
   ```python
   def scan_with_actual_duplication(self, eval_dataset, ...):
       for start, end in tqdm(configs_to_test):
           # Temporarily modify model to duplicate layers [start, end)
           
           # Option 1: Monkey-patch forward method
           original_forward = self.model.forward
           def duplicated_forward(input_ids, **kwargs):
               return manual_forward_with_duplication(
                   self.model, input_ids, start, end, **kwargs
               )
           self.model.forward = duplicated_forward
           
           # Evaluate
           perf = self.evaluator.evaluate_dataset(eval_dataset)
           
           # Restore original forward
           self.model.forward = original_forward
           
           # Store results
           results.append(EvalResult(...))
   ```

3. **Validation**:
   - Verify duplication changes outputs (compare to baseline)
   - Check that restoration works (baseline after = baseline before)
   - Test edge cases: duplicate first layer, last layer, all layers

---

### Phase 5: Mastery Projects (8-12 hours)

**Objective**: Demonstrate mastery through creative application

Choose 1-2 projects:

#### Project A: Cross-Model Analysis
Compare layer duplication patterns across model families:
- Qwen vs Llama vs GPT-2
- Do different architectures have different optimal duplications?
- Write up findings as a technical report

#### Project B: Task-Specific Circuits
Test if optimal duplication varies by task:
- Create evaluation datasets for: math, reasoning, factual recall, language
- Find best duplication config for each
- Hypothesis: Different tasks use different circuits

#### Project C: Dynamic Layer Duplication
Implement adaptive duplication:
- Duplicate different layers based on input properties
- E.g., if input looks like math, duplicate "math circuit"
- Requires input classification + conditional forward pass

#### Project D: Visualization Tool
Build interactive explorer:
- Streamlit/Gradio app
- Upload model → generate heatmap
- Click heatmap → see example outputs with that config
- Compare configs side-by-side

#### Project E: Theoretical Analysis
Mathematical investigation:
- Formalize layer duplication as recurrent application
- Analyze fixed points: Do hidden states converge with repeated application?
- Relate to optimization literature (proximal methods, iterative refinement)

---

## 🎯 Milestones and Checkpoints

### Milestone 1: Foundation Complete
**When**: After Phase 1 (Day 4)
**Check**: Can you...
- [ ] Explain attention mechanism to someone else?
- [ ] Write a simple PyTorch model from scratch?
- [ ] Manipulate tensor shapes confidently?
- [ ] Understand what each transformer layer computes?

### Milestone 2: Code Comprehension
**When**: After Phase 2 (Day 8)
**Check**: Can you...
- [ ] Trace a forward pass through the code by hand?
- [ ] Explain what layer duplication does at the implementation level?
- [ ] Modify the evaluation to use different tasks?
- [ ] Debug shape mismatches using print statements?

### Milestone 3: Experimental Intuition
**When**: After Phase 3 (Day 12)
**Check**: Can you...
- [ ] Run experiments and interpret results?
- [ ] Generate and read heatmaps?
- [ ] Form hypotheses about layer functions?
- [ ] Predict which layer ranges might be interesting?

### Milestone 4: Mastery
**When**: After Phase 4-5 (Day 16+)
**Check**: Can you...
- [ ] Implement layer duplication from scratch?
- [ ] Design novel experiments to test hypotheses?
- [ ] Critique the methodology and suggest improvements?
- [ ] Apply concepts to related problems?

---

## 📖 Reading Strategy

### For Code (`layer_duplication_annotated.py`)
1. **First pass**: Read top to bottom, comments only (30 min)
2. **Second pass**: Read code and comments together (2 hours)
3. **Third pass**: Run in debugger, step through line by line (2 hours)
4. **Fourth pass**: Modify and break things (1 hour)

### For Documentation (`TECHNICAL_GUIDE.md`, etc.)
1. **Skim** to get overall structure (15 min)
2. **Deep read** one section at a time with notes (30-60 min per section)
3. **Apply** concepts immediately in code or notebook
4. **Return** to reference as needed during coding

### Active Reading Techniques
- ✏️ Take notes by hand (forces processing)
- 💻 Type out examples (builds muscle memory)
- 🤔 Pause to predict before reading answers
- 📝 Summarize in your own words after each section
- 🔄 Teach concepts to someone (or rubber duck)

---

## 🛠️ Setup and Tools

### Required
```bash
pip install torch transformers numpy matplotlib seaborn tqdm
```

### Recommended
```bash
# Jupyter for interactive exploration
pip install jupyter

# Better tensor inspection
pip install torchinfo

# Profiling
pip install py-spy
```

### Optional but Helpful
```bash
# Weight & Biases for experiment tracking
pip install wandb

# LM Evaluation Harness for benchmarks
pip install lm-eval

# Type checking
pip install mypy
```

### IDE Setup
- **VS Code**: Install Python extension, Pylance
- **PyCharm**: Enable type hints inspection
- **Jupyter**: For interactive exploration

---

## 💡 Tips for Success

### Do:
- ✅ Go slow - depth beats speed
- ✅ Write code alongside reading
- ✅ Break things intentionally to understand
- ✅ Draw diagrams for complex concepts
- ✅ Take breaks - learning compounds

### Don't:
- ❌ Skip the math - work through it
- ❌ Just read - must code to internalize
- ❌ Move on without understanding
- ❌ Ignore errors - they teach
- ❌ Rush to experiments before foundations

### When Stuck:
1. **Read error message carefully** - usually tells you exactly what's wrong
2. **Print shapes** - 90% of PyTorch bugs are shape mismatches
3. **Simplify** - create minimal example that shows the issue
4. **Search** - PyTorch forum, Stack Overflow, GitHub issues
5. **Ask** - describe what you've tried, include code and error

---

## 🎓 Assessment: Do You Really Understand?

After completing the roadmap, you should be able to:

### Knowledge Questions
- What is the computational complexity of attention? Why?
- Why do transformers use layer normalization instead of batch normalization?
- What happens to gradients in deep networks? How do residuals help?
- What is the purpose of the key, query, and value in attention?

### Implementation Questions
- How would you add a new layer type to a HuggingFace model?
- How would you implement KV caching from scratch?
- How would you modify the code to duplicate non-contiguous layers?
- How would you batch evaluation for faster scanning?

### Design Questions
- How would you test if a specific layer is necessary?
- How would you find which layers are "similar" to each other?
- How would you extend duplication to other architectures (encoder-decoder)?
- How would you make duplication learnable (not just search)?

If you can answer most of these confidently, you've achieved mastery! 🎉

---

## 📚 Further Learning

### After This Project
1. **Implement attention from scratch** (no libraries)
2. **Read "Attention Is All You Need"** paper in detail
3. **Explore mechanistic interpretability** work (Anthropic, OpenAI)
4. **Study other architecture modifications**: sparse attention, linear attention, state space models
5. **Contribute to open source**: Fix bugs, add features to HuggingFace

### Related Topics
- Model pruning and compression
- Neural architecture search
- Interpretability and explainability
- Transfer learning and fine-tuning
- Efficient training techniques

---

## 🗓️ Sample Schedule

### Full-Time (2 weeks)
- Week 1: Phases 1-3 (Foundation → Experimentation)
- Week 2: Phases 4-5 (Theory → Mastery projects)

### Part-Time (6 weeks)
- Weeks 1-2: Phase 1 (Foundation)
- Weeks 3-4: Phase 2-3 (Code reading → Experiments)
- Weeks 5-6: Phase 4-5 (Theory → Projects)

### Weekends Only (3 months)
- Month 1: Phases 1-2
- Month 2: Phase 3
- Month 3: Phases 4-5

Adjust based on your schedule and prior knowledge. The key is consistency, not speed.

---

**You're ready to start! Begin with Phase 1, Day 1: Read `TECHNICAL_GUIDE.md` Section 1.** 

Take your time, enjoy the process, and remember: confusion is part of learning. Push through it, and understanding will come. Good luck! 🚀
