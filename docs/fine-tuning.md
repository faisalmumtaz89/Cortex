# Fine-Tuning Guide

Cortex supports interactive fine-tuning of models using LoRA (Low-Rank Adaptation) on Apple Silicon with MLX acceleration.

## Quick Start

1. Start Cortex:
```bash
cortex
```

2. Type `/finetune` to launch the interactive fine-tuning wizard

3. Follow the guided steps:
   - Select a base model to fine-tune
   - Provide training data (or create it interactively)
   - Configure training settings (or use presets)
   - Name your fine-tuned model
   - Start training!

## Features

### 🎯 Interactive Experience
- **Zero command complexity** - Just type `/finetune` and follow prompts
- **Guided workflow** - Step-by-step wizard with helpful defaults
- **Visual progress** - Real-time training progress with loss metrics
- **Progress feedback** - Live updates during training

### 🧠 Smart Configuration System
- **Model Parameter Detection**: Accurately detects actual parameters using SafeTensors headers, MLX format, and config.json
- **Automatic Categorization**: Models categorized by size (tiny <500M, small 500M-2B, medium 2B-8B, large 8B-20B, xlarge 20B+)
- **Dataset Analysis**: Analyzes dataset size, task type (chat/completion/structured), and content characteristics
- **Optimal Settings**: Automatically selects learning rates, epochs, LoRA parameters based on model and data
- **Memory Management**: Auto-applies quantization for large models to fit in available GPU memory
- **Time Estimation**: Provides accurate training time estimates based on model characteristics

### 🚀 Training Options

#### Quick Mode
- Optimized for speed and experimentation
- Fewer epochs with higher learning rates
- Perfect for testing concepts quickly

#### Balanced Mode (Recommended)
- Optimal balance of speed and quality
- Smart defaults based on model size
- Good for most production use cases

#### Quality Mode
- Best results with careful training
- More epochs with conservative settings
- Higher LoRA ranks for complex adaptations

#### Expert Mode
- Full control over all parameters
- Learning rate, LoRA rank, quantization
- Advanced users who know their requirements

### 📊 Dataset Formats

Supported formats:
- **JSONL** (recommended)
- **JSON**
- **CSV**
- **TXT** (Q&A or conversation format)

#### JSONL Format Example
```jsonl
{"prompt": "What is Python?", "response": "Python is a high-level programming language."}
{"prompt": "Explain OOP", "response": "Object-oriented programming is a paradigm..."}
```

#### CSV Format Example
```csv
prompt,response
"What is Python?","Python is a high-level programming language."
"Explain OOP","Object-oriented programming is a paradigm..."
```

#### Text Format Example
```
Q: What is Python?
A: Python is a high-level programming language.

Q: Explain OOP
A: Object-oriented programming is a paradigm...
```

### 🎨 Interactive Dataset Creation

Don't have a dataset? Create one interactively:

1. Choose "Create dataset interactively" when prompted
2. Enter prompt-response pairs one by one
3. Type 'done' when finished
4. Minimum 5 examples recommended

### 💾 Model Management

Fine-tuned models are saved as new, independent models:
- Original model remains untouched
- Fine-tuned model appears in `/model` list
- Can be loaded like any other model
- Supports all normal Cortex features

## Smart Configuration Technical Details

### Model Parameter Detection
The smart configuration system uses accurate parameter counting:

#### Detection Methods (in order of priority):
1. **SafeTensors Headers**: Reads tensor metadata without loading the full model
2. **MLX Format**: Uses MLX model structure to count parameters precisely  
3. **Config.json Analysis**: Extracts architecture details for calculation
4. **Size Estimation Fallback**: Improved size-based estimation as last resort

#### Model Categories and Settings:
```python
# Model size categories (parameters in billions)
MODEL_CATEGORIES = {
    "tiny": (0, 0.5),        # < 500M parameters (DistilBERT, small GPT-2)
    "small": (0.5, 2),       # 500M-2B (GPT-2, small Llama)  
    "medium": (2, 8),        # 2B-8B (Gemma-7B, Llama-2-7B)
    "large": (8, 20),        # 8B-20B (Llama-2-13B, Mistral variants)
    "xlarge": (20, inf)      # 20B+ (Llama-2-70B, GPT-3.5+)
}

# Category-specific optimal settings
CATEGORY_DEFAULTS = {
    "tiny": {
        "learning_rate": 5e-4,    # Higher LR for small models
        "epochs": 5,              # More epochs needed
        "lora_r": 8,              # Lower rank sufficient
        "lora_alpha": 16,
        "batch_size": 4,          # Can handle larger batches
    },
    "medium": {
        "learning_rate": 1e-4,    # Standard settings for most models
        "epochs": 3,
        "lora_r": 16,
        "lora_alpha": 32,
        "batch_size": 1,
    },
    "xlarge": {
        "learning_rate": 2e-5,    # Very conservative  
        "epochs": 2,
        "lora_r": 64,             # High rank for very large models
        "lora_alpha": 128,
        "batch_size": 1,
    }
}
```

### Dataset-Aware Adjustments
- **Small datasets (<100 examples)**: More epochs, less regularization
- **Large datasets (>5000 examples)**: Fewer epochs for efficiency
- **Long sequences (>2000 chars)**: Increased gradient accumulation
- **Task type detection**: Optimizes templates for chat vs completion

### Quantization Strategy
```python
# Auto-quantization based on model size
if model_size_gb > 30 and not config.quantization_bits:
    config.quantization_bits = 4    # 4-bit for very large models
elif model_size_gb > 13 and not config.quantization_bits:
    config.quantization_bits = 8    # 8-bit for medium-large models
```

## Technical Details

### LoRA Implementation
- **Efficient training** - Only trains small adapter weights (~1-5% of original model)
- **Memory optimized** - 4-bit quantization for large models (automatic)
- **Fast convergence** - Smart epoch selection based on model complexity
- **MLX accelerated** - Native Apple Silicon GPU utilization with Metal shaders

### MLX LoRA Trainer
- **Uses mlx_lm library**: Official MLX implementation for LoRA training
- **Fine-tuned model handling**: Handles fine-tuned models without re-conversion
- **SafeTensors format**: Saves adapters in standard SafeTensors format
- **Marker files**: Creates `fine-tuned.marker` for automatic detection

### Automatic Settings
- **Learning rate** - Precisely calculated based on actual parameter count
- **Batch size** - Optimized for Apple Silicon unified memory
- **LoRA rank** - Adjusted for model complexity (8 for tiny, 64 for xlarge)
- **Quantization** - Applied automatically based on available GPU memory
- **Gradient checkpointing** - Enabled for memory efficiency when needed

### Performance

Training time and memory usage vary widely based on model size, dataset length, and quantization. Use these as rough starting points and validate on your machine:

- Smaller models (<=7B) typically train faster and fit in lower memory tiers.
- Larger models require more memory and longer training times.

## Tips for Best Results

1. **Start small** - Try 10-20 high-quality examples first
2. **Be consistent** - Keep similar format across examples
3. **Use presets** - Start with Quick mode for testing
4. **Monitor loss** - Lower loss = better learning
5. **Test immediately** - Load and test your model right after training

## Example Workflow

```bash
cortex
> /finetune

[Step 1: Select Base Model]
Use currently loaded model 'llama-3.1-8b'? [Y/n]: y

[Step 2: Prepare Training Data]
1. Use existing file
2. Create dataset interactively
3. Use sample dataset
Select option: 1
Enter path to dataset: ~/my_data.jsonl
✓ Dataset validated: 50 examples

[Step 3: Configure Training]
Smart Configuration detected:
  Model: Medium (7.2B params, 14.5GB)
  Dataset: Small (50 examples)
  Task type: Chat
  
Select training preset:
1. Quick    (fast experimentation)
2. Balanced (recommended for most cases)
3. Quality  (best results, longer training)
4. Expert   (full customization)
Select preset: 2

⚡ Smart configuration applied:
   Using optimized settings for medium model
   Training for 4 epochs - extra iterations for small datasets
   Using standard learning rate (1e-4) - prevents overfitting

[Step 4: Name Your Model]
Enter name for fine-tuned model: llama-3.1-8b-custom

[Training Summary]
Configuration:

  Base model:    llama-3.1-8b
  Output model:  llama-3.1-8b-custom
  Dataset:       my_data.jsonl (50 examples)
  
  Model size:    Medium (7.2B params)
  Task type:     Chat
  Config source: Smart Balanced
  
  Epochs:        4
  Learning rate: 1e-4
  LoRA rank:     16
  Batch size:    1 (x8 acc.)
  
  Estimated time: ~8 minutes

Start fine-tuning? [Y/n]: y

⚡ Starting fine-tuning...
   ████████████████████ 100% | Epoch 4/4 | Loss: 0.8542

✓ Fine-tuning completed!
📍 Model saved to: ~/.cortex/mlx_models/llama-3.1-8b-custom
💡 To load your fine-tuned model:
   /model ~/.cortex/mlx_models/llama-3.1-8b-custom
   
   LoRA adapter size: 2.3 MB
   Base model: llama-3.1-8b

Load 'llama-3.1-8b-custom' now? [Y/n]: y
✓ Model loaded successfully!

> How do I use Python?
[Your fine-tuned model responds with customized knowledge]
```

## Troubleshooting

### Out of Memory
- Use Quick mode (1 epoch)
- Enable 4-bit quantization (automatic for large models)
- Reduce dataset size
- Close other applications

### Training Too Slow
- Use smaller LoRA rank (8 instead of 16)
- Reduce dataset size
- Ensure GPU acceleration is working (`/gpu` command)

### Poor Results
- Add more training examples
- Use Quality mode (3 epochs)
- Ensure consistent data format
- Check for data quality issues

## Advanced Usage

For advanced users who want more control:

### Custom Training Script
```python
from cortex.fine_tuning import LoRATrainer, TrainingConfig

config = TrainingConfig(
    epochs=3,
    learning_rate=1e-5,
    lora_r=32,
    quantization_bits=4
)

trainer = LoRATrainer(model_manager, config)
trainer.train(
    base_model_name="llama-3.1-8b",
    dataset_path="data.jsonl",
    output_name="llama-3.1-8b-expert"
)
```

### Batch Processing
Place multiple JSONL files in a directory and fine-tune sequentially for different domains.

## Future Enhancements

Planned features:
- [ ] Continual learning (add knowledge without forgetting)
- [ ] Multi-adapter support (switch between fine-tuned versions)
- [ ] Distributed training across multiple Macs
- [ ] Automatic hyperparameter optimization
- [ ] Training data quality analysis

## Support

For issues or questions:
- Check the troubleshooting section above
- Review your dataset format
- Ensure you have sufficient memory
- Report issues on GitHub
