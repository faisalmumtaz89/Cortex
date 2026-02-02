# Command Line Interface

## Overview

Cortex provides a Claude Code-style interactive CLI for GPU-accelerated LLM inference on Apple Silicon. The interface focuses on simplicity and efficiency - just type `cortex` and start chatting with AI models.

## Main Command

```bash
cortex
```

**Note**: Cortex currently operates in interactive mode only. All functionality is accessed through the interactive terminal interface.

### Interactive Mode (Default and Only Mode)

When you run `cortex`, it starts the interactive terminal UI with:

```bash
cortex
```

**Features:**
- Simple, clean Claude Code-style interface
- Real-time GPU-accelerated text generation 
- Slash commands for model management
- Keyboard shortcuts for quick actions
- Command auto-completion via readline
- Command history persisted in `~/.cortex_history`
- Signal handling (Ctrl+C cancels generation or exits when idle)

## Tooling (Experimental, Work in Progress)

Cortex includes early tooling support to let the model read/search the current repo and propose file edits. This is **experimental** and **work in progress**.

**What it can do (today):**
- Read files and list directories in the current repo
- Search for text/regex matches
- Propose edits (create, write, replace, insert, delete) with explicit confirmation

**Scope and safety:**
- Tool paths are **repo-relative**. Cortex will not access files outside the repo root.
- Any write/delete operation **requires explicit approval** before it is applied.
- The tool system is evolving: behavior and output may change, tool calls may fail, and UI presentation may be adjusted.

**Important (WIP):** Treat tooling as a preview. Use it on non-critical tasks first, and always review proposed changes before approving them.

## Interactive Commands

All functionality in Cortex is accessed through slash commands in the interactive interface:

### `/help` - Show Available Commands

Display all available commands with descriptions.

**Usage:**
```bash
/help
```

### `/download [repo_id] [filename]` - Download Models

Download models from HuggingFace Hub with an interactive interface.

**Usage:**
```bash
# Interactive download menu with numbered options
/download

# Direct download from HuggingFace
/download microsoft/DialoGPT-medium

# Download specific GGUF file  
/download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

**Features:**
- **Interactive Menu**: Shows locally available models to load
- **Numbered Selection**: Choose models by number
- **Direct Download**: Specify HuggingFace repo ID directly
- **Auto-load Option**: Prompt to load downloaded model immediately
- **Progress Output**: Uses HuggingFace download progress when available

### `/model [model_path]` - Load Models

Load a model for inference or show available models.

**Usage:**
```bash
# Interactive model selector and loader
/model

# Load specific model by path
/model ~/models/Mistral-7B-Instruct-v0.2-4bit
```

**Supported Formats:**
- **MLX**: Native Apple Silicon format (recommended)
- **SafeTensors**: HuggingFace SafeTensors format  
- **PyTorch**: Standard HuggingFace models
- **GGUF**: Quantized models (experimental)
- **GPTQ / AWQ**: Quantized models (optional deps required)

**Interactive Mode Features:**
- Shows all available models with sizes
- Numbered selection for easy loading  
- Current model indicator
- Delete models option
- Download new models option

### `/status` - System Status

Display comprehensive system and model information.

**Usage:**
```bash
/status
```

**Information shown:**
- Apple Silicon chip details
- GPU cores and memory
- Currently loaded model
- System configuration

### `/gpu` - GPU Information

Show detailed GPU status and memory usage.

**Usage:**
```bash  
/gpu
```

**Output includes:**
- Chip name and specifications
- GPU core count
- Total and available memory
- Metal/MPS/MLX support status
- Current memory usage by models

### `/benchmark` - Performance Test

Run a performance benchmark with the current model.

**Usage:**
```bash
/benchmark
```

**Metrics reported:**
- Tokens generated (100 tokens)
- Total generation time
- Tokens per second
- First token latency
- GPU utilization percentage (approximate CPU-based proxy)
- Memory usage

### `/clear` - Clear Conversation

Clear the current conversation history.

**Usage:**
```bash
/clear
```

### `/save` - Save Conversation

Save the current conversation to a JSON file.

**Usage:**
```bash
/save
```

**Output:**
- Saves to `~/.cortex/conversations/` directory
- Filename includes timestamp
- JSON format with complete conversation history

### `/finetune` - Fine-Tune Models

Launch the interactive fine-tuning wizard to customize models with your data using LoRA (Low-Rank Adaptation).

**Usage:**
```bash
/finetune
```

**Features:**
- **Interactive Wizard**: Step-by-step guided process for fine-tuning
- **Smart Configuration**: Automatically detects model size and optimizes training settings
- **Model Parameter Detection**: Accurately detects model parameters for optimal training configuration
- **Multiple Training Presets**:
  - Quick: Fast experimentation (fewer epochs)
  - Balanced: Optimal balance of speed and quality (recommended)
  - Quality: Best results with longer training (more epochs)
  - Expert: Full customization of all parameters
- **Dataset Support**: JSONL, CSV, TXT formats or create data interactively
- **Real-time Progress**: Visual training progress with loss metrics
- **MLX Acceleration**: Native Apple Silicon GPU training using MLX
- **LoRA Adapters**: Memory-efficient training, preserves original model

**Example Workflow:**
```bash
cortex
> /finetune

[Step 1: Select Base Model]
Use currently loaded model 'llama-3.1-8b'? [Y/n]: y
✓ Detected: Medium (7.2B params, 14.5GB)

[Step 2: Training Data]
1. Load from file (JSONL/CSV/TXT)
2. Create interactively  
3. Use sample dataset (for testing)
Select option: 1
Enter path to dataset: ~/my_training_data.jsonl
✓ Dataset validated: 150 examples

[Step 3: Configuration]
Smart Configuration detected:
  Model: Medium (7.2B params, 14.5GB) 
  Dataset: Medium (150 examples)
  Task type: Chat
  
Select preset:
1. Quick    (fast experimentation)
2. Balanced (recommended for most cases) ← 
3. Quality  (best results, longer training)
4. Expert   (full customization)
Select preset: 2
✓ Smart configuration applied

[Step 4: Output Model Name]
Model name [llama-3.1-8b-finetuned]: my-custom-model

[Training Summary]
Base model:    llama-3.1-8b
Output model:  my-custom-model  
Dataset:       my_training_data.jsonl (150 examples)
Epochs:        3
Learning rate: 1e-4
LoRA rank:     16
Estimated time: ~15 minutes

Start fine-tuning? [Y/n]: y

⚡ Starting fine-tuning...
   ████████████████████ 100% | Epoch 3/3 | Loss: 0.8542
✓ Fine-tuning completed!
✓ Model saved to: ~/.cortex/mlx_models/my-custom-model
💡 To load: /model ~/.cortex/mlx_models/my-custom-model
```

**Smart Configuration:**
The wizard automatically:
- Detects accurate model parameters using SafeTensors headers, MLX format, and config.json
- Categorizes models: tiny (<500M), small (500M-2B), medium (2B-8B), large (8B-20B), xlarge (20B+)
- Adjusts learning rates, epochs, and LoRA settings based on model size
- Analyzes dataset size and task type (chat, completion, structured)
- Applies quantization for large models automatically
- Provides time estimates based on model characteristics

### `/template [model_name] [subcommand]` - Manage Templates

Configure and manage chat templates for models.

**Usage:**
```bash
# Interactive template configuration for current model
/template

# Configure specific model
/template Qwen2-1.5B-Instruct

# Show current template status
/template status

# Reset to auto-detected defaults
/template reset
```

**Features:**
- **Auto-detection**: Automatically detects ChatML, Llama, Alpaca, Reasoning formats
- **Reasoning Filtering**: Automatically hides internal reasoning tokens for reasoning models
- **Interactive Setup**: Configure templates for unknown models  
- **Real-time Processing**: Streaming filter for reasoning models maintains low latency
- **Toggle Options**: Show/hide internal reasoning for reasoning models (debugging)
- **Persistent Settings**: Configurations saved across sessions

### `/login` - Login to HuggingFace

Login to HuggingFace Hub to access gated models that require authentication.

**Usage:**
```bash
/login
```

**Features:**
- Authenticate with HuggingFace Hub
- Required for accessing gated models (e.g., Llama, Gemma)
- Stores token for future sessions

### `/quit` or `/exit` - Exit Cortex

Exit the Cortex application.

**Usage:**
```bash
/quit
/exit
```

**Note:** You can also use `Ctrl+D` or type `quit`/`exit` without the slash.

## Keyboard Shortcuts

Cortex supports several keyboard shortcuts for efficient interaction:

### Global Shortcuts

| Shortcut | Action | Description |
|----------|--------|-------------|
| `Ctrl+C` | Cancel generation | Stop current AI response generation |
| `Ctrl+D` | Exit application | Exit Cortex (same as `/quit`) |
| `Tab` | Auto-complete | Auto-complete slash commands (readline) |
| `?` | Show help | Display the in-app help table |

### Input Features

| Feature | Description |
|---------|-------------|
| **Command History** | Use Up/Down arrows to navigate command history |
| **Auto-completion** | Tab completion for slash commands |
| **Signal Handling** | Graceful handling of interruption signals |
| **Readline Support** | Full readline editing capabilities |

## User Interface

Cortex provides a clean, Claude Code-inspired interface:

### Welcome Screen
```
╭─ Welcome to Cortex! ──────────────────────────────╮
│                                                   │
│  /help for help, /status for your current setup   │
│                                                   │
│  cwd: /current/working/directory                  │
╰───────────────────────────────────────────────────╯

※ Tip: Use /download to get models from HuggingFace
```

### Simple Input Prompt
```
> your message here
```

### Generation Indicator
```
⏺ AI response appears here in real-time...
```

### Command Help Display
```
╭─ Available Commands ──────────────────────────────╮
│                                                   │
│  /help       Show this help message               │
│  /status     Show current setup and GPU info      │
│  /download   Download a model from HuggingFace    │
│  /model      Load a model                         │
│  /finetune   Fine-tune models interactively       │
│  /template   Manage model chat templates          │
│  /clear      Clear conversation history           │
│  /save       Save current conversation            │
│  /gpu        Show GPU status                      │
│  /benchmark  Run performance benchmark            │
│  /login      Login to HuggingFace for gated models│
│  /quit       Exit Cortex                          │
│                                                   │
╰───────────────────────────────────────────────────╯
```

## Getting Started

1. **Start Cortex:**
   ```bash
   cortex
   ```

2. **Check your setup:**
   ```bash
   > /status
   ```

3. **Download a model (if needed):**
   ```bash
   > /download
   ```

4. **Start chatting:**
   ```bash
   > Hello! How are you today?
   ```

## Configuration

Cortex uses a `config.yaml` file in the project root for configuration. Key settings:

- `model_path`: Directory where models are stored (`~/models` by default)
- `default_model`: Model to load on startup (optional)
- GPU and performance settings are configured in the YAML file

See the [Configuration Guide](configuration.md) for detailed configuration options.
