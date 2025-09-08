# ContextEngineer

A Python package for optimizing context management in large language models through token budgeting and dynamic context optimization.

## Features

- **Token Budget Management**: Intelligent allocation of tokens across context buckets
- **Dynamic Context Optimization**: Select, compress, and reorder information to maximize relevance
- **Position-Aware Assembly**: Mitigate "Lost in the Middle" effect through strategic placement
- **Multiple Compression Strategies**: Extractive, abstractive, and signature-only compression
- **Policy-Based Configuration**: Task-specific optimization strategies
- **Extensible Architecture**: Custom tokenizers, compressors, and policies

## Installation

```bash
pip install context-engineer
```

For development with all optional dependencies:
```bash
pip install context-engineer[all]
```

## Quick Start

```python
from context_engineer import BudgetManager, ContextAssembler, TokenizerService
from context_engineer.config.settings import get_default_config

# Load default configuration
config = get_default_config()

# Initialize components
tokenizer = TokenizerService()
budget_manager = BudgetManager(tokenizer)
assembler = ContextAssembler(tokenizer)

# Configure buckets from config
budget_manager.configure_buckets(config.buckets)

# Define content sections
content_sections = {
    "system": "You are a helpful AI assistant.",
    "task": "Help the user with their question.",
    "history": "Previous conversation about weather patterns.",
    "rag": "Retrieved information about current weather conditions."
}

# Allocate budget
allocations = budget_manager.allocate_budget(
    model_context_limit=8000,
    output_budget=1200,
    content_scores={"system": 1.0, "task": 0.9, "history": 0.6, "rag": 0.8}
)

# Assemble context
result = assembler.assemble_context(
    content_sections=content_sections,
    budget_allocations=allocations,
    placement_policy=config.get_default_policy().placement
)

print(f"Assembled context: {result.total_tokens} tokens")
print(f"Sections: {[s.name for s in result.sections]}")
```

## Architecture

The package implements the token budgeting algorithm from the ContextEngineer specification:

1. **8-Bucket System**: Context is divided into semantic buckets (System, Task, Tools, History, Memory, RAG, Few-shot, Scratchpad)
2. **Token Budgeting**: Proportional allocation with minimum/maximum constraints
3. **Dynamic Optimization**: ROI-based allocation using content relevance scores
4. **Position Strategy**: Head/Middle/Tail placement to avoid "Lost in the Middle"

## Core Components

### TokenizerService
Unified tokenization supporting multiple backends (simple regex-based, tiktoken).

### BudgetManager
Token allocation across context buckets with fallback strategies.

### ContextAssembler
Position-aware context assembly with compression support.

### Compressor
Multiple compression strategies: truncate, extractive, abstractive, signature-only.

### PolicyEngine
Task-specific optimization policies (research, code generation, conversation, etc.).

## Configuration

The package supports YAML/JSON configuration files:

```yaml
model:
  name: gpt-4
  context_limit: 8192
  output_target: 1200
  output_headroom: 300

buckets:
  system:
    min_tokens: 300
    max_tokens: 800
    weight: 2.0
    sticky: true
  task:
    min_tokens: 300
    max_tokens: 1500
    weight: 2.5
    sticky: true

policies:
  default:
    drop_order: [fewshot, rag, history, tools]
    placement:
      head: [system, task, tools]
      middle: [rag, history]
      tail: [scratchpad]
```

## Testing

Run tests with pytest:

```bash
pytest tests/
```

## License

MIT License - see LICENSE file for details.
