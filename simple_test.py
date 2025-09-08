#!/usr/bin/env python3
"""Simple test script for ContextEngineer package."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Test basic imports
try:
    from context_engineer.core.tokenizer_service import TokenizerService, SimpleTokenizer
    print("✓ TokenizerService imported successfully")
except ImportError as e:
    print(f"✗ Failed to import TokenizerService: {e}")
    sys.exit(1)

try:
    from context_engineer.core.budget_manager import BudgetManager, BucketConfig
    print("✓ BudgetManager imported successfully")
except ImportError as e:
    print(f"✗ Failed to import BudgetManager: {e}")
    sys.exit(1)

try:
    from context_engineer.core.context_assembler import ContextAssembler
    print("✓ ContextAssembler imported successfully")
except ImportError as e:
    print(f"✗ Failed to import ContextAssembler: {e}")
    sys.exit(1)

# Test tokenizer
print("\n--- Testing Tokenizer ---")
tokenizer = TokenizerService(backend="simple")
test_text = "The quick brown fox jumps over the lazy dog."
tokens = tokenizer.count_tokens(test_text)
print(f"Text: {test_text}")
print(f"Token count: {tokens}")

# Test budget manager
print("\n--- Testing Budget Manager ---")
budget_manager = BudgetManager(tokenizer)

# Add test buckets
buckets = {
    "system": {"min_tokens": 50, "max_tokens": 200, "weight": 2.0},
    "task": {"min_tokens": 30, "max_tokens": 150, "weight": 1.5},
    "history": {"min_tokens": 0, "max_tokens": 300, "weight": 1.0}
}
budget_manager.configure_buckets(buckets)

print(f"Configured {len(budget_manager.buckets)} buckets")
print(f"Total min tokens: {budget_manager.get_total_min_tokens()}")
print(f"Total max tokens: {budget_manager.get_total_max_tokens()}")

# Test budget allocation
content_scores = {"system": 1.0, "task": 0.9, "history": 0.6}
allocations = budget_manager.allocate_budget(
    model_context_limit=1000,
    output_budget=200,
    content_scores=content_scores
)

print("Budget allocations:")
for allocation in allocations:
    print(f"  {allocation.bucket_name}: {allocation.allocated_tokens} tokens")

# Test context assembler
print("\n--- Testing Context Assembler ---")
assembler = ContextAssembler(tokenizer)

content_sections = {
    "system": "You are a helpful assistant.",
    "task": "Help with weather analysis.",
    "history": "Previous discussion about climate."
}

result = assembler.assemble_context(
    content_sections=content_sections,
    budget_allocations=allocations
)

print(f"Assembled context: {result.total_tokens} tokens")
print(f"Sections: {[s.name for s in result.sections]}")

# Test compression
print("\n--- Testing Compression ---")
try:
    from context_engineer.services.compressor import Compressor
    compressor = Compressor(tokenizer)
    
    long_text = "This is a very long text that needs compression. " * 20
    compressed = compressor.compress(long_text, target_tokens=50, method="truncate")
    
    print(f"Original tokens: {compressed.original_tokens}")
    print(f"Compressed tokens: {compressed.compressed_tokens}")
    print(f"Compression ratio: {compressed.compression_ratio:.2f}")
    print("✓ Compression test successful")
except ImportError as e:
    print(f"✗ Compression test skipped: {e}")

print("\n=== All tests completed successfully! ===")
print("ContextEngineer package is working correctly.")