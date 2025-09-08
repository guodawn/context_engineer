#!/usr/bin/env python3
"""
Example usage of ContextEngineer package.
"""

from context_engineer import BudgetManager, ContextAssembler, TokenizerService, Compressor, PolicyEngine
from context_engineer.config.settings import get_default_config
from context_engineer.services.policy_engine import TaskType, RiskLevel, CostTarget, PolicyContext


def main():
    """Demonstrate ContextEngineer functionality."""
    
    print("=== ContextEngineer Example ===\n")
    
    # Initialize components
    tokenizer = TokenizerService(backend="simple")
    budget_manager = BudgetManager(tokenizer)
    assembler = ContextAssembler(tokenizer)
    compressor = Compressor(tokenizer)
    policy_engine = PolicyEngine()
    
    # Load default configuration
    config = get_default_config()
    budget_manager.configure_buckets(config.buckets)
    
    print("1. Tokenizer Service Demo")
    print("-" * 30)
    
    sample_text = "The quick brown fox jumps over the lazy dog."
    token_count = tokenizer.count_tokens(sample_text)
    estimate = tokenizer.estimate_tokens(sample_text)
    
    print(f"Sample text: {sample_text}")
    print(f"Token count: {token_count}")
    print(f"Estimate: {estimate}")
    print(f"Tokenizer info: {tokenizer.get_tokenizer_info()}")
    print()
    
    print("2. Budget Allocation Demo")
    print("-" * 30)
    
    # Define content sections
    content_sections = {
        "system": "You are a helpful AI assistant specialized in weather analysis.",
        "task": "Analyze the current weather patterns and provide a forecast.",
        "history": "User previously asked about temperature trends and was given general information about seasonal patterns.",
        "rag": "Recent weather data shows a low pressure system moving in from the west, bringing precipitation and cooler temperatures.",
        "tools": "Available tools: weather_api.get_current_conditions, weather_api.get_forecast, location_service.get_coordinates",
        "memory": "User location: San Francisco, CA. User preference: Metric units.",
        "fewshot": "Example: Q: What's the weather? A: I need your location to provide accurate weather information.",
        "scratchpad": "Thinking: Need to get current conditions first, then provide forecast based on location."
    }
    
    # Calculate content relevance scores
    task_description = "Analyze weather patterns and provide forecast"
    content_scores = {}
    for section, content in content_sections.items():
        # Simple relevance scoring based on keyword overlap
        task_words = set(task_description.lower().split())
        content_words = set(content.lower().split())
        overlap = len(task_words.intersection(content_words))
        content_scores[section] = min(1.0, overlap / len(task_words)) if task_words else 0.0
    
    # Allocate budget
    allocations = budget_manager.allocate_budget(
        model_context_limit=8000,
        output_budget=1200,
        content_scores=content_scores
    )
    
    print("Budget Allocations:")
    for allocation in allocations:
        print(f"  {allocation.bucket_name}: {allocation.allocated_tokens} tokens (priority: {allocation.priority:.2f})")
    
    total_allocated = sum(alloc.allocated_tokens for alloc in allocations)
    print(f"Total allocated: {total_allocated} tokens")
    print()
    
    print("3. Context Assembly Demo")
    print("-" * 30)
    
    # Assemble context
    result = assembler.assemble_context(
        content_sections=content_sections,
        budget_allocations=allocations,
        placement_policy=config.get_default_policy().placement
    )
    
    print(f"Assembled context: {result.total_tokens} tokens")
    print(f"Sections included: {[s.name for s in result.sections]}")
    print(f"Placement: {result.placement_map}")
    print()
    
    print("4. Compression Demo")
    print("-" * 30)
    
    # Compress a long section
    long_content = "This is a very long piece of content that needs to be compressed. " * 50
    target_tokens = 50
    
    compression_result = compressor.compress(
        content=long_content,
        target_tokens=target_tokens,
        method="extractive"
    )
    
    print(f"Original tokens: {compression_result.original_tokens}")
    print(f"Compressed tokens: {compression_result.compressed_tokens}")
    print(f"Compression ratio: {compression_result.compression_ratio:.2f}")
    print(f"Method used: {compression_result.method_used}")
    print(f"Compressed content preview: {compression_result.compressed_content[:100]}...")
    print()
    
    print("5. Policy Engine Demo")
    print("-" * 30)
    
    # Create policy context
    policy_context = PolicyContext(
        task_type=TaskType.RESEARCH,
        risk_level=RiskLevel.MEDIUM,
        cost_target=CostTarget.BALANCED,
        content_types={"weather_data", "forecasting"},
        priority_sections=["system", "task", "rag"],
        excluded_sections=["fewshot"],
        metadata={"domain": "meteorology", "urgency": "normal"}
    )
    
    # Select policy
    policy_decision = policy_engine.select_policy(policy_context)
    
    print(f"Selected policy: {policy_decision.policy_name}")
    print(f"Reasoning: {policy_decision.reasoning}")
    print(f"Bucket overrides: {policy_decision.bucket_overrides}")
    print(f"Placement strategy: {policy_decision.placement_strategy}")
    print(f"Compression methods: {policy_decision.compression_methods}")
    print()
    
    print("6. Full Integration Demo")
    print("-" * 30)
    
    # Apply policy-based configuration
    if policy_decision.bucket_overrides:
        for bucket_name, overrides in policy_decision.bucket_overrides.items():
            if bucket_name in budget_manager.buckets:
                bucket = budget_manager.buckets[bucket_name]
                for attr, value in overrides.items():
                    if hasattr(bucket, attr):
                        setattr(bucket, attr, value)
    
    # Re-allocate with policy-based configuration
    policy_allocations = budget_manager.allocate_budget(
        model_context_limit=8000,
        output_budget=1200,
        content_scores=content_scores
    )
    
    # Re-assemble with policy-based placement
    policy_result = assembler.assemble_context(
        content_sections=content_sections,
        budget_allocations=policy_allocations,
        placement_policy=policy_decision.placement_strategy
    )
    
    print(f"Policy-optimized context: {policy_result.total_tokens} tokens")
    print(f"Optimized sections: {[s.name for s in policy_result.sections]}")
    
    # Show context stats
    stats = assembler.get_context_stats(policy_result)
    print(f"Context statistics: {stats}")
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()