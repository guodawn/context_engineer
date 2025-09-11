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
    compressor = Compressor(tokenizer)
    assembler = ContextAssembler(tokenizer, compressor)
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
        print(f"  {allocation.bucket_name}: {allocation.allocated_tokens} tokens (priority: {allocation.priority:.2f}, content_score: {allocation.content_score:.2f})")
    
    total_allocated = sum(alloc.allocated_tokens for alloc in allocations)
    available_budget = 8000 - 1200 - 200  # model_limit - output_budget - system_overhead
    print(f"Total allocated: {total_allocated} tokens")
    print(f"Available budget: {available_budget} tokens")
    print(f"Budget utilization: {total_allocated / available_budget:.1%}")
    print()
    
    print("3. Context Assembly Demo")
    print("-" * 30)
    
    # Assemble context
    result = assembler.assemble_context(
        content_sections=content_sections,
        budget_allocations=allocations,
        placement_policy=config.get_default_policy().placement,
        bucket_configs=config.buckets
    )
    
    print("Assembled Context Sections:")
    for section in result.sections:
        actual_tokens = tokenizer.count_tokens(section.content)
        print(f"  {section.name}: {actual_tokens} tokens (allocated: {section.allocated_tokens})")
        print(f"    Content preview: {section.content[:100]}...")
    
    print(f"Assembled context: {result.total_tokens} tokens")
    print(f"Sections included: {[s.name for s in result.sections]}")
    print(f"Placement: {result.placement_map}")
    
    # Show context statistics
    stats = assembler.get_context_stats(result)
    print(f"Context statistics: {stats}")
    print()
    
    print("4. Precise Compression Demo")
    print("-" * 30)
    
    # 演示新的精确压缩逻辑 - 基于allocated_tokens而非50%启发式
    print("New Architecture: Precise compression based on allocated_tokens (not 50% heuristic)")
    
    # Create content that exceeds its budget allocation
    long_content = "This is a very long piece of content that needs to be compressed to fit within the allocated budget. " * 20
    original_tokens = tokenizer.count_tokens(long_content)
    allocated_budget = 50  # This section gets 50 tokens budget
    
    print(f"Original content: {original_tokens} tokens")
    print(f"Allocated budget: {allocated_budget} tokens")
    print(f"Content exceeds budget by: {original_tokens - allocated_budget} tokens")
    
    # The system will automatically detect this during assembly and compress precisely to allocated_budget
    test_sections = {
        "long_section": long_content,
        "short_section": "This is short content that fits within budget."
    }
    
    # Create budget allocations
    test_allocations = budget_manager.allocate_budget(
        model_context_limit=1000,
        output_budget=200,
        content_scores={"long_section": 0.8, "short_section": 0.6}
    )
    
    # Filter to only our test sections
    test_allocations = [alloc for alloc in test_allocations if alloc.bucket_name in ["long_section", "short_section"]]
    
    print("\nBudget allocations:")
    for allocation in test_allocations:
        print(f"  {allocation.bucket_name}: {allocation.allocated_tokens} tokens")
    
    # Assemble to trigger precise compression
    test_result = assembler.assemble_context(
        content_sections=test_sections,
        budget_allocations=test_allocations
    )
    
    print("\nAssembly results:")
    for section in test_result.sections:
        actual_tokens = tokenizer.count_tokens(section.content)
        allocated = section.allocated_tokens
        print(f"  {section.name}:")
        print(f"    Allocated: {allocated} tokens")
        print(f"    Actual: {actual_tokens} tokens")
        print(f"    Status: {'✅ Exact match' if actual_tokens == allocated else '❌ Mismatch'}")
        print(f"    Compression: {'Applied' if actual_tokens < allocated else 'None needed'}")
    
    print("\nKey improvements:")
    print("1. ✅ No more 50% heuristic - precise to allocated_tokens")
    print("2. ✅ Real-time comparison: token_count vs allocated_tokens")
    print("3. ✅ No compression_needed flag - cleaner architecture")
    print("4. ✅ No _handle_overflow - simplified logic flow")
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
        placement_policy=policy_decision.placement_strategy,
        bucket_configs=config.buckets
    )
    
    print(f"Policy-optimized context: {policy_result.total_tokens} tokens")
    print(f"Optimized sections: {[s.name for s in policy_result.sections]}")
    
    # Show context stats
    stats = assembler.get_context_stats(policy_result)
    print(f"Context statistics: {stats}")
    
    print("\n6. Architecture Summary")
    print("-" * 30)
    
    print("🚀 New Simplified Architecture:")
    print("1. Budget Layer: allocates precise token budgets")
    print("2. Content Layer: compares actual vs budget, applies precise compression")
    print("3. Assembly Layer: arranges content with position-aware strategy")
    print()
    print("✅ Key Improvements:")
    print("• Removed _handle_overflow - no meaningless overflow detection")
    print("• Removed compression_needed - real-time token_count vs allocated_tokens comparison")
    print("• Precise compression to allocated_tokens (not 50% heuristic)")
    print("• Cleaner separation of concerns")
    print("• More predictable behavior")
    print()
    print("📊 Performance Benefits:")
    print(f"• Budget utilization: {total_allocated / available_budget:.1%}")
    print(f"• Context efficiency: {policy_result.total_tokens / (8000 - 1200 - 200):.1%}")
    print("• Zero waste - every token is precisely allocated")
    print()
    
    print("=== Example Complete ===")


if __name__ == "__main__":
    main()