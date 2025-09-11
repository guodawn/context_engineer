#!/usr/bin/env python3
"""Test script to demonstrate compression strategies being applied."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from context_engineer import BudgetManager, ContextAssembler, TokenizerService, Compressor
from context_engineer.config.settings import get_default_config
from context_engineer.core.budget_manager import BudgetAllocation


def test_compression_strategies():
    """Test that different compression strategies are applied correctly."""
    
    print("=== Testing Compression Strategies ===\n")
    
    # Initialize components
    tokenizer = TokenizerService(backend="simple")
    budget_manager = BudgetManager(tokenizer)
    compressor = Compressor(tokenizer)
    assembler = ContextAssembler(tokenizer, compressor)
    
    # Load configuration
    config = get_default_config()
    budget_manager.configure_buckets(config.buckets)
    
    # Create content that will require compression for some sections
    content_sections = {
        "system": "You are a helpful assistant.",
        "task": "Summarize this technical documentation.",
        "history": "Previous conversation about the project requirements and timeline.",
        "tools": """Available tools:
        1. get_weather_data(location: str) - Get current weather data for a location
        2. get_forecast(location: str, days: int) - Get weather forecast for specified days
        3. analyze_data(data: str) - Analyze weather patterns and trends
        4. generate_report(analysis: str) - Generate formatted weather report
        5. send_notification(message: str, recipient: str) - Send notification to user""",
        "rag": """Weather patterns are complex atmospheric phenomena that involve the movement of air masses, 
        temperature variations, and precipitation. These patterns are influenced by factors such as geographic location, 
        seasonal changes, and global climate systems. Understanding weather patterns is crucial for accurate forecasting 
        and climate analysis. Recent studies have shown that machine learning models can improve weather prediction 
        accuracy by analyzing historical data and identifying subtle patterns that traditional methods might miss. 
        This has significant implications for agriculture, transportation, and disaster preparedness.""",
        "fewshot": """Example 1: Q: What's the weather like? A: I need your location to provide accurate weather information.
        Example 2: Q: Will it rain tomorrow? A: Let me check the forecast for your area.
        Example 3: Q: How hot will it get? A: I'll analyze the temperature trends for you.""",
    }
    
    # Create budget allocations that will force compression for some sections
    allocations = [
        BudgetAllocation("system", 20, 2.0),
        BudgetAllocation("task", 15, 1.5),
        BudgetAllocation("history", 10, 1.0, compression_needed=True),  # Force compression
        BudgetAllocation("tools", 30, 0.8, compression_needed=True),   # Force compression
        BudgetAllocation("rag", 25, 2.8, compression_needed=True),     # Force compression
        BudgetAllocation("fewshot", 20, 0.5, compression_needed=True), # Force compression
    ]
    
    print("1. Original content sizes:")
    for name, content in content_sections.items():
        tokens = tokenizer.count_tokens(content)
        print(f"  {name}: {tokens} tokens")
    
    print(f"\n2. Budget allocations (with compression needed):")
    for allocation in allocations:
        print(f"  {allocation.bucket_name}: {allocation.allocated_tokens} tokens (compression: {allocation.compression_needed})")
    
    print(f"\n3. Bucket compression methods:")
    for name, bucket in config.buckets.items():
        if hasattr(bucket, 'compress') and bucket.compress:
            print(f"  {name}: {bucket.compress}")
    
    print(f"\n4. Assembling context with compression strategies:")
    
    # Assemble context with compression
    result = assembler.assemble_context(
        content_sections=content_sections,
        budget_allocations=allocations,
        placement_policy=config.get_default_policy().placement,
        bucket_configs=config.buckets
    )
    
    print(f"\n5. Results after compression:")
    for section in result.sections:
        if section.compression_needed:
            original_tokens = tokenizer.count_tokens(content_sections.get(section.name, ""))
            print(f"  {section.name}: {original_tokens} -> {section.token_count} tokens (compressed)")
        else:
            print(f"  {section.name}: {section.token_count} tokens (no compression)")
    
    print(f"\n6. Final assembled context:")
    print(f"Total tokens: {result.total_tokens}")
    print(f"Sections: {[s.name for s in result.sections]}")
    
    # Show some compressed content examples
    print(f"\n7. Compressed content examples:")
    for section in result.sections:
        if section.compression_needed and section.name in ["tools", "rag", "fewshot"]:
            print(f"\n{section.name.upper()} (compressed):")
            print(f"{section.content[:200]}...")
    
    print("\n=== Compression Strategies Test Complete ===")


if __name__ == "__main__":
    test_compression_strategies()