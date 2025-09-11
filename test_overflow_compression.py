#!/usr/bin/env python3
"""
专门测试触发溢出和压缩的场景
"""

from context_engineer.core.budget_manager import BudgetManager, CompatBucketConfig as BucketConfig
from context_engineer.core.context_assembler import ContextAssembler
from context_engineer.core.tokenizer_service import TokenizerService

def test_overflow_compression():
    print("=== 测试溢出触发压缩机制 ===\n")
    
    tokenizer = TokenizerService()
    bm = BudgetManager(tokenizer)
    
    # 配置桶 - 关键：让最小需求总量超过预算
    buckets = {
        "system_safety": BucketConfig("system_safety", min_tokens=50, max_tokens=100, weight=1.0, 
                                    sticky=True, content_score=0.9),
        "task_instructions": BucketConfig("task_instructions", min_tokens=40, max_tokens=80, weight=0.8, 
                                         sticky=True, content_score=0.85),
        "history": BucketConfig("history", min_tokens=60, max_tokens=120, weight=0.6, 
                               droppable=True, content_score=0.7),
        "tools_schema": BucketConfig("tools_schema", min_tokens=45, max_tokens=90, weight=0.9,
                                    content_score=0.8),
    }
    
    for bucket in buckets.values():
        bm.add_bucket(bucket)
    
    bm.set_drop_order(["history", "tools_schema", "task_instructions", "system_safety"])
    
    # 创建内容
    content_sections = {
        "system_safety": "System safety guidelines and security protocols must be followed at all times. " * 8,
        "task_instructions": "Detailed task execution instructions with step-by-step procedures. " * 10,
        "history": "Previous conversation history with multiple exchanges between user and assistant. " * 15,
        "tools_schema": "Tool definitions and API schemas with parameter specifications. " * 12,
    }
    
    print("实际内容token计数:")
    for name, content in content_sections.items():
        actual_tokens = tokenizer.count_tokens(content)
        print(f"  {name}: {actual_tokens} tokens")
    
    # 计算最小需求
    min_required = bm.get_total_min_tokens()
    print(f"\n最小需求总量: {min_required} tokens")
    
    # 设置预算小于最小需求，强制触发按比例缩减
    model_limit = min_required - 50  # 比最小需求还少50 tokens
    output_budget = 30
    system_overhead = 200
    available_budget = model_limit - output_budget - system_overhead
    
    print(f"\n紧张预算配置:")
    print(f"  模型限制: {model_limit}")
    print(f"  输出预算: {output_budget}")
    print(f"  系统开销: {system_overhead}")
    print(f"  可用预算: {available_budget}")
    print(f"  最小需求: {min_required}")
    print(f"  预算缺口: {available_budget - min_required}")
    
    budget_allocations = bm.allocate_budget(
        model_context_limit=model_limit,
        output_budget=output_budget,
        content_scores={
            "system_safety": 0.95,
            "task_instructions": 0.90,
            "history": 0.70,
            "tools_schema": 0.85,
        }
    )
    
    print(f"\n预算分配结果 (按比例缩减):")
    total_allocated = 0
    for alloc in budget_allocations:
        total_allocated += alloc.allocated_tokens
        compression_marker = " (需要压缩)" if alloc.compression_needed else ""
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens{compression_marker}")
    print(f"总分配: {total_allocated} tokens")
    print(f"可用预算: {available_budget}")
    
    # 现在测试压缩逻辑
    assembler = ContextAssembler(tokenizer)
    
    assembled_context = assembler.assemble_context(
        content_sections=content_sections,
        budget_allocations=budget_allocations
    )
    
    print(f"\n精确压缩结果:")
    for section in assembled_context.sections:
        original_tokens = tokenizer.count_tokens(content_sections[section.name])
        compressed_tokens = tokenizer.count_tokens(section.content)
        allocated = section.allocated_tokens
        
        print(f"  {section.name}:")
        print(f"    原始内容: {original_tokens} tokens")
        print(f"    分配预算: {allocated} tokens")
        print(f"    压缩结果: {compressed_tokens} tokens")
        print(f"    压缩率: {(original_tokens - compressed_tokens) / original_tokens:.1%}")
        print(f"    预算符合: {'✅' if compressed_tokens <= allocated else '❌'} {compressed_tokens - allocated:+d}")
        print(f"    需要压缩: {section.compression_needed}")
        print()
    
    print(f"总token数: {assembled_context.total_tokens}")
    
    # 验证精确性
    total_budget = sum(alloc.allocated_tokens for alloc in budget_allocations)
    total_compressed = sum(tokenizer.count_tokens(section.content) for section in assembled_context.sections)
    
    print(f"\n精确性验证:")
    print(f"总预算分配: {total_budget} tokens")
    print(f"总压缩结果: {total_compressed} tokens")
    print(f"差异: {total_compressed - total_budget:+d} tokens")
    
    if total_compressed <= total_budget:
        print("✅ 成功：精确压缩完全符合预算分配！")
    else:
        print("❌ 问题：压缩结果超出了预算分配")

if __name__ == "__main__":
    test_overflow_compression()