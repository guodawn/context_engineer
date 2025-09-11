#!/usr/bin/env python3
"""
测试新的精确压缩逻辑 - 完全按照allocated_tokens进行压缩
"""

from context_engineer.core.budget_manager import BudgetManager, CompatBucketConfig as BucketConfig
from context_engineer.core.context_assembler import ContextAssembler
from context_engineer.core.tokenizer_service import TokenizerService

def test_precise_compression():
    print("=== 测试精确压缩逻辑 ===\n")
    
    # 创建tokenizer服务
    tokenizer = TokenizerService()
    
    # 创建预算管理器
    bm = BudgetManager(tokenizer)
    
    # 配置桶
    buckets = {
        "system_safety": BucketConfig("system_safety", min_tokens=50, max_tokens=200, weight=1.0, 
                                    sticky=True, content_score=0.9),
        "task_instructions": BucketConfig("task_instructions", min_tokens=30, max_tokens=150, weight=0.8, 
                                         sticky=True, content_score=0.85),
        "history": BucketConfig("history", min_tokens=100, max_tokens=400, weight=0.6, 
                               droppable=True, content_score=0.7),
        "tools_schema": BucketConfig("tools_schema", min_tokens=80, max_tokens=300, weight=0.9,
                                    content_score=0.8),
    }
    
    for bucket in buckets.values():
        bm.add_bucket(bucket)
    
    bm.set_drop_order(["history"])
    
    # 场景1：内容超过预算，需要压缩
    print("场景1：内容超过预算，需要精确压缩")
    
    # 创建大量内容（故意超过预算）
    content_sections = {
        "system_safety": "This is a very long system safety message that contains important safety guidelines and warnings. " * 10,
        "task_instructions": "These are detailed task instructions that explain exactly what needs to be done step by step. " * 8,
        "history": "This is conversation history with many previous exchanges between user and assistant. " * 15,
        "tools_schema": "These are tool schemas and function definitions with detailed parameter specifications. " * 12,
    }
    
    # 计算每个section的实际token数量
    actual_tokens = {}
    for name, content in content_sections.items():
        actual_tokens[name] = tokenizer.count_tokens(content)
        print(f"{name}: 实际内容 {actual_tokens[name]} tokens")
    
    print()
    
    # 进行预算分配
    budget_allocations = bm.allocate_budget(
        model_context_limit=1000,  # 总预算1000 tokens
        output_budget=200,         # 输出预留200 tokens  
        content_scores={
            "system_safety": 0.95,
            "task_instructions": 0.90,
            "history": 0.70,
            "tools_schema": 0.85,
        }
    )
    
    print("预算分配结果:")
    total_allocated = 0
    for alloc in budget_allocations:
        total_allocated += alloc.allocated_tokens
        compression_marker = " (需要压缩)" if alloc.compression_needed else ""
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens{compression_marker}")
    print(f"总分配: {total_allocated} tokens")
    print()
    
    # 创建上下文组装器并组装内容
    assembler = ContextAssembler(tokenizer)
    
    assembled_context = assembler.assemble_context(
        content_sections=content_sections,
        budget_allocations=budget_allocations,
        placement_policy={
            "head": ["system_safety", "task_instructions"],
            "middle": ["tools_schema", "history"],
            "tail": []
        }
    )
    
    print("精确压缩结果:")
    for section in assembled_context.sections:
        actual_tokens = tokenizer.count_tokens(section.content)
        allocated = section.allocated_tokens
        difference = actual_tokens - allocated
        
        print(f"  {section.name}:")
        print(f"    分配预算: {allocated} tokens")
        print(f"    实际内容: {actual_tokens} tokens")
        print(f"    压缩结果: {actual_tokens} tokens")
        print(f"    差异: {difference:+d} tokens")
        print(f"    需要压缩: {section.compression_needed}")
        print(f"    内容预览: {section.content[:100]}...")
        print()
    
    print(f"总token数: {assembled_context.total_tokens}")
    print(f"丢弃的section: {assembled_context.dropped_sections}")
    
    print("\n" + "="*60)
    print("场景2：内容符合预算，无需压缩")
    
    # 创建较小的内容（符合预算）
    small_content = {
        "system_safety": "Short safety note.",
        "task_instructions": "Brief instructions.",
        "history": "Minimal history.",
        "tools_schema": "Simple schema.",
    }
    
    # 计算实际token数
    for name, content in small_content.items():
        tokens = tokenizer.count_tokens(content)
        print(f"{name}: 实际内容 {tokens} tokens")
    
    # 使用相同的预算分配（因为预算分配主要取决于模型限制和权重）
    assembled_small = assembler.assemble_context(
        content_sections=small_content,
        budget_allocations=budget_allocations,  # 使用相同的预算
        placement_policy={
            "head": ["system_safety", "task_instructions"],
            "middle": ["tools_schema", "history"],
            "tail": []
        }
    )
    
    print("\n无需压缩的结果:")
    for section in assembled_small.sections:
        actual_tokens = tokenizer.count_tokens(section.content)
        allocated = section.allocated_tokens
        
        print(f"  {section.name}:")
        print(f"    分配预算: {allocated} tokens")
        print(f"    实际内容: {actual_tokens} tokens")
        print(f"    需要压缩: {section.compression_needed}")
        if section.compression_needed:
            print(f"    ⚠️  警告：标记为需要压缩但内容符合预算")
        print()

if __name__ == "__main__":
    test_precise_compression()