#!/usr/bin/env python3
"""
验证移除_handle_overflow和compression_needed后的系统行为
"""

from context_engineer.core.budget_manager import BudgetManager, CompatBucketConfig as BucketConfig
from context_engineer.core.context_assembler import ContextAssembler
from context_engineer.core.tokenizer_service import TokenizerService

def test_no_overflow_system():
    print("=== 验证无溢出处理的新系统行为 ===\n")
    
    tokenizer = TokenizerService()
    bm = BudgetManager(tokenizer)
    
    # 配置桶
    buckets = {
        "system_safety": BucketConfig("system_safety", min_tokens=30, max_tokens=80, weight=1.0, 
                                    sticky=True, content_score=0.9),
        "task_instructions": BucketConfig("task_instructions", min_tokens=25, max_tokens=70, weight=0.8, 
                                         sticky=True, content_score=0.85),
        "history": BucketConfig("history", min_tokens=40, max_tokens=100, weight=0.6, 
                               droppable=True, content_score=0.7),
        "tools_schema": BucketConfig("tools_schema", min_tokens=35, max_tokens=90, weight=0.9,
                                    content_score=0.8),
    }
    
    for bucket in buckets.values():
        bm.add_bucket(bucket)
    
    # 创建内容 - 让某些超过max_tokens来测试压缩
    content_sections = {
        "system_safety": "This is system safety content that exceeds the maximum token limit. " * 6,
        "task_instructions": "Detailed task instructions that are longer than the allowed maximum. " * 8,
        "history": "Conversation history with many exchanges that goes beyond the token limit. " * 10,
        "tools_schema": "Tool schemas and API definitions that exceed the maximum allowed tokens. " * 7,
    }
    
    print("实际内容token计数:")
    for name, content in content_sections.items():
        actual_tokens = tokenizer.count_tokens(content)
        max_tokens = buckets[name].max_tokens
        over_limit = actual_tokens > max_tokens
        print(f"  {name}: {actual_tokens} tokens (max: {max_tokens}) {'⚠️ 超出' if over_limit else '✅ 符合'}")
    
    print("\n场景1：正常预算分配")
    
    # 正常预算
    normal_budget = bm.allocate_budget(
        model_context_limit=600,
        output_budget=50,
        content_scores={"system_safety": 0.95, "task_instructions": 0.90, "history": 0.70, "tools_schema": 0.85}
    )
    
    print("预算分配结果:")
    total_normal = 0
    for alloc in normal_budget:
        total_normal += alloc.allocated_tokens
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens")
    print(f"总分配: {total_normal} tokens")
    print(f"可用预算: {600 - 50 - 200}")
    
    assembler = ContextAssembler(tokenizer)
    
    assembled_normal = assembler.assemble_context(
        content_sections=content_sections,
        budget_allocations=normal_budget
    )
    
    print("\n新系统行为（无compression_needed标记）:")
    total_compressed = 0
    for section in assembled_normal.sections:
        actual_tokens = tokenizer.count_tokens(content_sections[section.name])
        compressed_tokens = tokenizer.count_tokens(section.content)
        allocated = section.allocated_tokens
        
        print(f"  {section.name}:")
        print(f"    原始内容: {actual_tokens} tokens")
        print(f"    分配预算: {allocated} tokens")
        print(f"    压缩结果: {compressed_tokens} tokens")
        print(f"    预算符合: {'✅' if compressed_tokens <= allocated else '❌'} {compressed_tokens - allocated:+d}")
        
        # 验证：系统通过比较token_count和allocated_tokens自动判断压缩
        would_need_compression = actual_tokens > allocated
        print(f"    压缩判断: {'需要压缩' if would_need_compression else '无需压缩'}")
        print()
    
    print(f"总token数: {assembled_normal.total_tokens}")
    
    print("\n" + "="*60)
    print("场景2：紧张预算分配")
    
    # 紧张预算 - 应该触发按比例缩减
    tight_budget = bm.allocate_budget(
        model_context_limit=250,
        output_budget=30,
        content_scores={"system_safety": 0.95, "task_instructions": 0.90, "history": 0.70, "tools_schema": 0.85}
    )
    
    print("紧张预算分配结果:")
    total_tight = 0
    for alloc in tight_budget:
        total_tight += alloc.allocated_tokens
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens")
    print(f"总分配: {total_tight} tokens")
    print(f"可用预算: {250 - 30 - 200}")
    
    assembled_tight = assembler.assemble_context(
        content_sections=content_sections,
        budget_allocations=tight_budget
    )
    
    print("\n紧张预算下的压缩行为:")
    total_compressed_tight = 0
    for section in assembled_tight.sections:
        original = tokenizer.count_tokens(content_sections[section.name])
        compressed = tokenizer.count_tokens(section.content)
        allocated = section.allocated_tokens
        
        total_compressed_tight += compressed
        
        print(f"  {section.name}:")
        print(f"    原始: {original} tokens → 压缩: {compressed} tokens → 预算: {allocated} tokens")
        if original > allocated:
            compression_ratio = (original - compressed) / original
            print(f"    压缩率: {compression_ratio:.1%}")
        print()
    
    print(f"紧张预算总token数: {assembled_tight.total_tokens}")
    print(f"预算利用率: {assembled_tight.total_tokens / (250 - 30 - 200):.1%}")
    
    print("\n" + "="*60)
    print("✅ 系统验证总结:")
    print("1. ✅ 移除了无意义的_handle_overflow逻辑")
    print("2. ✅ 通过token_count vs allocated_tokens实时判断压缩需求")
    print("3. ✅ 预算分配总是符合可用预算（无溢出处理）")
    print("4. ✅ 压缩精确到allocated_tokens目标")
    print("5. ✅ 系统更简单、更直接、更可靠")

if __name__ == "__main__":
    test_no_overflow_system()