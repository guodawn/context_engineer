#!/usr/bin/env python3
"""
在合理场景下测试精确压缩逻辑
"""

from context_engineer.core.budget_manager import BudgetManager, CompatBucketConfig as BucketConfig
from context_engineer.core.context_assembler import ContextAssembler
from context_engineer.core.tokenizer_service import TokenizerService

def test_reasonable_compression():
    print("=== 在合理场景下测试精确压缩 ===\n")
    
    tokenizer = TokenizerService()
    bm = BudgetManager(tokenizer)
    
    # 配置合理的桶大小
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
    
    bm.set_drop_order(["history", "tools_schema", "task_instructions", "system_safety"])
    
    # 创建内容 - 故意让某些section超过max_tokens
    content_sections = {
        "system_safety": "Critical safety protocols and security guidelines must be followed. " * 6,  # ~90 tokens
        "task_instructions": "Step-by-step execution procedures and implementation guidelines. " * 8,  # ~105 tokens  
        "history": "Previous conversation exchanges and contextual information from earlier interactions. " * 10,  # ~150 tokens
        "tools_schema": "API function definitions and parameter specifications for available tools. " * 7,  # ~98 tokens
    }
    
    print("实际内容token计数:")
    for name, content in content_sections.items():
        actual_tokens = tokenizer.count_tokens(content)
        max_tokens = buckets[name].max_tokens
        over_limit = actual_tokens > max_tokens
        print(f"  {name}: {actual_tokens} tokens (max: {max_tokens}) {'⚠️ 超出限制' if over_limit else '✅ 符合限制'}")
    print()
    
    # 设置预算让某些内容超过分配
    model_limit = 300  # 相对紧张的预算
    output_budget = 30
    
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
    
    print(f"预算分配结果 (模型限制: {model_limit}, 输出: {output_budget}):")
    total_allocated = 0
    for alloc in budget_allocations:
        total_allocated += alloc.allocated_tokens
        compression_marker = " (需要压缩)" if alloc.compression_needed else ""
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens{compression_marker}")
    print(f"总分配: {total_allocated} tokens")
    print(f"可用预算: {model_limit - output_budget - 200}")
    print()
    
    # 关键测试：对比原始50%压缩 vs 新的精确压缩
    print("="*60)
    print("对比测试：原始内容 vs 预算分配")
    
    for name, content in content_sections.items():
        actual_tokens = tokenizer.count_tokens(content)
        allocated = next(alloc.allocated_tokens for alloc in budget_allocations if alloc.bucket_name == name)
        
        # 旧的50%压缩逻辑
        old_target = max(1, actual_tokens // 2)
        
        # 新的精确压缩逻辑
        new_target = allocated
        
        print(f"\n{name}:")
        print(f"  实际内容: {actual_tokens} tokens")
        print(f"  预算分配: {allocated} tokens")
        print(f"  旧50%目标: {old_target} tokens")
        print(f"  新精确目标: {new_target} tokens")
        print(f"  目标差异: {new_target - old_target:+d} tokens")
        
        if actual_tokens > allocated:
            print(f"  ⚠️  需要压缩：从 {actual_tokens} 压缩到 {allocated}")
        else:
            print(f"  ✅ 无需压缩：内容符合预算")
    
    print("\n" + "="*60)
    print("执行精确压缩:")
    
    assembler = ContextAssembler(tokenizer)
    
    assembled_context = assembler.assemble_context(
        content_sections=content_sections,
        budget_allocations=budget_allocations
    )
    
    print("精确压缩结果:")
    total_original = 0
    total_compressed = 0
    total_budget = 0
    
    for section in assembled_context.sections:
        original_tokens = tokenizer.count_tokens(content_sections[section.name])
        compressed_tokens = tokenizer.count_tokens(section.content)
        allocated = section.allocated_tokens
        
        total_original += original_tokens
        total_compressed += compressed_tokens
        total_budget += allocated
        
        print(f"  {section.name}:")
        print(f"    原始内容: {original_tokens} tokens")
        print(f"    分配预算: {allocated} tokens")
        print(f"    压缩结果: {compressed_tokens} tokens")
        
        if section.compression_needed:
            compression_ratio = (original_tokens - compressed_tokens) / original_tokens
            print(f"    压缩率: {compression_ratio:.1%}")
            print(f"    预算符合: {'✅' if compressed_tokens <= allocated else '❌'} {compressed_tokens - allocated:+d}")
        else:
            print(f"    状态: 无需压缩")
        print()
    
    print(f"整体统计:")
    print(f"  原始总量: {total_original} tokens")
    print(f"  预算总量: {total_budget} tokens") 
    print(f"  压缩总量: {total_compressed} tokens")
    print(f"  整体压缩率: {(total_original - total_compressed) / total_original:.1%}")
    print(f"  预算符合: {'✅' if total_compressed <= total_budget else '❌'} {total_compressed - total_budget:+d}")
    
    if total_compressed <= total_budget:
        print("✅ 成功：精确压缩完全符合预算分配！")
        print("✅ 改进：每个section都精确压缩到其allocated_tokens目标")
    else:
        print("❌ 问题：部分section压缩后仍超出预算")
    
    # 对比旧的50%压缩会是什么结果
    print(f"\n对比：如果采用旧的50%压缩逻辑:")
    old_total = 0
    for name, content in content_sections.items():
        original = tokenizer.count_tokens(content)
        old_target = max(1, original // 2)
        old_total += old_target
    
    print(f"  旧50%总量: {old_total} tokens")
    print(f"  新精确总量: {total_compressed} tokens")
    print(f"  差异: {total_compressed - old_total:+d} tokens")
    
    if total_compressed < old_total:
        print("✅ 新逻辑更节省空间")
    elif total_compressed > old_total:
        print("✅ 新逻辑保留更多内容")
    else:
        print("= 两者结果相同")

if __name__ == "__main__":
    test_reasonable_compression()