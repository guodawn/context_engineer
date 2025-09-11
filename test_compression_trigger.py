#!/usr/bin/env python3
"""
专门测试触发压缩的正确场景
"""

from context_engineer.core.budget_manager import BudgetManager, CompatBucketConfig as BucketConfig
from context_engineer.core.context_assembler import ContextAssembler
from context_engineer.core.tokenizer_service import TokenizerService

def test_compression_trigger():
    print("=== 测试正确触发压缩的场景 ===\n")
    
    tokenizer = TokenizerService()
    bm = BudgetManager(tokenizer)
    
    # 关键配置：让某些桶的实际内容超过其max_tokens，强制触发溢出
    buckets = {
        "system_safety": BucketConfig("system_safety", min_tokens=20, max_tokens=50, weight=1.0, 
                                    sticky=True, content_score=0.9),
        "task_instructions": BucketConfig("task_instructions", min_tokens=15, max_tokens=40, weight=0.8, 
                                         sticky=True, content_score=0.85),
        "history": BucketConfig("history", min_tokens=30, max_tokens=80, weight=0.6, 
                               droppable=True, content_score=0.7),
        "tools_schema": BucketConfig("tools_schema", min_tokens=25, max_tokens=60, weight=0.9,
                                    content_score=0.8),
    }
    
    for bucket in buckets.values():
        bm.add_bucket(bucket)
    
    bm.set_drop_order(["history", "tools_schema", "task_instructions", "system_safety"])
    
    print("步骤1：创建超过max_tokens的内容")
    # 创建内容，故意超过max_tokens限制
    large_content = {
        "system_safety": "This is system safety content that exceeds the maximum token limit. " * 8,
        "task_instructions": "Detailed task instructions that are longer than the allowed maximum. " * 10,
        "history": "Conversation history with many exchanges that goes beyond the token limit. " * 12,
        "tools_schema": "Tool schemas and API definitions that exceed the maximum allowed tokens. " * 9,
    }
    
    print("内容token计数和限制对比:")
    for name, content in large_content.items():
        actual_tokens = tokenizer.count_tokens(content)
        max_tokens = buckets[name].max_tokens
        over_limit = actual_tokens > max_tokens
        print(f"  {name}: {actual_tokens} tokens (max: {max_tokens}) {'⚠️ 超出' if over_limit else '✅ 符合'}")
    
    # 计算最小需求总量
    min_required = bm.get_total_min_tokens()
    max_possible = bm.get_total_max_tokens()
    
    print(f"\n桶配置分析:")
    print(f"  最小需求总量: {min_required} tokens")
    print(f"  最大可能总量: {max_possible} tokens")
    
    print(f"\n步骤2：设置预算策略")
    
    # 策略A：让预算小于最大需求，但大于最小需求
    # 这样会触发优化分配，但不一定触发溢出压缩
    model_limit_a = min_required + 100  # 比最小需求多100
    output_budget_a = 30
    
    print(f"策略A - 优化分配 (模型: {model_limit_a}, 输出: {output_budget_a}):")
    budget_a = bm.allocate_budget(
        model_context_limit=model_limit_a,
        output_budget=output_budget_a,
        content_scores={"system_safety": 0.95, "task_instructions": 0.90, "history": 0.70, "tools_schema": 0.85}
    )
    
    total_a = sum(alloc.allocated_tokens for alloc in budget_a)
    print(f"  总分配: {total_a} tokens")
    for alloc in budget_a:
        compression_marker = " (需要压缩)" if alloc.compression_needed else ""
        print(f"    {alloc.bucket_name}: {alloc.allocated_tokens} tokens{compression_marker}")
    
    print(f"\n策略B - 强制溢出 (让预算小于内容需求)")
    
    # 首先让预算足够大，获得初始分配
    initial_budget = bm.allocate_budget(
        model_context_limit=1000,
        output_budget=30,
        content_scores={"system_safety": 0.95, "task_instructions": 0.90, "history": 0.70, "tools_schema": 0.85}
    )
    
    print("初始分配 (充足预算):")
    for alloc in initial_budget:
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens")
    
    # 然后人为制造溢出情况：模拟内容超过分配
    print(f"\n人为制造溢出场景:")
    
    # 手动修改某些allocation，让它们需要压缩
    forced_overflow = []
    for alloc in initial_budget:
        new_alloc = type(alloc)(
            bucket_name=alloc.bucket_name,
            allocated_tokens=alloc.allocated_tokens // 3,  # 强制减少到1/3
            priority=alloc.priority,
            compression_needed=True,  # 强制标记为需要压缩
            content_score=alloc.content_score
        )
        forced_overflow.append(new_alloc)
    
    print("强制溢出分配:")
    total_forced = 0
    for alloc in forced_overflow:
        total_forced += alloc.allocated_tokens
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens (强制压缩)")
    print(f"  强制总分配: {total_forced} tokens")
    
    print(f"\n步骤3：测试精确压缩")
    
    assembler = ContextAssembler(tokenizer)
    
    # 使用强制溢出分配测试压缩
    assembled_context = assembler.assemble_context(
        content_sections=large_content,
        budget_allocations=forced_overflow
    )
    
    print("精确压缩结果:")
    total_original = 0
    total_compressed = 0
    total_budget = 0
    
    for section in assembled_context.sections:
        original_tokens = tokenizer.count_tokens(large_content[section.name])
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
        print(f"    内容预览: {section.content[:80]}...")
        print()
    
    print(f"整体统计:")
    print(f"  原始总量: {total_original} tokens")
    print(f"  预算总量: {total_budget} tokens")
    print(f"  压缩总量: {total_compressed} tokens")
    print(f"  整体压缩率: {(total_original - total_compressed) / total_original:.1%}")
    print(f"  预算符合: {'✅' if total_compressed <= total_budget else '❌'} {total_compressed - total_budget:+d}")
    
    if total_compressed <= total_budget:
        print("✅ 成功：精确压缩完全符合预算分配！")
        print("✅ 每个section都精确压缩到其allocated_tokens目标")
    else:
        print("❌ 问题：压缩结果仍超出预算")
    
    # 对比旧的50%压缩
    print(f"\n对比：如果采用旧的50%压缩:")
    old_total = 0
    for name, content in large_content.items():
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
        print("= 两者相同")

if __name__ == "__main__":
    test_compression_trigger()