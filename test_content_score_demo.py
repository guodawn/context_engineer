#!/usr/bin/env python3
"""
演示content_score对预算分配的明显影响
"""

from context_engineer.core.budget_manager import BudgetManager, CompatBucketConfig as BucketConfig

def demonstrate_content_score_impact():
    print("=== 演示content_score对分配的明显影响 ===\n")
    
    # 创建预算管理器
    bm = BudgetManager()
    
    # 配置桶，使用较大的max_tokens留下优化空间
    buckets = {
        "system_safety": BucketConfig("system_safety", min_tokens=100, max_tokens=400, weight=1.0, 
                                    sticky=True, content_score=0.9),
        "task_instructions": BucketConfig("task_instructions", min_tokens=50, max_tokens=300, weight=0.8, 
                                         sticky=True, content_score=0.8),
        "tools_schema": BucketConfig("tools_schema", min_tokens=150, max_tokens=600, weight=0.9,
                                    content_score=0.85),
        "history": BucketConfig("history", min_tokens=100, max_tokens=800, weight=0.6, 
                               droppable=True, content_score=0.7),
        "memory": BucketConfig("memory", min_tokens=120, max_tokens=500, weight=0.7,
                              content_score=0.75),
        "rag_evidence": BucketConfig("rag_evidence", min_tokens=80, max_tokens=400, weight=0.5, 
                                   droppable=True, content_score=0.6),
        "few_shot_examples": BucketConfig("few_shot_examples", min_tokens=50, max_tokens=300, weight=0.4, 
                                        droppable=True, content_score=0.5),
        "scratchpad": BucketConfig("scratchpad", min_tokens=30, max_tokens=200, weight=0.3, 
                                  droppable=True, content_score=0.4)
    }
    
    # 添加桶配置
    for bucket in buckets.values():
        bm.add_bucket(bucket)
    
    bm.set_drop_order(["scratchpad", "few_shot_examples", "rag_evidence", "history"])
    
    print("场景：设置较大的上下文窗口，让优化阶段有充足的预算空间")
    model_limit = 8000
    output_budget = 1000
    available_budget = model_limit - output_budget - 200
    
    print(f"模型限制: {model_limit}")
    print(f"输出预算: {output_budget}")
    print(f"系统开销: 200")
    print(f"可用预算: {available_budget}")
    print(f"最小需求总量: {bm.get_total_min_tokens()}")
    print(f"最大需求总量: {bm.get_total_max_tokens()}")
    print()
    
    print("=== 对比1：高content_score vs 低content_score ===")
    
    # 高content_score配置
    high_scores = {
        "system_safety": 0.95,
        "task_instructions": 0.93,
        "tools_schema": 0.90,
        "history": 0.88,
        "memory": 0.85,
        "rag_evidence": 0.83,
        "few_shot_examples": 0.80,
        "scratchpad": 0.78
    }
    
    # 低content_score配置
    low_scores = {
        "system_safety": 0.25,
        "task_instructions": 0.23,
        "tools_schema": 0.20,
        "history": 0.18,
        "memory": 0.15,
        "rag_evidence": 0.13,
        "few_shot_examples": 0.10,
        "scratchpad": 0.08
    }
    
    allocations_high = bm.allocate_budget(model_limit, output_budget, high_scores)
    allocations_low = bm.allocate_budget(model_limit, output_budget, low_scores)
    
    print("高content_score配置 (重视所有内容):")
    total_high = sum(alloc.allocated_tokens for alloc in allocations_high)
    print(f"总分配: {total_high}")
    for alloc in allocations_high:
        bucket_config = bm.get_bucket_config(alloc.bucket_name)
        score = high_scores.get(alloc.bucket_name, "N/A")
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens (score: {score})")
    
    print("\n低content_score配置 (轻视所有内容):")
    total_low = sum(alloc.allocated_tokens for alloc in allocations_low)
    print(f"总分配: {total_low}")
    for alloc in allocations_low:
        bucket_config = bm.get_bucket_config(alloc.bucket_name)
        score = low_scores.get(alloc.bucket_name, "N/A")
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens (score: {score})")
    
    print(f"\n差异对比:")
    for i, (alloc_high, alloc_low) in enumerate(zip(allocations_high, allocations_low)):
        diff = alloc_high.allocated_tokens - alloc_low.allocated_tokens
        print(f"  {alloc_high.bucket_name}: +{diff} tokens (高score vs 低score)")
    
    print("\n=== 对比2：使用BucketConfig默认值 vs 自定义值 ===")
    
    # 使用BucketConfig中的默认值
    allocations_default = bm.allocate_budget(model_limit, output_budget)
    
    # 自定义content_scores，重点突出某些桶
    custom_scores = {
        "system_safety": 0.99,      # 极高
        "tools_schema": 0.98,       # 极高  
        "rag_evidence": 0.97,       # 极高
        "task_instructions": 0.4,   # 较低（与默认值0.8对比）
        "history": 0.35,            # 较低（与默认值0.7对比）
        "memory": 0.3,              # 较低（与默认值0.75对比）
        "few_shot_examples": 0.25,  # 很低
        "scratchpad": 0.2           # 很低
    }
    
    allocations_custom = bm.allocate_budget(model_limit, output_budget, custom_scores)
    
    print("使用BucketConfig默认值:")
    total_default = sum(alloc.allocated_tokens for alloc in allocations_default)
    print(f"总分配: {total_default}")
    for alloc in allocations_default:
        bucket_config = bm.get_bucket_config(alloc.bucket_name)
        default_score = bucket_config.content_score if bucket_config else "N/A"
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens (默认score: {default_score})")
    
    print("\n自定义content_scores（突出system_safety, tools_schema, rag_evidence）:")
    total_custom = sum(alloc.allocated_tokens for alloc in allocations_custom)
    print(f"总分配: {total_custom}")
    for alloc in allocations_custom:
        bucket_config = bm.get_bucket_config(alloc.bucket_name)
        custom_score = custom_scores.get(alloc.bucket_name, "N/A")
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens (自定义score: {custom_score})")
    
    print(f"\n重点桶增益分析:")
    key_buckets = ["system_safety", "tools_schema", "rag_evidence"]
    for bucket_name in key_buckets:
        default_alloc = next(a for a in allocations_default if a.bucket_name == bucket_name)
        custom_alloc = next(a for a in allocations_custom if a.bucket_name == bucket_name)
        gain = custom_alloc.allocated_tokens - default_alloc.allocated_tokens
        print(f"  {bucket_name}: +{gain} tokens")

if __name__ == "__main__":
    demonstrate_content_score_impact()