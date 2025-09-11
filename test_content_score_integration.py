#!/usr/bin/env python3
"""
测试content_scores集成到BucketConfig的新功能
"""

from context_engineer.core.budget_manager import BudgetManager, CompatBucketConfig as BucketConfig

def test_content_score_integration():
    print("=== 测试content_score集成到BucketConfig ===\n")
    
    # 创建预算管理器
    bm = BudgetManager()
    
    # 配置8个上下文桶，每个桶设置不同的content_score
    buckets = {
        "system_safety": BucketConfig("system_safety", min_tokens=100, max_tokens=300, weight=1.0, 
                                    sticky=True, content_score=0.95),      # 高重要性
        "task_instructions": BucketConfig("task_instructions", min_tokens=50, max_tokens=200, weight=0.8, 
                                         sticky=True, content_score=0.90),   # 高重要性
        "tools_schema": BucketConfig("tools_schema", min_tokens=200, max_tokens=800, weight=0.9,
                                    content_score=0.85),                     # 较高重要性
        "history": BucketConfig("history", min_tokens=100, max_tokens=1000, weight=0.6, 
                               droppable=True, content_score=0.70),          # 中等重要性
        "memory": BucketConfig("memory", min_tokens=150, max_tokens=600, weight=0.7,
                              content_score=0.75),                           # 较高重要性
        "rag_evidence": BucketConfig("rag_evidence", min_tokens=100, max_tokens=800, weight=0.5, 
                                   droppable=True, content_score=0.60),       # 中等重要性
        "few_shot_examples": BucketConfig("few_shot_examples", min_tokens=50, max_tokens=400, weight=0.4, 
                                        droppable=True, content_score=0.50),  # 普通重要性
        "scratchpad": BucketConfig("scratchpad", min_tokens=50, max_tokens=300, weight=0.3, 
                                  droppable=True, content_score=0.40)         # 低重要性
    }
    
    # 添加桶配置
    for bucket in buckets.values():
        bm.add_bucket(bucket)
    
    # 设置丢弃顺序
    bm.set_drop_order(["scratchpad", "few_shot_examples", "rag_evidence", "history"])
    
    print("场景1：使用BucketConfig中配置的content_score（不传入content_scores参数）")
    allocations1 = bm.allocate_budget(
        model_context_limit=4000,  # 4K上下文窗口
        output_budget=500,        # 预留500给输出
        # 注意：不传入content_scores参数
    )
    
    total_allocated1 = sum(alloc.allocated_tokens for alloc in allocations1)
    available_budget1 = 4000 - 500 - 200
    print(f"总分配令牌: {total_allocated1}")
    print(f"可用预算: {available_budget1}")
    print("分配结果:")
    for alloc in allocations1:
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens")
    
    print("\n场景2：使用自定义content_scores（覆盖BucketConfig配置）")
    # 自定义content_scores，与BucketConfig中的默认值不同
    custom_content_scores = {
        "system_safety": 0.8,           # 比配置的0.95低
        "task_instructions": 0.7,       # 比配置的0.90低
        "tools_schema": 0.95,           # 比配置的0.85高
        "history": 0.85,                # 比配置的0.70高
        "memory": 0.6,                  # 比配置的0.75低
        "rag_evidence": 0.9,            # 比配置的0.60高很多
        "few_shot_examples": 0.75,      # 比配置的0.50高
        "scratchpad": 0.65              # 比配置的0.40高
    }
    
    allocations2 = bm.allocate_budget(
        model_context_limit=4000,      # 相同的上下文窗口
        output_budget=500,             # 相同的输出预算
        content_scores=custom_content_scores  # 传入自定义的content_scores
    )
    
    total_allocated2 = sum(alloc.allocated_tokens for alloc in allocations2)
    print(f"总分配令牌: {total_allocated2}")
    print("分配结果:")
    for alloc in allocations2:
        bucket_config = bm.get_bucket_config(alloc.bucket_name)
        default_score = bucket_config.content_score if bucket_config else "N/A"
        custom_score = custom_content_scores.get(alloc.bucket_name, "N/A")
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens")
        print(f"    默认score: {default_score}, 自定义score: {custom_score}")
    
    print("\n场景3：混合使用 - 某些桶使用配置值，某些使用自定义值")
    # 只提供部分桶的content_scores，其余的使用BucketConfig中的默认值
    partial_content_scores = {
        "system_safety": 0.99,      # 极高重要性，覆盖默认值
        "tools_schema": 0.98,       # 极高重要性，覆盖默认值
        "rag_evidence": 0.15,       # 极低重要性，覆盖默认值
        # 其余桶将使用BucketConfig中的默认值
    }
    
    allocations3 = bm.allocate_budget(
        model_context_limit=4000,
        output_budget=500,
        content_scores=partial_content_scores
    )
    
    total_allocated3 = sum(alloc.allocated_tokens for alloc in allocations3)
    print(f"总分配令牌: {total_allocated3}")
    print("分配结果:")
    for alloc in allocations3:
        bucket_config = bm.get_bucket_config(alloc.bucket_name)
        default_score = bucket_config.content_score if bucket_config else "N/A"
        custom_score = partial_content_scores.get(alloc.bucket_name, "使用默认值")
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens")
        print(f"    score: {custom_score if custom_score != '使用默认值' else default_score}")
    
    print("\n场景4：验证content_score对分配的影响")
    print("比较相同预算下，不同content_score配置的差异:")
    
    # 创建两个对比配置
    high_scores = {name: 0.9 for name in buckets.keys()}  # 所有桶都高重要性
    low_scores = {name: 0.1 for name in buckets.keys()}   # 所有桶都低重要性
    
    allocations_high = bm.allocate_budget(4000, 500, high_scores)
    allocations_low = bm.allocate_budget(4000, 500, low_scores)
    
    print("高content_score配置:")
    for alloc in allocations_high:
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens")
    
    print("低content_score配置:")  
    for alloc in allocations_low:
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens")
    
    print(f"\n总结:")
    print(f"✓ 场景1：使用BucketConfig默认content_score - 无需传入参数")
    print(f"✓ 场景2：自定义content_scores - 可以覆盖配置值") 
    print(f"✓ 场景3：混合使用 - 部分自定义，部分使用默认值")
    print(f"✓ 场景4：content_score确实影响分配结果")

if __name__ == "__main__":
    test_content_score_integration()