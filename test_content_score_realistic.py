#!/usr/bin/env python3
"""
在更现实的场景下演示content_score的影响
"""

from context_engineer.core.budget_manager import BudgetManager, CompatBucketConfig as BucketConfig

def test_realistic_content_score():
    print("=== 现实场景下的content_score影响测试 ===\n")
    
    bm = BudgetManager()
    
    # 配置更现实的参数，让预算处于紧张但不极端的状态
    buckets = {
        "system_safety": BucketConfig("system_safety", min_tokens=150, max_tokens=350, weight=1.2, 
                                    sticky=True, content_score=0.92),
        "task_instructions": BucketConfig("task_instructions", min_tokens=80, max_tokens=250, weight=1.0, 
                                         sticky=True, content_score=0.88),
        "tools_schema": BucketConfig("tools_schema", min_tokens=200, max_tokens=500, weight=1.1,
                                    content_score=0.85),
        "history": BucketConfig("history", min_tokens=120, max_tokens=400, weight=0.8, 
                               droppable=True, content_score=0.70),
        "memory": BucketConfig("memory", min_tokens=100, max_tokens=350, weight=0.9,
                              content_score=0.75),
        "rag_evidence": BucketConfig("rag_evidence", min_tokens=80, max_tokens=300, weight=0.7, 
                                   droppable=True, content_score=0.65),
        "few_shot_examples": BucketConfig("few_shot_examples", min_tokens=60, max_tokens=250, weight=0.6, 
                                        droppable=True, content_score=0.55),
        "scratchpad": BucketConfig("scratchpad", min_tokens=40, max_tokens=200, weight=0.5, 
                                  droppable=True, content_score=0.45)
    }
    
    for bucket in buckets.values():
        bm.add_bucket(bucket)
    
    bm.set_drop_order(["scratchpad", "few_shot_examples", "rag_evidence", "history"])
    
    # 设置预算，让初始分配满足最小需求，但优化阶段有竞争
    min_required = bm.get_total_min_tokens()  # 830
    max_possible = bm.get_total_max_tokens()  # 2600
    
    # 设置预算在最小需求和最大需求之间
    model_limit = min_required + 800  # 1630 total, 1330 input budget
    output_budget = 200
    system_overhead = 200
    available_budget = model_limit - output_budget - system_overhead
    
    print(f"预算配置:")
    print(f"  模型限制: {model_limit}")
    print(f"  输出预算: {output_budget}")
    print(f"  系统开销: {system_overhead}")
    print(f"  可用输入预算: {available_budget}")
    print(f"  最小需求: {min_required}")
    print(f"  最大需求: {max_possible}")
    print(f"  预算比例: {available_budget/max_possible:.1%}")
    print()
    
    print("=== 对比：高content_score vs 低content_score ===")
    
    # 高content_score - 重视所有内容
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
    
    # 低content_score - 轻视所有内容  
    low_scores = {
        "system_safety": 0.30,
        "task_instructions": 0.28,
        "tools_schema": 0.25,
        "history": 0.23,
        "memory": 0.20,
        "rag_evidence": 0.18,
        "few_shot_examples": 0.15,
        "scratchpad": 0.13
    }
    
    allocations_high = bm.allocate_budget(model_limit, output_budget, high_scores)
    allocations_low = bm.allocate_budget(model_limit, output_budget, low_scores)
    
    print("高content_score配置 (重视内容):")
    total_high = sum(alloc.allocated_tokens for alloc in allocations_high)
    initial_total = sum(buckets[name].min_tokens for name in buckets.keys())
    optimization_total = total_high - initial_total
    print(f"总分配: {total_high} (初始: {initial_total}, 优化: {optimization_total})")
    for alloc in allocations_high:
        score = high_scores.get(alloc.bucket_name, "N/A")
        min_tok = buckets[alloc.bucket_name].min_tokens
        extra = alloc.allocated_tokens - min_tok
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens (score: {score}, 额外: +{extra})")
    
    print(f"\n低content_score配置 (轻视内容):")
    total_low = sum(alloc.allocated_tokens for alloc in allocations_low)
    optimization_total_low = total_low - initial_total
    print(f"总分配: {total_low} (初始: {initial_total}, 优化: {optimization_total_low})")
    for alloc in allocations_low:
        score = low_scores.get(alloc.bucket_name, "N/A")
        min_tok = buckets[alloc.bucket_name].min_tokens
        extra = alloc.allocated_tokens - min_tok
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens (score: {score}, 额外: +{extra})")
    
    print(f"\n优化阶段差异对比:")
    print(f"高content_score优化分配: {optimization_total} tokens")
    print(f"低content_score优化分配: {optimization_total_low} tokens")
    print(f"差异: {optimization_total - optimization_total_low} tokens")
    
    for alloc_high, alloc_low in zip(allocations_high, allocations_low):
        high_extra = alloc_high.allocated_tokens - buckets[alloc_high.bucket_name].min_tokens
        low_extra = alloc_low.allocated_tokens - buckets[alloc_low.bucket_name].min_tokens
        diff = high_extra - low_extra
        if diff != 0:
            print(f"  {alloc_high.bucket_name}: +{diff} tokens (高score vs 低score)")
    
    print(f"\n=== 使用BucketConfig默认值 vs 极端自定义 ===")
    
    # 使用BucketConfig中的默认值
    allocations_default = bm.allocate_budget(model_limit, output_budget)
    
    # 极端自定义：只重视前三个桶，轻视其他桶
    extreme_scores = {
        "system_safety": 0.99,      # 极高
        "task_instructions": 0.98,  # 极高
        "tools_schema": 0.97,       # 极高
        "history": 0.20,            # 很低（对比默认值0.70）
        "memory": 0.15,             # 很低（对比默认值0.75）
        "rag_evidence": 0.10,       # 极低（对比默认值0.65）
        "few_shot_examples": 0.05,  # 极低（对比默认值0.55）
        "scratchpad": 0.02          # 极低（对比默认值0.45）
    }
    
    allocations_extreme = bm.allocate_budget(model_limit, output_budget, extreme_scores)
    
    print("使用BucketConfig默认值:")
    total_default = sum(alloc.allocated_tokens for alloc in allocations_default)
    opt_default = total_default - initial_total
    print(f"总分配: {total_default} (优化: {opt_default})")
    for alloc in allocations_default:
        bucket_config = bm.get_bucket_config(alloc.bucket_name)
        default_score = bucket_config.content_score if bucket_config else "N/A"
        min_tok = buckets[alloc.bucket_name].min_tokens
        extra = alloc.allocated_tokens - min_tok
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens (默认score: {default_score}, 额外: +{extra})")
    
    print(f"\n极端自定义（重视核心，轻视其他）:")
    total_extreme = sum(alloc.allocated_tokens for alloc in allocations_extreme)
    opt_extreme = total_extreme - initial_total
    print(f"总分配: {total_extreme} (优化: {opt_extreme})")
    for alloc in allocations_extreme:
        bucket_config = bm.get_bucket_config(alloc.bucket_name)
        extreme_score = extreme_scores.get(alloc.bucket_name, "N/A")
        min_tok = buckets[alloc.bucket_name].min_tokens
        extra = alloc.allocated_tokens - min_tok
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens (自定义score: {extreme_score}, 额外: +{extra})")
    
    print(f"\n极端配置的影响:")
    key_buckets = ["system_safety", "task_instructions", "tools_schema"]
    for bucket_name in key_buckets:
        default_alloc = next(a for a in allocations_default if a.bucket_name == bucket_name)
        extreme_alloc = next(a for a in allocations_extreme if a.bucket_name == bucket_name)
        default_extra = default_alloc.allocated_tokens - buckets[bucket_name].min_tokens
        extreme_extra = extreme_alloc.allocated_tokens - buckets[bucket_name].min_tokens
        gain = extreme_extra - default_extra
        print(f"  {bucket_name}: +{gain} tokens 额外分配")

if __name__ == "__main__":
    test_realistic_content_score()