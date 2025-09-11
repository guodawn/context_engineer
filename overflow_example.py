#!/usr/bin/env python3
"""
具体例子：演示 _handle_overflow 的触发情况
"""

from context_engineer.core.budget_manager import BudgetManager, CompatBucketConfig as BucketConfig

def demonstrate_overflow():
    # 创建预算管理器
    bm = BudgetManager()
    
    # 配置8个上下文桶（模拟实际系统）
    # 新增：为每个桶配置默认的content_score
    buckets = {
        "system_safety": BucketConfig("system_safety", min_tokens=100, max_tokens=300, weight=1.0, 
                                    sticky=True, content_score=0.9),
        "task_instructions": BucketConfig("task_instructions", min_tokens=50, max_tokens=200, weight=0.8, 
                                         sticky=True, content_score=0.85),
        "tools_schema": BucketConfig("tools_schema", min_tokens=200, max_tokens=800, weight=0.9,
                                    content_score=0.8),
        "history": BucketConfig("history", min_tokens=100, max_tokens=1000, weight=0.6, 
                               droppable=True, content_score=0.7),
        "memory": BucketConfig("memory", min_tokens=150, max_tokens=600, weight=0.7,
                              content_score=0.75),
        "rag_evidence": BucketConfig("rag_evidence", min_tokens=100, max_tokens=800, weight=0.5, 
                                   droppable=True, content_score=0.6),
        "few_shot_examples": BucketConfig("few_shot_examples", min_tokens=50, max_tokens=400, weight=0.4, 
                                        droppable=True, content_score=0.5),
        "scratchpad": BucketConfig("scratchpad", min_tokens=50, max_tokens=300, weight=0.3, 
                                  droppable=True, content_score=0.4)
    }
    
    # 添加桶配置
    for bucket in buckets.values():
        bm.add_bucket(bucket)
    
    # 设置丢弃顺序：从最不重要的开始
    bm.set_drop_order(["scratchpad", "few_shot_examples", "rag_evidence", "history"])
    
    # 场景1：正常情况 - 上下文窗口足够大
    print("=== 场景1：正常情况 ===")
    content_scores = {
        "system_safety": 0.9,
        "task_instructions": 0.8,
        "tools_schema": 0.85,
        "history": 0.6,
        "memory": 0.7,
        "rag_evidence": 0.5,
        "few_shot_examples": 0.4,
        "scratchpad": 0.3
    }
    
    allocations = bm.allocate_budget(
        model_context_limit=8000,  # 8K上下文窗口
        output_budget=1000,        # 预留1K给输出
        content_scores=content_scores
    )
    
    total_allocated = sum(alloc.allocated_tokens for alloc in allocations)
    print(f"总分配令牌: {total_allocated}")
    print(f"可用预算: {8000 - 1000 - 200}")
    print("分配结果:")
    for alloc in allocations:
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens")
    
    print("\n=== 场景2：触发溢出 ===")
    # 场景2：触发溢出 - 上下文窗口很小
    allocations_overflow = bm.allocate_budget(
        model_context_limit=2000,  # 只有2K上下文窗口
        output_budget=500,         # 预留500给输出
        content_scores=content_scores
    )
    
    total_allocated_overflow = sum(alloc.allocated_tokens for alloc in allocations_overflow)
    available_budget = 2000 - 500 - 200
    print(f"总分配令牌: {total_allocated_overflow}")
    print(f"可用预算: {available_budget}")
    print(f"溢出量: {total_allocated_overflow - available_budget}")
    print("分配结果:")
    for alloc in allocations_overflow:
        compression_marker = " (需要压缩)" if alloc.compression_needed else ""
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens{compression_marker}")
    
    print("\n=== 场景3：极端情况 - 预算严重不足 ===")
    # 场景3：极端情况 - 预算连最小需求都满足不了
    allocations_extreme = bm.allocate_budget(
        model_context_limit=1000,  # 只有1K上下文窗口
        output_budget=300,         # 预留300给输出
        content_scores=content_scores
    )
    
    total_allocated_extreme = sum(alloc.allocated_tokens for alloc in allocations_extreme)
    available_extreme = 1000 - 300 - 200
    print(f"总分配令牌: {total_allocated_extreme}")
    print(f"可用预算: {available_extreme}")
    print(f"最小需求总量: {bm.get_total_min_tokens()}")
    print("分配结果:")
    for alloc in allocations_extreme:
        compression_marker = " (需要压缩)" if alloc.compression_needed else ""
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens{compression_marker}")

    print("\n=== 场景4：真正触发溢出处理 ===")
    print("策略：预算刚好满足最小需求，但内容评分优化会导致超出预算")
    
    # 关键策略：设置预算刚好等于最小需求总量
    min_required = bm.get_total_min_tokens()  # 800
    model_limit = min_required + 300 + 200 + 50  # 最小需求 + 输出 + 系统开销 + 少量余量 = 1350
    
    # 高内容评分：会促使优化阶段增加分配
    high_content_scores = {
        "system_safety": 0.95,      # 极高评分
        "task_instructions": 0.9,   # 极高评分  
        "tools_schema": 0.85,       # 高评分
        "history": 0.8,             # 较高评分
        "memory": 0.75,             # 较高评分
        "rag_evidence": 0.7,        # 中等评分
        "few_shot_examples": 0.65,  # 中等评分
        "scratchpad": 0.6           # 中等评分
    }
    
    allocations_real_overflow = bm.allocate_budget(
        model_context_limit=model_limit,  # 1350 tokens
        output_budget=300,                # 预留300给输出
        content_scores=high_content_scores
    )
    
    total_allocated_real = sum(alloc.allocated_tokens for alloc in allocations_real_overflow)
    available_budget_real = model_limit - 300 - 200
    print(f"模型限制: {model_limit}")
    print(f"输出预算: 300")
    print(f"系统开销: 200") 
    print(f"可用预算: {available_budget_real}")
    print(f"最小需求: {min_required}")
    print(f"总分配: {total_allocated_real}")
    print(f"溢出量: {total_allocated_real - available_budget_real}")
    print("分配结果:")
    for alloc in allocations_real_overflow:
        compression_marker = " (需要压缩)" if alloc.compression_needed else ""
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens{compression_marker}")

    print("\n=== 场景5：极端高评分触发强制溢出 ===")
    print("策略：极高的内容评分导致优化阶段大量超分")
    
    # 更极端的场景：预算略高于最小需求，但内容评分极高
    model_limit2 = min_required + 400  # 1200 (比最小需求多400)
    
    # 极高内容评分
    extreme_scores = {
        "system_safety": 1.0,       # 满分
        "task_instructions": 0.98,  # 接近满分
        "tools_schema": 0.95,       # 极高
        "history": 0.92,            # 极高
        "memory": 0.88,             # 很高
        "rag_evidence": 0.85,       # 很高
        "few_shot_examples": 0.8,   # 高
        "scratchpad": 0.75          # 较高
    }
    
    allocations_extreme_overflow = bm.allocate_budget(
        model_context_limit=model_limit2,  # 1200 tokens
        output_budget=200,                  # 较少的输出预算
        content_scores=extreme_scores
    )
    
    total_extreme = sum(alloc.allocated_tokens for alloc in allocations_extreme_overflow)
    available_extreme2 = model_limit2 - 200 - 200
    print(f"模型限制: {model_limit2}")
    print(f"输出预算: 200")
    print(f"系统开销: 200")
    print(f"可用预算: {available_extreme2}")
    print(f"总分配: {total_extreme}")
    print(f"溢出量: {total_extreme - available_extreme2}")
    print("分配结果:")
    overflow_triggered = False
    for alloc in allocations_extreme_overflow:
        compression_marker = " (需要压缩)" if alloc.compression_needed else ""
        if alloc.compression_needed:
            overflow_triggered = True
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens{compression_marker}")
    
    if overflow_triggered:
        print("✓ 溢出处理已触发！")
    else:
        print("✗ 溢出处理未触发")

    print("\n=== 场景6：强制触发溢出 - 手动制造条件 ===")
    print("策略：修改桶配置，让初始分配+优化分配必然超出预算")
    
    # 创建新的预算管理器，配置特殊的桶参数
    bm2 = BudgetManager()
    
    # 关键策略：设置较大的最小值，但相对较小的最大值
    # 这样初始分配会接近预算上限，优化阶段容易超出
    special_buckets = {
        "system_safety": BucketConfig("system_safety", min_tokens=200, max_tokens=400, weight=2.0, sticky=True),  # 高权重
        "task_instructions": BucketConfig("task_instructions", min_tokens=150, max_tokens=300, weight=1.8, sticky=True),
        "tools_schema": BucketConfig("tools_schema", min_tokens=250, max_tokens=500, weight=1.5),
        "history": BucketConfig("history", min_tokens=180, max_tokens=600, weight=1.2, droppable=True),
        "memory": BucketConfig("memory", min_tokens=200, max_tokens=450, weight=1.0),
        "rag_evidence": BucketConfig("rag_evidence", min_tokens=150, max_tokens=400, weight=0.8, droppable=True),
        "few_shot_examples": BucketConfig("few_shot_examples", min_tokens=100, max_tokens=350, weight=0.6, droppable=True),
        "scratchpad": BucketConfig("scratchpad", min_tokens=80, max_tokens=250, weight=0.5, droppable=True)
    }
    
    for bucket in special_buckets.values():
        bm2.add_bucket(bucket)
    
    bm2.set_drop_order(["scratchpad", "few_shot_examples", "rag_evidence", "history"])
    
    # 设置预算：初始分配刚好满足，但优化时会超出
    min_required2 = bm2.get_total_min_tokens()
    tight_budget = min_required2 + 250  # 只比最小需求多250
    
    # 极高的内容评分，会强烈推动优化超分
    force_overflow_scores = {
        "system_safety": 0.99,
        "task_instructions": 0.97,
        "tools_schema": 0.95,
        "history": 0.93,
        "memory": 0.91,
        "rag_evidence": 0.89,
        "few_shot_examples": 0.87,
        "scratchpad": 0.85
    }
    
    print(f"特殊配置 - 最小需求: {min_required2}")
    print(f"紧预算: {tight_budget}")
    print(f"可用输入预算: {tight_budget - 200}")  # 减去系统开销
    
    allocations_force = bm2.allocate_budget(
        model_context_limit=tight_budget,
        output_budget=200,  # 较少的输出预算
        content_scores=force_overflow_scores
    )
    
    total_force = sum(alloc.allocated_tokens for alloc in allocations_force)
    available_force = tight_budget - 200 - 200
    print(f"总分配: {total_force}")
    print(f"可用预算: {available_force}")
    print(f"差值: {total_force - available_force}")
    
    overflow_final = False
    print("分配结果:")
    for alloc in allocations_force:
        compression_marker = " (需要压缩)" if alloc.compression_needed else ""
        if alloc.compression_needed:
            overflow_final = True
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens{compression_marker}")
    
    if overflow_final:
        print("🎉 成功触发溢出处理！")
    else:
        print("😔 仍未触发溢出处理")
        
    print("\n=== 场景7：终极溢出触发 - 直接修改逻辑验证 ===")
    print("策略：创建一个场景，确保优化阶段会超出预算")
    
    # 让我们手动计算一个必然触发的情况
    # 初始分配：最小需求 = 1310
    # 可用预算：1400  
    # 剩余预算：90
    # 优化阶段：高评分内容会尽可能多拿这90，但我们要让它超出
    
    bm3 = BudgetManager()
    ultimate_buckets = {
        "system_safety": BucketConfig("system_safety", min_tokens=300, max_tokens=800, weight=3.0, sticky=True),  # 超高权重
        "task_instructions": BucketConfig("task_instructions", min_tokens=200, max_tokens=600, weight=2.5, sticky=True),
        "tools_schema": BucketConfig("tools_schema", min_tokens=250, max_tokens=700, weight=2.0),
        "history": BucketConfig("history", min_tokens=180, max_tokens=500, weight=1.5, droppable=True),
        "memory": BucketConfig("memory", min_tokens=200, max_tokens=450, weight=1.2),
        "rag_evidence": BucketConfig("rag_evidence", min_tokens=100, max_tokens=400, weight=1.0, droppable=True),
        "few_shot_examples": BucketConfig("few_shot_examples", min_tokens=60, max_tokens=300, weight=0.8, droppable=True),
        "scratchpad": BucketConfig("scratchpad", min_tokens=50, max_tokens=200, weight=0.5, droppable=True)
    }
    
    for bucket in ultimate_buckets.values():
        bm3.add_bucket(bucket)
    
    bm3.set_drop_order(["scratchpad", "few_shot_examples", "rag_evidence", "history"])
    
    min_req = bm3.get_total_min_tokens()  # 1340
    ultimate_limit = min_req + 300  # 1640 total, 1340 input budget
    
    # 满分评分，确保优化阶段会最大化分配
    perfect_scores = {name: 1.0 for name in ultimate_buckets.keys()}
    
    print(f"终极配置:")
    print(f"  最小需求: {min_req}")
    print(f"  模型限制: {ultimate_limit}")
    print(f"  输出预算: 200")
    print(f"  系统开销: 200") 
    print(f"  输入预算: {ultimate_limit - 400}")
    
    allocations_ultimate = bm3.allocate_budget(
        model_context_limit=ultimate_limit,
        output_budget=200,
        content_scores=perfect_scores
    )
    
    total_ultimate = sum(alloc.allocated_tokens for alloc in allocations_ultimate)
    available_ultimate = ultimate_limit - 200 - 200
    print(f"\n结果:")
    print(f"  总分配: {total_ultimate}")
    print(f"  可用预算: {available_ultimate}")
    print(f"  溢出检测: {'触发' if total_ultimate > available_ultimate else '未触发'}")
    
    ultimate_triggered = False
    for alloc in allocations_ultimate:
        if alloc.compression_needed:
            ultimate_triggered = True
            break
    
    print(f"  压缩标记: {'有' if ultimate_triggered else '无'}")
    if ultimate_triggered:
        print("🏆 终于触发溢出处理了！")

    print("\n=== 场景8：演示新的content_score集成功能 ===")
    print("新增功能：content_score可以直接配置在BucketConfig中")
    
    # 场景8a：使用BucketConfig中配置的默认content_score
    print("\n8a. 使用BucketConfig默认content_score（无需传入参数）:")
    allocations_default = bm.allocate_budget(
        model_context_limit=3000,
        output_budget=400
        # 注意：不传入content_scores参数
    )
    
    total_default = sum(alloc.allocated_tokens for alloc in allocations_default)
    available_default = 3000 - 400 - 200
    print(f"总分配: {total_default}, 可用预算: {available_default}")
    for alloc in allocations_default:
        bucket_config = bm.get_bucket_config(alloc.bucket_name)
        default_score = bucket_config.content_score if bucket_config else "N/A"
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens (默认score: {default_score})")
    
    # 场景8b：自定义content_scores覆盖默认值
    print("\n8b. 自定义content_scores覆盖默认值:")
    custom_scores = {
        "system_safety": 0.99,      # 比默认的0.9更高
        "tools_schema": 0.98,       # 比默认的0.8更高
        "rag_evidence": 0.97,       # 比默认的0.6更高
        "task_instructions": 0.3,   # 比默认的0.85更低
        "history": 0.25,            # 比默认的0.7更低
        "memory": 0.2,              # 比默认的0.75更低
    }
    
    allocations_custom = bm.allocate_budget(
        model_context_limit=3000,
        output_budget=400,
        content_scores=custom_scores
    )
    
    total_custom = sum(alloc.allocated_tokens for alloc in allocations_custom)
    print(f"总分配: {total_custom}, 可用预算: {available_default}")
    for alloc in allocations_custom:
        bucket_config = bm.get_bucket_config(alloc.bucket_name)
        default_score = bucket_config.content_score if bucket_config else "N/A"
        custom_score = custom_scores.get(alloc.bucket_name, "使用默认值")
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens")
        print(f"    默认score: {default_score}, 自定义score: {custom_score}")
    
    print("\n=== 场景9：直接演示溢出处理逻辑 ===")
    print("策略：手动调用_handle_overflow来演示其工作原理")
    
    # 创建一个正常分配，然后手动触发溢出处理
    bm4 = BudgetManager()
    demo_buckets = {
        "system_safety": BucketConfig("system_safety", min_tokens=100, max_tokens=300, weight=1.0, sticky=True),
        "task_instructions": BucketConfig("task_instructions", min_tokens=50, max_tokens=200, weight=0.8, sticky=True),
        "tools_schema": BucketConfig("tools_schema", min_tokens=200, max_tokens=800, weight=0.9),
        "history": BucketConfig("history", min_tokens=100, max_tokens=1000, weight=0.6, droppable=True),
        "memory": BucketConfig("memory", min_tokens=150, max_tokens=600, weight=0.7),
        "rag_evidence": BucketConfig("rag_evidence", min_tokens=100, max_tokens=800, weight=0.5, droppable=True),
        "few_shot_examples": BucketConfig("few_shot_examples", min_tokens=50, max_tokens=400, weight=0.4, droppable=True),
        "scratchpad": BucketConfig("scratchpad", min_tokens=50, max_tokens=300, weight=0.3, droppable=True)
    }
    
    for bucket in demo_buckets.values():
        bm4.add_bucket(bucket)
    
    bm4.set_drop_order(["scratchpad", "few_shot_examples", "rag_evidence", "history"])
    
    # 先进行正常分配
    normal_allocations = bm4.allocate_budget(
        model_context_limit=2000,
        output_budget=300,
        content_scores=content_scores
    )
    
    print("正常分配结果:")
    total_normal = sum(alloc.allocated_tokens for alloc in normal_allocations)
    available_normal = 2000 - 300 - 200
    print(f"总分配: {total_normal}, 可用预算: {available_normal}")
    for alloc in normal_allocations:
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens}")
    
    # 现在手动制造溢出情况：人为增加某些桶的分配
    print(f"\n手动制造溢出情况:")
    overflow_allocations = []
    for alloc in normal_allocations:
        new_alloc = type(alloc)(
            bucket_name=alloc.bucket_name,
            allocated_tokens=alloc.allocated_tokens + 100,  # 每个桶增加100 tokens
            priority=alloc.priority,
            compression_needed=False,
            content_score=alloc.content_score
        )
        overflow_allocations.append(new_alloc)
    
    total_overflow = sum(alloc.allocated_tokens for alloc in overflow_allocations)
    print(f"人为增加后的总分配: {total_overflow}")
    print(f"超出预算: {total_overflow - available_normal}")
    
    # 手动调用溢出处理
    print(f"\n调用_handle_overflow进行溢出处理:")
    handled_allocations = bm4._handle_overflow(overflow_allocations, available_normal)
    
    total_handled = sum(alloc.allocated_tokens for alloc in handled_allocations)
    print(f"处理后的总分配: {total_handled}")
    print(f"处理后的超出量: {total_handled - available_normal}")
    
    print("溢出处理结果:")
    for alloc in handled_allocations:
        compression_marker = " (已压缩)" if alloc.compression_needed else ""
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens{compression_marker}")
    
    print(f"\n✅ 溢出处理演示完成！")
    print(f"关键观察:")
    print(f"- 可丢弃的桶(scratchpad, few_shot_examples等)被压缩")
    print(f"- sticky桶(system_safety, task_instructions)保持相对完整")
    print(f"- 总分配量被控制在预算范围内")

if __name__ == "__main__":
    demonstrate_overflow()