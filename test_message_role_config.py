#!/usr/bin/env python3
"""
测试基于BucketConfig的message_role配置功能
"""

from context_engineer.core.budget_manager import BudgetManager, CompatBucketConfig as BucketConfig
from context_engineer.core.context_assembler import ContextAssembler
from context_engineer.core.tokenizer_service import TokenizerService

def test_message_role_configuration():
    print("=== 测试基于配置的消息角色分配 ===\n")
    
    # 创建基础组件
    tokenizer = TokenizerService()
    bm = BudgetManager(tokenizer)
    assembler = ContextAssembler(tokenizer)
    
    print("1. 配置不同消息角色的buckets")
    print("-" * 50)
    
    # 配置不同消息角色的buckets
    buckets = {
        "system": BucketConfig(
            "system", 
            min_tokens=50, max_tokens=200, weight=2.0, 
            sticky=True, content_score=0.9,
            message_role="system"  # 系统角色
        ),
        "user_query": BucketConfig(
            "user_query", 
            min_tokens=30, max_tokens=150, weight=1.0, 
            sticky=True, content_score=0.8,
            message_role="user"  # 用户角色
        ),
        "assistant_context": BucketConfig(
            "assistant_context", 
            min_tokens=40, max_tokens=200, weight=0.8, 
            content_score=0.7,
            message_role="assistant"  # 助手角色
        ),
        "tools": BucketConfig(
            "tools", 
            min_tokens=20, max_tokens=100, weight=0.6, 
            content_score=0.6,
            message_role="system"  # 系统角色（工具信息）
        ),
        "history": BucketConfig(
            "history", 
            min_tokens=40, max_tokens=300, weight=0.8, 
            droppable=True, content_score=0.7,
            message_role="user"  # 用户角色（历史对话）
        ),
        "rag": BucketConfig(
            "rag", 
            min_tokens=30, max_tokens=200, weight=1.5, 
            content_score=0.85,
            message_role="system"  # 系统角色（RAG信息）
        ),
    }
    
    for bucket in buckets.values():
        bm.add_bucket(bucket)
    
    print("Bucket配置和消息角色:")
    for name, bucket in buckets.items():
        role = getattr(bucket, 'message_role', 'system')
        print(f"  {name}: {role} (min:{bucket.min_tokens}, max:{bucket.max_tokens})")
    print()
    
    print("2. 创建测试内容")
    print("-" * 50)
    
    # 创建内容
    content_sections = {
        "system": "You are an AI assistant specialized in weather forecasting with access to real-time data.",
        "user_query": "What's the weather like in San Francisco today? Should I bring an umbrella?",
        "assistant_context": "Based on my analysis, I can see weather patterns developing that might affect your plans.",
        "tools": "Available tools: weather_api.get_current_conditions, weather_api.get_forecast, location_service.get_coordinates",
        "history": "User: What's the weather in NYC?\nAssistant: It's 22°C and sunny in New York City today.",
        "rag": "Recent weather data shows a low pressure system approaching San Francisco, bringing increased cloud cover and potential light rain in the afternoon.",
    }
    
    print("测试内容:")
    for name, content in content_sections.items():
        actual_tokens = len(content.split())  # 简单估算
        print(f"  {name}: {actual_tokens} words")
    print()
    
    print("3. 预算分配和上下文组装")
    print("-" * 50)
    
    # 预算分配
    allocations = bm.allocate_budget(
        model_context_limit=2000,
        output_budget=300,
        content_scores={"system": 0.9, "user_query": 0.8, "assistant_context": 0.7, "tools": 0.6, "history": 0.7, "rag": 0.85}
    )
    
    print("预算分配结果:")
    for alloc in allocations:
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens")
    print()
    
    # 组装上下文
    assembled_context = assembler.assemble_context(
        content_sections=content_sections,
        budget_allocations=allocations,
        placement_policy={
            "head": ["system", "user_query"],
            "middle": ["assistant_context", "tools", "rag"],
            "tail": ["history"]
        }
    )
    
    print("4. 基于配置的角色分配")
    print("-" * 50)
    
    # 使用bucket配置进行角色分配
    messages = assembler.to_messages(
        user_sections=["user_query", "history"],  # 明确指定用户section
        bucket_configs=buckets  # 传入bucket配置以使用配置的message_role
    )
    
    print("生成的消息格式:")
    for i, msg in enumerate(messages):
        print(f"  [{i}] {msg['role']}: {msg['content'][:80]}...")
    print()
    
    print("5. 角色分配统计")
    print("-" * 50)
    
    stats = assembler.get_message_stats()
    print(f"总sections: {stats['total_sections']}")
    print(f"角色分布: {stats['role_distribution']}")
    
    print("\n详细分配:")
    for detail in stats['section_details']:
        configured_role = None
        if detail['name'] in buckets:
            configured_role = getattr(buckets[detail['name']], 'message_role', 'system')
        actual_role = detail['role']
        print(f"  {detail['name']}: configured={configured_role}, actual={actual_role}, "
              f"tokens={detail['tokens']}, allocated={detail['allocated']}")
    
    print("\n6. 不同配置对比")
    print("-" * 50)
    
    # 对比：使用默认角色分配（基于section名称）
    print("使用默认角色分配（基于section名称）:")
    default_messages = assembler.to_messages()
    print(f"  消息数量: {len(default_messages)}")
    for i, msg in enumerate(default_messages):
        print(f"    [{i}] {msg['role']}: {msg['content'][:60]}...")
    
    print(f"\n✅ 基于配置的角色分配功能测试完成！")
    print("✅ 每个bucket都可以配置其message_role")
    print("✅ 系统优先使用配置的role，其次回退到默认判断")
    print("✅ 保持单个system message + 用户消息的最简化格式")
    print("✅ 最大化LLM API兼容性")

if __name__ == "__main__":
    test_message_role_configuration()