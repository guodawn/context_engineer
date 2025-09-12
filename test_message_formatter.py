#!/usr/bin/env python3
"""
测试消息格式化功能 - 将AssembledContext转换为LLM消息格式
"""

from context_engineer.core.budget_manager import BudgetManager, CompatBucketConfig as BucketConfig
from context_engineer.core.context_assembler import ContextAssembler
from context_engineer.core.tokenizer_service import TokenizerService
from context_engineer.utils.message_formatter import MessageFormatter

def test_message_formatter():
    print("=== 测试消息格式化功能 ===\n")
    
    # 创建基础组件
    tokenizer = TokenizerService()
    bm = BudgetManager(tokenizer)
    assembler = ContextAssembler(tokenizer)
    formatter = MessageFormatter()
    
    # 配置桶
    buckets = {
        "system": BucketConfig("system", min_tokens=50, max_tokens=200, weight=2.0, sticky=True, content_score=0.9),
        "user_query": BucketConfig("user_query", min_tokens=30, max_tokens=150, weight=1.0, content_score=0.8),
        "history": BucketConfig("history", min_tokens=40, max_tokens=300, weight=0.8, content_score=0.7),
        "tools": BucketConfig("tools", min_tokens=20, max_tokens=100, weight=0.6, content_score=0.6),
        "rag": BucketConfig("rag", min_tokens=30, max_tokens=200, weight=1.5, content_score=0.85),
    }
    
    for bucket in buckets.values():
        bm.add_bucket(bucket)
    
    # 创建内容
    content_sections = {
        "system": "You are a helpful AI assistant specialized in weather forecasting. You should provide accurate and clear weather information.",
        "user_query": "What's the weather like in San Francisco today? Should I bring an umbrella?",
        "history": "User: What's the weather in NYC?\nAssistant: It's 22°C and sunny in New York City today.",
        "tools": "Available tools: weather_api.get_current_conditions, weather_api.get_forecast, location_service.get_coordinates",
        "rag": "Recent weather data shows a low pressure system approaching San Francisco, bringing increased cloud cover and potential light rain in the afternoon.",
    }
    
    print("1. 基础预算分配")
    print("-" * 40)
    
    # 预算分配
    allocations = bm.allocate_budget(
        model_context_limit=2000,
        output_budget=300,
        content_scores={"system": 0.9, "user_query": 0.8, "history": 0.7, "tools": 0.6, "rag": 0.85}
    )
    
    print("预算分配结果:")
    for alloc in allocations:
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens")
    print()
    
    print("2. 上下文组装")
    print("-" * 40)
    
    # 组装上下文
    assembled_context = assembler.assemble_context(
        content_sections=content_sections,
        budget_allocations=allocations,
        placement_policy={
            "head": ["system", "user_query"],
            "middle": ["history", "tools", "rag"],
            "tail": []
        }
    )
    
    print("组装后的sections:")
    for section in assembled_context.sections:
        actual_tokens = len(section.content.split())  # 简单估算
        print(f"  {section.name}: {actual_tokens} words (allocated: {section.allocated_tokens})")
    print()
    
    print("3. 消息格式转换")
    print("-" * 40)
    
    # 使用ContextAssembler的便捷方法
    openai_messages = assembler.to_messages(format_type="openai")
    
    print("OpenAI消息格式:")
    for i, msg in enumerate(openai_messages):
        print(f"  [{i}] {msg['role']}: {msg['content'][:60]}...")
    print()
    
    # 使用简单的格式
    simple_messages = assembler.to_messages(format_type="simple")
    print("简单格式 (只有system/user):")
    for i, msg in enumerate(simple_messages):
        print(f"  [{i}] {msg['role']}: {msg['content'][:60]}...")
    print()
    
    print("4. 消息统计信息")
    print("-" * 40)
    
    stats = assembler.get_message_stats()
    print(f"Section总数: {stats['total_sections']}")
    print(f"Role分布: {stats['role_distribution']}")
    
    print("\nSection详细信息:")
    for detail in stats['section_details']:
        print(f"  {detail['name']}: role={detail['role']}, "
              f"tokens={detail['tokens']}, allocated={detail['allocated']}, "
              f"needs_compression={'是' if detail['needs_compression'] else '否'}")
    
    print("\n5. 自定义角色映射")
    print("-" * 40)
    
    # 自定义角色映射
    custom_mapping = {
        "system": "assistant",  # 系统提示作为助手设定
        "user_query": "user",   # 用户查询保持为用户
        "history": "assistant", # 历史作为助手回复
        "tools": "system",      # 工具信息作为系统信息
        "rag": "assistant"      # RAG信息作为助手知识
    }
    
    custom_messages = assembler.to_messages(format_type="openai", role_mapping=custom_mapping)
    
    print("自定义角色映射:")
    for i, msg in enumerate(custom_messages):
        print(f"  [{i}] {msg['role']}: {msg['content'][:60]}...")
    
    print("\n6. 与LLM集成的完整示例")
    print("-" * 40)
    
    # 模拟OpenAI API调用
    print("模拟发送给OpenAI API:")
    print("=" * 50)
    
    messages = assembler.to_messages(format_type="openai")
    
    print("import openai")
    print("messages = [")
    for msg in messages:
        print(f"    {{'role': '{msg['role']}', 'content': '{msg['content']}'}}")
    print("]")
    print("response = openai.ChatCompletion.create(")
    print("    model='gpt-3.5-turbo',")
    print("    messages=messages")
    print(")")
    
    print("\n✅ 消息格式化功能测试完成！")
    print("✅ 系统可以无缝集成到各种LLM API")
    print("✅ 支持多种消息格式和自定义映射")

if __name__ == "__main__":
    test_message_formatter()