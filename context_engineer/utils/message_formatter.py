"""
Message formatter for converting AssembledContext to various LLM message formats.
"""

from typing import List, Dict, Any, Optional
from ..core.context_assembler import AssembledContext, ContextSection


class MessageFormatter:
    """Converts AssembledContext to different LLM message formats."""
    
    def __init__(self):
        """Initialize message formatter."""
        # 默认的section到role的映射
        self.default_role_mapping = {
            "system": "system",
            "user": "user", 
            "assistant": "assistant",
            "task": "user",  # 任务描述通常作为用户输入
            "tools": "system",  # 工具信息作为系统上下文
            "history": "user",  # 历史对话作为用户输入
            "memory": "system",  # 记忆信息作为系统上下文
            "rag": "system",  # RAG信息作为系统上下文
            "fewshot": "assistant",  # few-shot示例作为助手回复
            "scratchpad": "assistant"  # 思考过程作为助手内部状态
        }
    
    def to_openai_messages(
        self, 
        assembled_context: AssembledContext,
        role_mapping: Optional[Dict[str, str]] = None,
        include_placement: bool = False
    ) -> List[Dict[str, str]]:
        """
        将AssembledContext转换为OpenAI消息格式。
        
        Args:
            assembled_context: 组装后的上下文
            role_mapping: 自定义的section到role的映射
            include_placement: 是否在消息中包含位置信息
            
        Returns:
            OpenAI格式的消息列表
        """
        if not assembled_context.sections:
            return []
        
        # 使用自定义映射或默认映射
        mapping = role_mapping or self.default_role_mapping
        
        messages = []
        
        # 按照placement顺序处理：head -> middle -> tail
        placement_order = ["head", "middle", "tail"]
        
        for placement in placement_order:
            if placement in assembled_context.placement_map:
                section_names = assembled_context.placement_map[placement]
                
                for section_name in section_names:
                    # 找到对应的section
                    section = next((s for s in assembled_context.sections if s.name == section_name), None)
                    if not section or not section.content.strip():
                        continue
                    
                    # 确定role
                    role = mapping.get(section_name, "system")  # 默认使用system role
                    
                    # 构建消息内容
                    content = section.content.strip()
                    if include_placement:
                        content = f"[{placement}] {content}"
                    
                    messages.append({
                        "role": role,
                        "content": content
                    })
        
        return messages
    
    def to_openai_messages_simple(
        self, 
        assembled_context: AssembledContext,
        user_sections: Optional[List[str]] = None,
        bucket_configs: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """
        最简化的OpenAI消息格式：单个system message + 用户消息。
        
        策略：
        - 基于bucket配置的message_role来决定每个section的角色
        - 除用户消息外，全部合并为单个system message
        - 保持位置顺序和逻辑结构
        - 最大化LLM兼容性
        
        Args:
            assembled_context: 组装后的上下文
            user_sections: 指定哪些section应该作为user role（默认：["user_query", "user", "input"]）
            bucket_configs: Bucket配置，用于获取message_role（可选）
            
        Returns:
            简化的消息列表 [system, user] 或类似结构
        """
        if not assembled_context.sections:
            return []
        
        # 默认的用户section
        default_user = ["user_query", "user", "input"]
        user_sections = user_sections or default_user
        
        messages = []
        system_parts = []
        user_parts = []
        
        # 按照placement顺序处理
        placement_order = ["head", "middle", "tail"]
        
        for placement in placement_order:
            if placement in assembled_context.placement_map:
                section_names = assembled_context.placement_map[placement]
                
                for section_name in section_names:
                    section = next((s for s in assembled_context.sections if s.name == section_name), None)
                    if not section or not section.content.strip():
                        continue
                    
                    # 获取配置的message_role（如果有bucket配置）
                    configured_role = None
                    if bucket_configs and section_name in bucket_configs:
                        bucket_config = bucket_configs[section_name]
                        configured_role = getattr(bucket_config, 'message_role', None)
                    
                    # 优先使用配置的role，其次基于section名称判断
                    if configured_role and configured_role == "user":
                        is_user = True
                    elif configured_role and configured_role in ["system", "assistant"]:
                        is_user = False
                    else:
                        # 回退到基于section名称的默认判断
                        is_user = section_name in user_sections
                    
                    if is_user:
                        # 用户消息单独处理
                        user_parts.append(section.content.strip())
                    else:
                        # 系统消息合并处理
                        system_parts.append(section.content.strip())
        
        # 构建最终消息
        if system_parts:
            # 合并所有系统消息为一个
            combined_system = "\n\n".join(system_parts)
            messages.append({
                "role": "system",
                "content": combined_system
            })
        
        if user_parts:
            # 合并所有用户消息为一个（如果有多个）
            combined_user = "\n\n".join(user_parts)
            messages.append({
                "role": "user", 
                "content": combined_user
            })
        
        return messages
    
    def to_anthropic_messages(
        self, 
        assembled_context: AssembledContext,
        role_mapping: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        转换为Anthropic Claude的消息格式。
        
        Args:
            assembled_context: 组装后的上下文
            role_mapping: 自定义的section到role的映射
            
        Returns:
            Anthropic格式的消息列表
        """
        if not assembled_context.sections:
            return []
        
        # Anthropic主要使用user和assistant，将system内容合并到user
        mapping = role_mapping or {
            "system": "user",
            "user": "user", 
            "assistant": "assistant"
        }
        
        messages = []
        
        # 按照placement顺序处理
        placement_order = ["head", "middle", "tail"]
        
        for placement in placement_order:
            if placement in assembled_context.placement_map:
                section_names = assembled_context.placement_map[placement]
                
                for section_name in section_names:
                    section = next((s for s in assembled_context.sections if s.name == section_name), None)
                    if not section or not section.content.strip():
                        continue
                    
                    role = mapping.get(section_name, "user")
                    
                    messages.append({
                        "role": role,
                        "content": section.content.strip()
                    })
        
        return messages
    
    def create_custom_mapping(
        self, 
        section_roles: Dict[str, str]
    ) -> Dict[str, str]:
        """
        创建自定义的section到role的映射。
        
        Args:
            section_roles: 用户定义的section到role的映射
            
        Returns:
            完整的映射字典
        """
        # 基于默认映射，允许用户自定义
        custom_mapping = self.default_role_mapping.copy()
        custom_mapping.update(section_roles)
        return custom_mapping
    
    def get_section_role_summary(
        self, 
        assembled_context: AssembledContext,
        role_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        获取section到role的映射摘要。
        
        Args:
            assembled_context: 组装后的上下文
            role_mapping: 使用的映射
            
        Returns:
            映射摘要信息
        """
        mapping = role_mapping or self.default_role_mapping
        
        summary = {
            "total_sections": len(assembled_context.sections),
            "role_distribution": {},
            "section_details": []
        }
        
        role_counts = {}
        
        for section in assembled_context.sections:
            role = mapping.get(section.name, "system")
            role_counts[role] = role_counts.get(role, 0) + 1
            
            needs_compression = section.token_count > section.allocated_tokens
            
            summary["section_details"].append({
                "name": section.name,
                "role": role,
                "tokens": section.token_count,
                "allocated": section.allocated_tokens,
                "needs_compression": needs_compression,
                "placement": section.placement
            })
        
        summary["role_distribution"] = role_counts
        return summary


# 便捷函数，直接添加到ContextAssembler类中
def extend_context_assembler():
    """为ContextAssembler添加消息格式化功能。"""
    from ..core.context_assembler import ContextAssembler
    
    def to_messages(self, format_type: str = "openai", **kwargs) -> List[Dict[str, str]]:
        """
        将AssembledContext转换为指定格式的消息。
        
        Args:
            format_type: 消息格式类型 ("openai", "anthropic", "simple")
            **kwargs: 传递给格式化器的其他参数
            
        Returns:
            格式化后的消息列表
        """
        if not hasattr(self, '_message_formatter'):
            self._message_formatter = MessageFormatter()
        
        if format_type == "openai":
            return self._message_formatter.to_openai_messages(self.last_result, **kwargs)
        elif format_type == "anthropic":
            return self._message_formatter.to_anthropic_messages(self.last_result, **kwargs)
        elif format_type == "simple":
            return self._message_formatter.to_openai_messages_simple(self.last_result, **kwargs)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    # 添加到ContextAssembler类
    ContextAssembler.to_messages = to_messages
    ContextAssembler._message_formatter = None  # 延迟初始化
    
    return MessageFormatter()  # 返回实例供使用


if __name__ == "__main__":
    # 测试代码
    from context_engineer.core.context_assembler import ContextAssembler, ContextSection, AssembledContext
    from context_engineer.core.tokenizer_service import TokenizerService
    
    # 创建测试数据
    tokenizer = TokenizerService()
    
    # 创建测试sections
    test_sections = [
        ContextSection(
            name="system",
            content="You are a helpful weather assistant.",
            priority=1.0,
            placement="head",
            token_count=8,
            allocated_tokens=50
        ),
        ContextSection(
            name="user_query", 
            content="What's the weather in San Francisco?",
            priority=0.8,
            placement="head",
            token_count=7,
            allocated_tokens=30
        ),
        ContextSection(
            name="weather_data",
            content="Current weather: 22°C, partly cloudy",
            priority=0.9,
            placement="middle",
            token_count=6,
            allocated_tokens=40
        )
    ]
    
    # 创建测试AssembledContext
    test_context = AssembledContext(
        full_context="You are a helpful weather assistant.\n\nWhat's the weather in San Francisco?\n\nCurrent weather: 22°C, partly cloudy",
        sections=test_sections,
        total_tokens=21,
        placement_map={"head": ["system", "user_query"], "middle": ["weather_data"], "tail": []},
        dropped_sections=[]
    )
    
    # 测试转换功能
    formatter = MessageFormatter()
    
    print("=== OpenAI Format ===")
    openai_messages = formatter.to_openai_messages(test_context)
    for msg in openai_messages:
        print(f"{msg['role']}: {msg['content'][:50]}...")
    
    print("\n=== Simple Format ===")
    simple_messages = formatter.to_openai_messages_simple(test_context)
    for msg in simple_messages:
        print(f"{msg['role']}: {msg['content'][:50]}...")
    
    print("\n=== Role Summary ===")
    summary = formatter.get_section_role_summary(test_context)
    print(f"Role distribution: {summary['role_distribution']}")
    print(f"Total sections: {summary['total_sections']}")