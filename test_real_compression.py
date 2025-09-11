#!/usr/bin/env python3
"""
创建真正需要压缩的场景来测试精确压缩逻辑
"""

from context_engineer.core.budget_manager import BudgetManager, CompatBucketConfig as BucketConfig
from context_engineer.core.context_assembler import ContextAssembler
from context_engineer.core.tokenizer_service import TokenizerService

def test_real_compression_scenario():
    print("=== 测试真实压缩场景 ===\n")
    
    # 创建tokenizer服务
    tokenizer = TokenizerService()
    
    # 创建预算管理器
    bm = BudgetManager(tokenizer)
    
    # 配置较小的桶，强制触发压缩
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
    
    bm.set_drop_order(["history"])
    
    print("场景：创建大量内容，强制触发压缩机制")
    
    # 创建大量内容（确保超过预算）
    large_content = {
        "system_safety": """
        SYSTEM SAFETY GUIDELINES:
        
        1. Always validate input data before processing
        2. Never execute untrusted code without proper sandboxing  
        3. Ensure all API calls are authenticated and authorized
        4. Monitor system resources and implement rate limiting
        5. Log all security events for audit purposes
        6. Use encryption for sensitive data transmission
        7. Implement proper error handling without information leakage
        8. Regular security audits and vulnerability assessments
        9. Keep all dependencies updated to latest secure versions
        10. Follow principle of least privilege for all operations
        """,
        
        "task_instructions": """
        TASK EXECUTION INSTRUCTIONS:
        
        Step 1: Analyze the provided requirements thoroughly
        Step 2: Break down complex tasks into manageable sub-tasks
        Step 3: Identify potential risks and mitigation strategies
        Step 4: Design solution architecture with clear components
        Step 5: Implement functionality following best practices
        Step 6: Write comprehensive tests for all components
        Step 7: Document code with clear comments and examples
        Step 8: Perform code review and quality assurance
        Step 9: Deploy to staging environment for testing
        Step 10: Monitor performance and gather feedback
        """,
        
        "history": """
        CONVERSATION HISTORY:
        
        User: I need help with implementing a new feature for my application.
        Assistant: I'd be happy to help you implement a new feature. Could you please provide more details about what specific functionality you're looking for?
        
        User: I want to add a user authentication system with login, logout, and registration capabilities.
        Assistant: That's a great feature to implement! Let me help you design a comprehensive authentication system. We'll need to consider security, user experience, and database design.
        
        User: Yes, security is very important. I also want to include password reset functionality and email verification.
        Assistant: Excellent requirements! I'll help you build a secure authentication system with all those features. Let me start by outlining the architecture and then we can implement it step by step.
        
        User: Perfect! I also need role-based access control with admin and regular user roles.
        Assistant: Role-based access control is crucial for application security. I'll include that in our authentication system design. We'll implement middleware to check user roles and permissions.
        
        User: Great! Can we also add two-factor authentication for extra security?
        Assistant: Absolutely! Two-factor authentication adds an important layer of security. I'll integrate that into our authentication system as well.
        """,
        
        "tools_schema": """
        TOOL SCHEMAS AND DEFINITIONS:
        
        Authentication Tools:
        - create_user(username: str, email: str, password: str) -> User
        - authenticate_user(username: str, password: str) -> AuthToken
        - verify_token(token: str) -> User
        - reset_password(email: str) -> ResetToken
        
        Database Tools:
        - query_users(filter_criteria: dict) -> List[User]
        - update_user(user_id: int, updates: dict) -> User
        - delete_user(user_id: int) -> bool
        - create_session(user_id: int) -> Session
        
        Security Tools:
        - hash_password(password: str) -> str
        - verify_password(password: str, hash: str) -> bool
        - generate_token(length: int = 32) -> str
        - send_email(to: str, subject: str, body: str) -> bool
        
        Validation Tools:
        - validate_email(email: str) -> bool
        - validate_password(password: str) -> ValidationResult
        - check_rate_limit(ip_address: str) -> bool
        """,
    }
    
    # 计算每个section的实际token数量
    print("实际内容token计数:")
    for name, content in large_content.items():
        actual_tokens = tokenizer.count_tokens(content)
        print(f"  {name}: {actual_tokens} tokens")
    print()
    
    # 设置非常紧张的预算，强制触发压缩
    model_limit = 400  # 非常紧张的预算
    output_budget = 50
    
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
    
    print(f"紧张预算下的分配结果 (模型限制: {model_limit}, 输出: {output_budget}):")
    total_allocated = 0
    for alloc in budget_allocations:
        total_allocated += alloc.allocated_tokens
        compression_marker = " (需要压缩)" if alloc.compression_needed else ""
        print(f"  {alloc.bucket_name}: {alloc.allocated_tokens} tokens{compression_marker}")
    print(f"总分配: {total_allocated} tokens")
    print(f"可用预算: {model_limit - output_budget - 200}")
    print()
    
    # 创建上下文组装器并组装内容
    assembler = ContextAssembler(tokenizer)
    
    assembled_context = assembler.assemble_context(
        content_sections=large_content,
        budget_allocations=budget_allocations,
        placement_policy={
            "head": ["system_safety", "task_instructions"],
            "middle": ["tools_schema", "history"],
            "tail": []
        }
    )
    
    print("精确压缩结果:")
    for section in assembled_context.sections:
        original_tokens = tokenizer.count_tokens(large_content[section.name])
        compressed_tokens = tokenizer.count_tokens(section.content)
        allocated = section.allocated_tokens
        
        print(f"  {section.name}:")
        print(f"    原始内容: {original_tokens} tokens")
        print(f"    分配预算: {allocated} tokens") 
        print(f"    压缩结果: {compressed_tokens} tokens")
        print(f"    压缩率: {(original_tokens - compressed_tokens) / original_tokens:.1%}")
        print(f"    预算符合: {'✅' if compressed_tokens <= allocated else '❌'} {compressed_tokens - allocated:+d}")
        print(f"    需要压缩: {section.compression_needed}")
        print(f"    内容长度: {len(section.content)} chars")
        print()
    
    print(f"总token数: {assembled_context.total_tokens}")
    print(f"预算利用率: {assembled_context.total_tokens / (model_limit - output_budget - 200):.1%}")
    print(f"丢弃的section: {assembled_context.dropped_sections}")
    
    # 验证精确压缩的效果
    print("\n" + "="*60)
    print("验证：预算分配 vs 实际压缩结果对比")
    
    total_budget = 0
    total_compressed = 0
    for section in assembled_context.sections:
        total_budget += section.allocated_tokens
        total_compressed += tokenizer.count_tokens(section.content)
    
    print(f"总预算分配: {total_budget} tokens")
    print(f"总压缩结果: {total_compressed} tokens")
    print(f"整体预算符合: {'✅' if total_compressed <= total_budget else '❌'} {total_compressed - total_budget:+d}")
    
    if total_compressed > total_budget:
        print("❌ 问题：压缩结果超出了预算分配！")
    else:
        print("✅ 成功：精确压缩符合预算要求！")

if __name__ == "__main__":
    test_real_compression_scenario()