"""Budget manager for token allocation across context buckets."""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from ..core.tokenizer_service import TokenizerService


# Import BucketConfig from config, but handle compatibility
from dataclasses import fields

class CompatBucketConfig:
    """Compatible BucketConfig that works with both core and config modules."""
    def __init__(self, name: str, min_tokens: int, max_tokens: int, weight: float,
                 sticky: bool = False, compress: Optional[str] = None, 
                 select: bool = False, rerank: Optional[str] = None,
                 droppable: bool = False, placement: str = "middle",
                 content_score: float = 0.5,
                 message_role: str = "system"):
        self.name = name
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.weight = weight
        self.sticky = sticky
        self.compress = compress
        self.select = select
        self.rerank = rerank
        self.droppable = droppable
        self.placement = placement
        self.content_score = content_score
        self.message_role = message_role

try:
    from ..config.settings import BucketConfig as ConfigBucketConfig
    
    # Create a wrapper that adds name field to config BucketConfig
    class BucketConfig:
        """Wrapper BucketConfig that ensures name field compatibility."""
        def __init__(self, name: str, **kwargs):
            self.name = name
            # Create underlying config bucket
            self._config_bucket = ConfigBucketConfig(**kwargs)
            # Copy all attributes
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def __getattr__(self, name):
            return getattr(self._config_bucket, name)
            
except ImportError:
    # Use our own BucketConfig if config module not available
    BucketConfig = CompatBucketConfig


@dataclass
class BudgetAllocation:
    """Result of budget allocation for a bucket."""
    bucket_name: str
    allocated_tokens: int
    priority: float
    content_score: float = 0.0
    # 移除了compression_needed字段，压缩决策完全基于内容vs预算的实时比较


class BudgetManager:
    """Manages token allocation across context buckets."""
    
    def __init__(self, tokenizer_service: Optional[TokenizerService] = None):
        """
        Initialize budget manager.
        
        Args:
            tokenizer_service: TokenizerService instance for token counting
        """
        self.tokenizer = tokenizer_service or TokenizerService()
        self.buckets: Dict[str, BucketConfig] = {}
        self.drop_order: List[str] = []
    
    def add_bucket(self, bucket_config: BucketConfig):
        """Add a bucket configuration."""
        self.buckets[bucket_config.name] = bucket_config
    
    def configure_buckets(self, bucket_configs: Dict[str, Any]):
        """
        Configure multiple buckets from dictionary or BucketConfig objects.
        
        Args:
            bucket_configs: Dictionary of bucket configurations (dict or BucketConfig objects)
        """
        for name, config in bucket_configs.items():
            if hasattr(config, 'min_tokens') and hasattr(config, 'max_tokens') and hasattr(config, 'weight'):
                # Looks like a BucketConfig object - create compatible one with name
                bucket_data = {
                    'min_tokens': config.min_tokens,
                    'max_tokens': config.max_tokens,
                    'weight': config.weight,
                    'sticky': getattr(config, 'sticky', False),
                    'compress': getattr(config, 'compress', None),
                    'select': getattr(config, 'select', False),
                    'rerank': getattr(config, 'rerank', None),
                    'droppable': getattr(config, 'droppable', False),
                    'placement': getattr(config, 'placement', 'middle'),
                    'content_score': getattr(config, 'content_score', 0.5),
                    'message_role': getattr(config, 'message_role', 'system')
                }
                bucket = BucketConfig(name=name, **bucket_data)
                self.add_bucket(bucket)
            elif isinstance(config, dict):
                # Dictionary configuration
                bucket = BucketConfig(name=name, **config)
                self.add_bucket(bucket)
            else:
                raise ValueError(f"Invalid bucket configuration type for {name}: {type(config)}")
    
    def calculate_budget(self, 
                        model_context_limit: int,
                        output_budget: int,
                        system_overhead: int = 200) -> int:
        """
        Calculate available input budget.
        
        Args:
            model_context_limit: Model's context window limit
            output_budget: Tokens reserved for output
            system_overhead: System overhead tokens
            
        Returns:
            Available input budget
        """
        return model_context_limit - output_budget - system_overhead
    
    def allocate_budget(self, 
                       model_context_limit: int,
                       output_budget: int,
                       content_scores: Optional[Dict[str, float]] = None,
                       system_overhead: int = 200) -> List[BudgetAllocation]:
        """
        Allocate budget across buckets using the algorithm from context_engineer.md.
        
        Args:
            model_context_limit: Model's context window limit
            output_budget: Tokens reserved for output
            content_scores: Relevance scores for each bucket's content (optional)
            system_overhead: System overhead tokens
            
        Returns:
            List of budget allocations
        """
        available_budget = self.calculate_budget(model_context_limit, output_budget, system_overhead)
        
        # Step 1: Initial allocation with minimum requirements
        allocations = self._initial_allocation(available_budget)
        remaining_budget = available_budget - sum(alloc.allocated_tokens for alloc in allocations)
        
        # Step 2: Dynamic optimization based on content scores
        if remaining_budget > 0:
            # 如果没有提供content_scores，使用BucketConfig中的默认值
            if content_scores is None:
                content_scores = {name: bucket.content_score for name, bucket in self.buckets.items()}
            allocations = self._optimize_allocation(allocations, content_scores, remaining_budget)
        
        # 移除无意义的溢出处理逻辑
        # 前面的分配逻辑已经确保总分配不会超过可用预算
        # 压缩决策应该基于实际内容 vs allocated_tokens，而不是预算层面的"溢出"
        return allocations
    
    def _initial_allocation(self, available_budget: int) -> List[BudgetAllocation]:
        """Step 1: Initial allocation with minimum requirements."""
        allocations = []
        min_total = sum(bucket.min_tokens for bucket in self.buckets.values())
        
        if available_budget < min_total:
            # Not enough budget for minimum requirements
            # 处理极端情况：预算为负值或零
            if available_budget <= 0:
                # 极端情况：每个桶至少分配1个token，按最小需求比例
                for bucket in self.buckets.values():
                    allocated = max(1, int((bucket.min_tokens / min_total) * 1))  # 按比例分配最少1个
                    allocation = BudgetAllocation(
                        bucket_name=bucket.name,
                        allocated_tokens=allocated,
                        priority=bucket.weight
                    )
                    allocations.append(allocation)
            else:
                # 正常比例缩减：按比例分配正数预算
                for bucket in self.buckets.values():
                    allocated = max(1, int((bucket.min_tokens / min_total) * available_budget))
                    allocation = BudgetAllocation(
                        bucket_name=bucket.name,
                        allocated_tokens=allocated,
                        priority=bucket.weight
                    )
                    allocations.append(allocation)
        else:
            # Normal allocation
            remaining_budget = available_budget - min_total
            total_weight = sum(bucket.weight for bucket in self.buckets.values())
            
            for bucket in self.buckets.values():
                # Calculate additional allocation based on weight
                additional = int((bucket.weight / total_weight) * remaining_budget)
                final_allocation = min(bucket.max_tokens, bucket.min_tokens + additional)
                
                allocation = BudgetAllocation(
                    bucket_name=bucket.name,
                    allocated_tokens=final_allocation,
                    priority=bucket.weight
                )
                allocations.append(allocation)
        
        return allocations
    
    def _optimize_allocation(self, 
                           allocations: List[BudgetAllocation],
                           content_scores: Dict[str, float],
                           remaining_budget: int) -> List[BudgetAllocation]:
        """Step 2: Optimize allocation based on content relevance scores."""
        # Calculate marginal utility for each bucket
        utilities = []
        for allocation in allocations:
            bucket = self.buckets[allocation.bucket_name]
            current_score = content_scores.get(bucket.name, 0.0)
            
            # Calculate potential improvement
            current_tokens = allocation.allocated_tokens
            max_tokens = bucket.max_tokens
            
            if current_tokens < max_tokens:
                # Marginal utility = score improvement per token
                marginal_utility = current_score / (current_tokens + 1)
                utilities.append((bucket.name, marginal_utility, max_tokens - current_tokens))
        
        # Sort by marginal utility (descending)
        utilities.sort(key=lambda x: x[1], reverse=True)
        
        # Allocate remaining budget based on marginal utility
        for bucket_name, utility, available in utilities:
            if remaining_budget <= 0:
                break
            
            allocation = next(a for a in allocations if a.bucket_name == bucket_name)
            allocate_amount = min(available, remaining_budget)
            allocation.allocated_tokens += allocate_amount
            remaining_budget -= allocate_amount
        
        return allocations
    
    # 移除了无意义的_handle_overflow方法
    # 压缩逻辑现在完全基于实际内容 vs allocated_tokens 的比较
    # 而不是基于预算层面的"溢出"检测
    
    def get_bucket_config(self, bucket_name: str) -> Optional[BucketConfig]:
        """Get configuration for a specific bucket."""
        return self.buckets.get(bucket_name)
    
    def set_drop_order(self, drop_order: List[str]):
        """Set the order for dropping content when budget is exceeded."""
        self.drop_order = drop_order
    
    def validate_configuration(self) -> bool:
        """Validate the bucket configuration."""
        if not self.buckets:
            return False
        
        # Check that all buckets in drop order exist
        for bucket_name in self.drop_order:
            if bucket_name not in self.buckets:
                return False
        
        # Check basic constraints
        for bucket in self.buckets.values():
            if bucket.min_tokens > bucket.max_tokens:
                return False
            if bucket.weight < 0:
                return False
        
        return True
    
    def get_total_min_tokens(self) -> int:
        """Get total minimum tokens required across all buckets."""
        return sum(bucket.min_tokens for bucket in self.buckets.values())
    
    def get_total_max_tokens(self) -> int:
        """Get total maximum tokens across all buckets."""
        return sum(bucket.max_tokens for bucket in self.buckets.values())