"""Configuration settings for context engineering."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import yaml
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    context_limit: int
    output_target: int
    output_headroom: int
    
    @property
    def output_budget(self) -> int:
        """Total output budget including headroom."""
        return self.output_target + self.output_headroom


@dataclass
class BucketConfig:
    """Configuration for a context bucket."""
    min_tokens: int
    max_tokens: int
    weight: float
    sticky: bool = False
    compress: Optional[str] = None
    select: bool = False
    rerank: Optional[str] = None
    droppable: bool = False
    placement: str = "middle"
    content_score: float = 0.5


@dataclass
class PolicyConfig:
    """Policy configuration."""
    drop_order: List[str]
    placement: Dict[str, List[str]]
    overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextConfig:
    """Main configuration for context engineering."""
    model: ModelConfig
    buckets: Dict[str, BucketConfig]
    policies: Dict[str, PolicyConfig]
    system_overhead: int = 200
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ContextConfig':
        """Create configuration from dictionary."""
        # Parse model config
        model_data = config_dict.get('model', {})
        model = ModelConfig(
            name=model_data.get('name', 'gpt-4'),
            context_limit=model_data.get('context_limit', 8192),
            output_target=model_data.get('output_target', 1200),
            output_headroom=model_data.get('output_headroom', 300)
        )
        
        # Parse bucket configs
        buckets = {}
        buckets_data = config_dict.get('buckets', {})
        for name, bucket_data in buckets_data.items():
            buckets[name] = BucketConfig(**bucket_data)
        
        # Parse policy configs
        policies = {}
        policies_data = config_dict.get('policies', {})
        for name, policy_data in policies_data.items():
            policies[name] = PolicyConfig(**policy_data)
        
        system_overhead = config_dict.get('system_overhead', 200)
        
        return cls(
            model=model,
            buckets=buckets,
            policies=policies,
            system_overhead=system_overhead
        )
    
    @classmethod
    def from_yaml(cls, file_path: str) -> 'ContextConfig':
        """Load configuration from YAML file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, file_path: str) -> 'ContextConfig':
        """Load configuration from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model': {
                'name': self.model.name,
                'context_limit': self.model.context_limit,
                'output_target': self.model.output_target,
                'output_headroom': self.model.output_headroom
            },
            'buckets': {
                name: {
                    'min_tokens': bucket.min_tokens,
                    'max_tokens': bucket.max_tokens,
                    'weight': bucket.weight,
                    'sticky': bucket.sticky,
                    'compress': bucket.compress,
                    'select': bucket.select,
                    'rerank': bucket.rerank,
                    'droppable': bucket.droppable,
                    'placement': bucket.placement,
                    'content_score': bucket.content_score
                }
                for name, bucket in self.buckets.items()
            },
            'policies': {
                name: {
                    'drop_order': policy.drop_order,
                    'placement': policy.placement,
                    'overrides': policy.overrides
                }
                for name, policy in self.policies.items()
            },
            'system_overhead': self.system_overhead
        }
    
    def save_yaml(self, file_path: str):
        """Save configuration to YAML file."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
    
    def save_json(self, file_path: str):
        """Save configuration to JSON file."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def get_bucket_config(self, bucket_name: str) -> Optional[BucketConfig]:
        """Get configuration for a specific bucket."""
        return self.buckets.get(bucket_name)
    
    def get_policy_config(self, policy_name: str) -> Optional[PolicyConfig]:
        """Get configuration for a specific policy."""
        return self.policies.get(policy_name)
    
    def get_default_policy(self) -> Optional[PolicyConfig]:
        """Get the default policy configuration."""
        return self.policies.get('default')
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of issues.
        
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        # Validate model config
        if self.model.context_limit <= 0:
            issues.append("Model context limit must be positive")
        
        if self.model.output_budget >= self.model.context_limit:
            issues.append("Output budget exceeds model context limit")
        
        # Validate bucket configs
        if not self.buckets:
            issues.append("No buckets configured")
        
        for name, bucket in self.buckets.items():
            if bucket.min_tokens > bucket.max_tokens:
                issues.append(f"Bucket '{name}': min_tokens > max_tokens")
            
            if bucket.weight < 0:
                issues.append(f"Bucket '{name}': weight must be non-negative")
            
            if bucket.placement not in ['head', 'middle', 'tail']:
                issues.append(f"Bucket '{name}': invalid placement '{bucket.placement}'")
        
        # Validate policy configs
        for name, policy in self.policies.items():
            for bucket_name in policy.drop_order:
                if bucket_name not in self.buckets:
                    issues.append(f"Policy '{name}': unknown bucket in drop_order: '{bucket_name}'")
            
            for position, buckets in policy.placement.items():
                if position not in ['head', 'middle', 'tail']:
                    issues.append(f"Policy '{name}': invalid placement position '{position}'")
                
                for bucket_name in buckets:
                    if bucket_name not in self.buckets:
                        issues.append(f"Policy '{name}': unknown bucket in placement: '{bucket_name}'")
        
        return issues


def get_default_config() -> ContextConfig:
    """Get default configuration based on context_engineer.md specifications."""
    return ContextConfig.from_dict({
        'model': {
            'name': 'gpt-4',
            'context_limit': 8192,
            'output_target': 1200,
            'output_headroom': 300
        },
        'buckets': {
            'system': {
                'min_tokens': 300,
                'max_tokens': 800,
                'weight': 2.0,
                'sticky': True
            },
            'task': {
                'min_tokens': 300,
                'max_tokens': 1500,
                'weight': 2.5,
                'sticky': True
            },
            'tools': {
                'min_tokens': 120,
                'max_tokens': 400,
                'weight': 0.8,
                'compress': 'signature_only'
            },
            'history': {
                'min_tokens': 0,
                'max_tokens': 3000,
                'weight': 1.2,
                'compress': 'task_summary'
            },
            'memory': {
                'min_tokens': 0,
                'max_tokens': 800,
                'weight': 0.8,
                'select': True
            },
            'rag': {
                'min_tokens': 0,
                'max_tokens': 5000,
                'weight': 2.8,
                'select': True,
                'rerank': 'listwise'
            },
            'fewshot': {
                'min_tokens': 0,
                'max_tokens': 1200,
                'weight': 0.5,
                'droppable': True
            },
            'scratchpad': {
                'min_tokens': 0,
                'max_tokens': 800,
                'weight': 0.6,
                'placement': 'tail'
            }
        },
        'policies': {
            'default': {
                'drop_order': ['fewshot', 'rag', 'history', 'tools'],
                'placement': {
                    'head': ['system', 'task', 'tools'],
                    'middle': ['rag', 'history'],
                    'tail': ['scratchpad']
                }
            },
            'research_heavy': {
                'drop_order': ['fewshot', 'history', 'tools'],
                'placement': {
                    'head': ['system', 'task'],
                    'middle': ['rag', 'history'],
                    'tail': ['scratchpad', 'tools']
                },
                'overrides': {
                    'rag.weight': 3.5,
                    'model.output_target': 2000,
                    'history.compress': 'aggressive_extract'
                }
            }
        },
        'system_overhead': 200
    })