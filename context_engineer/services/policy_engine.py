"""Policy engine for managing different context engineering strategies."""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from ..config.settings import ContextConfig, PolicyConfig
from ..core.budget_manager import BudgetManager


class TaskType(Enum):
    """Types of tasks that may require different context strategies."""
    GENERAL = "general"
    RESEARCH = "research"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    CONVERSATION = "conversation"
    TECHNICAL_SUPPORT = "technical_support"
    DATA_PROCESSING = "data_processing"


class RiskLevel(Enum):
    """Risk levels for different operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CostTarget(Enum):
    """Cost optimization targets."""
    MINIMIZE = "minimize"
    BALANCED = "balanced"
    MAXIMIZE_QUALITY = "maximize_quality"


@dataclass
class PolicyContext:
    """Context for policy decision making."""
    task_type: TaskType
    risk_level: RiskLevel
    cost_target: CostTarget
    content_types: Set[str]  # Types of content available
    priority_sections: List[str]  # Sections that must be included
    excluded_sections: List[str]  # Sections that should be excluded
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyDecision:
    """Result of policy decision."""
    policy_name: str
    bucket_overrides: Dict[str, Dict[str, Any]]
    placement_strategy: Dict[str, List[str]]
    compression_methods: Dict[str, str]
    drop_order: List[str]
    system_overrides: Dict[str, Any]
    reasoning: str


class PolicyEngine:
    """Engine for selecting and applying context engineering policies."""
    
    def __init__(self, config: Optional[ContextConfig] = None):
        """
        Initialize policy engine.
        
        Args:
            config: ContextConfig instance with policy definitions
        """
        self.config = config
        self.policies: Dict[str, PolicyConfig] = {}
        self.task_mappings: Dict[TaskType, str] = {}
        self.risk_mappings: Dict[RiskLevel, str] = {}
        
        if config:
            self._load_config_policies()
        else:
            self._setup_default_policies()
    
    def _load_config_policies(self):
        """Load policies from configuration."""
        if self.config and self.config.policies:
            self.policies = self.config.policies
    
    def _setup_default_policies(self):
        """Setup default policies based on context_engineer.md."""
        default_policy = PolicyConfig(
            drop_order=['fewshot', 'rag', 'history', 'tools'],
            placement={
                'head': ['system', 'task', 'tools'],
                'middle': ['rag', 'history'],
                'tail': ['scratchpad']
            },
            overrides={}
        )
        
        research_policy = PolicyConfig(
            drop_order=['fewshot', 'history', 'tools'],
            placement={
                'head': ['system', 'task'],
                'middle': ['rag', 'history'],
                'tail': ['scratchpad', 'tools']
            },
            overrides={
                'rag.weight': 3.5,
                'history.compress': 'aggressive_extract'
            }
        )
        
        code_policy = PolicyConfig(
            drop_order=['fewshot', 'history', 'memory'],
            placement={
                'head': ['system', 'task', 'tools'],
                'middle': ['memory', 'history'],
                'tail': ['scratchpad']
            },
            overrides={
                'tools.weight': 2.0,
                'tools.compress': 'signature_only'
            }
        )
        
        self.policies = {
            'default': default_policy,
            'research_heavy': research_policy,
            'code_generation': code_policy
        }
        
        # Setup default mappings
        self.task_mappings = {
            TaskType.GENERAL: 'default',
            TaskType.RESEARCH: 'research_heavy',
            TaskType.CODE_GENERATION: 'code_generation',
            TaskType.ANALYSIS: 'research_heavy',
            TaskType.TECHNICAL_SUPPORT: 'default'
        }
        
        self.risk_mappings = {
            RiskLevel.LOW: 'default',
            RiskLevel.MEDIUM: 'default',
            RiskLevel.HIGH: 'research_heavy',
            RiskLevel.CRITICAL: 'research_heavy'
        }
    
    def select_policy(self, context: PolicyContext) -> PolicyDecision:
        """
        Select appropriate policy based on context.
        
        Args:
            context: PolicyContext with task and environment information
            
        Returns:
            PolicyDecision with specific configurations to apply
        """
        # Determine base policy
        base_policy_name = self._determine_base_policy(context)
        base_policy = self.policies.get(base_policy_name, self.policies['default'])
        
        # Apply context-specific modifications
        overrides = self._apply_context_overrides(context, base_policy)
        placement_strategy = self._optimize_placement(context, base_policy)
        compression_methods = self._select_compression_methods(context)
        drop_order = self._optimize_drop_order(context, base_policy)
        system_overrides = self._apply_system_overrides(context)
        
        reasoning = self._generate_reasoning(context, base_policy_name, overrides)
        
        return PolicyDecision(
            policy_name=base_policy_name,
            bucket_overrides=overrides,
            placement_strategy=placement_strategy,
            compression_methods=compression_methods,
            drop_order=drop_order,
            system_overrides=system_overrides,
            reasoning=reasoning
        )
    
    def _determine_base_policy(self, context: PolicyContext) -> str:
        """Determine the base policy to use."""
        # Check task type mapping
        if context.task_type in self.task_mappings:
            return self.task_mappings[context.task_type]
        
        # Check risk level mapping
        if context.risk_level in self.risk_mappings:
            return self.risk_mappings[context.risk_level]
        
        # Default policy
        return 'default'
    
    def _apply_context_overrides(self, context: PolicyContext, base_policy: PolicyConfig) -> Dict[str, Dict[str, Any]]:
        """Apply context-specific bucket overrides."""
        overrides = base_policy.overrides.copy()
        
        # Task-specific overrides
        if context.task_type == TaskType.RESEARCH:
            overrides.update({
                'rag.weight': 3.5,
                'history.weight': 1.5,
                'memory.weight': 1.2
            })
        elif context.task_type == TaskType.CODE_GENERATION:
            overrides.update({
                'tools.weight': 2.5,
                'system.weight': 2.0,
                'fewshot.weight': 1.5
            })
        elif context.task_type == TaskType.CONVERSATION:
            overrides.update({
                'history.weight': 2.5,
                'memory.weight': 1.8,
                'task.weight': 1.5
            })
        
        # Risk-based overrides
        if context.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            overrides.update({
                'system.sticky': True,
                'task.sticky': True,
                'tools.sticky': True
            })
        
        # Cost-based overrides
        if context.cost_target == CostTarget.MINIMIZE:
            overrides.update({
                'rag.max_tokens': 1000,  # Reduce RAG tokens
                'history.max_tokens': 500,
                'fewshot.max_tokens': 200
            })
        elif context.cost_target == CostTarget.MAXIMIZE_QUALITY:
            overrides.update({
                'rag.max_tokens': 8000,  # Increase RAG tokens
                'history.max_tokens': 3000,
                'memory.max_tokens': 1500
            })
        
        # Content type specific overrides
        if 'code' in context.content_types:
            overrides.update({
                'tools.compress': 'signature_only',
                'fewshot.weight': 1.5
            })
        
        if 'conversation' in context.content_types:
            overrides.update({
                'history.compress': 'task_summary',
                'memory.select': True
            })
        
        return self._organize_overrides_by_bucket(overrides)
    
    def _organize_overrides_by_bucket(self, overrides: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Organize flat override dictionary by bucket."""
        bucket_overrides = {}
        
        for key, value in overrides.items():
            if '.' in key:
                bucket_name, attribute = key.split('.', 1)
                if bucket_name not in bucket_overrides:
                    bucket_overrides[bucket_name] = {}
                bucket_overrides[bucket_name][attribute] = value
        
        return bucket_overrides
    
    def _optimize_placement(self, context: PolicyContext, base_policy: PolicyConfig) -> Dict[str, List[str]]:
        """Optimize placement strategy based on context."""
        placement = base_policy.placement.copy()
        
        # Ensure priority sections are placed optimally
        for section in context.priority_sections:
            # Move priority sections to head if not already there
            if section not in placement['head']:
                # Remove from other placements
                for pos_list in placement.values():
                    if section in pos_list:
                        pos_list.remove(section)
                # Add to head
                placement['head'].append(section)
        
        # Remove excluded sections
        for section in context.excluded_sections:
            for pos_list in placement.values():
                if section in pos_list:
                    pos_list.remove(section)
        
        # Task-specific placement optimizations
        if context.task_type == TaskType.RESEARCH:
            # Prioritize RAG and memory in middle
            if 'rag' not in placement['middle']:
                placement['middle'].insert(0, 'rag')
            if 'memory' not in placement['middle']:
                placement['middle'].append('memory')
        
        return placement
    
    def _select_compression_methods(self, context: PolicyContext) -> Dict[str, str]:
        """Select compression methods based on context."""
        methods = {}
        
        # Default methods
        base_methods = {
            'history': 'task_summary',
            'rag': 'extractive',
            'tools': 'signature_only',
            'fewshot': 'truncate'
        }
        
        methods.update(base_methods)
        
        # Context-specific adjustments
        if context.task_type == TaskType.RESEARCH:
            methods.update({
                'rag': 'extractive',
                'history': 'aggressive_extract'
            })
        elif context.task_type == TaskType.CODE_GENERATION:
            methods.update({
                'tools': 'signature_only',
                'fewshot': 'extractive'
            })
        
        # Risk-based adjustments
        if context.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            methods.update({
                'system': 'truncate',  # Minimal compression for critical info
                'task': 'truncate'
            })
        
        return methods
    
    def _optimize_drop_order(self, context: PolicyContext, base_policy: PolicyConfig) -> List[str]:
        """Optimize drop order based on context."""
        drop_order = base_policy.drop_order.copy()
        
        # Never drop priority sections
        for section in context.priority_sections:
            if section in drop_order:
                drop_order.remove(section)
        
        # Add excluded sections to drop order first
        for section in reversed(context.excluded_sections):
            if section not in drop_order:
                drop_order.insert(0, section)
        
        # Context-specific drop order adjustments
        if context.task_type == TaskType.RESEARCH:
            # Keep RAG longer
            if 'rag' in drop_order:
                drop_order.remove('rag')
                drop_order.append('rag')
        
        return drop_order
    
    def _apply_system_overrides(self, context: PolicyContext) -> Dict[str, Any]:
        """Apply system-level overrides."""
        overrides = {}
        
        # Cost-based system overrides
        if context.cost_target == CostTarget.MINIMIZE:
            overrides.update({
                'model.output_target': 500,  # Reduce output target
                'system_overhead': 100  # Reduce system overhead
            })
        elif context.cost_target == CostTarget.MAXIMIZE_QUALITY:
            overrides.update({
                'model.output_target': 2000,  # Increase output target
                'system_overhead': 300  # Allow more overhead for quality
            })
        
        # Risk-based system overrides
        if context.risk_level == RiskLevel.CRITICAL:
            overrides.update({
                'model.output_headroom': 500,  # More headroom for critical tasks
                'system_overhead': 400  # More overhead for safety
            })
        
        return overrides
    
    def _generate_reasoning(self, context: PolicyContext, policy_name: str, overrides: Dict[str, Dict[str, Any]]) -> str:
        """Generate reasoning for policy selection."""
        reasoning_parts = [
            f"Selected policy '{policy_name}' based on:",
            f"- Task type: {context.task_type.value}",
            f"- Risk level: {context.risk_level.value}",
            f"- Cost target: {context.cost_target.value}"
        ]
        
        if context.priority_sections:
            reasoning_parts.append(f"- Priority sections: {', '.join(context.priority_sections)}")
        
        if overrides:
            reasoning_parts.append(f"- Applied {len(overrides)} bucket overrides")
        
        return " ".join(reasoning_parts)
    
    def add_policy(self, name: str, policy: PolicyConfig):
        """Add a custom policy."""
        self.policies[name] = policy
    
    def set_task_mapping(self, task_type: TaskType, policy_name: str):
        """Set policy mapping for a specific task type."""
        if policy_name not in self.policies:
            raise ValueError(f"Unknown policy: {policy_name}")
        
        self.task_mappings[task_type] = policy_name
    
    def set_risk_mapping(self, risk_level: RiskLevel, policy_name: str):
        """Set policy mapping for a specific risk level."""
        if policy_name not in self.policies:
            raise ValueError(f"Unknown policy: {policy_name}")
        
        self.risk_mappings[risk_level] = policy_name
    
    def get_policy(self, name: str) -> Optional[PolicyConfig]:
        """Get a specific policy by name."""
        return self.policies.get(name)
    
    def list_policies(self) -> List[str]:
        """List all available policy names."""
        return list(self.policies.keys())
    
    def validate_policy(self, policy_name: str, context: PolicyContext) -> List[str]:
        """
        Validate that a policy is appropriate for a given context.
        
        Args:
            policy_name: Name of the policy to validate
            context: Context to validate against
            
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        if policy_name not in self.policies:
            issues.append(f"Unknown policy: {policy_name}")
            return issues
        
        policy = self.policies[policy_name]
        
        # Check that priority sections are not in drop order
        for section in context.priority_sections:
            if section in policy.drop_order:
                issues.append(f"Priority section '{section}' is in drop order for policy '{policy_name}'")
        
        # Check that excluded sections are handled
        for section in context.excluded_sections:
            found = False
            for placement_list in policy.placement.values():
                if section in placement_list:
                    found = True
                    break
            if found:
                issues.append(f"Excluded section '{section}' is in placement for policy '{policy_name}'")
        
        return issues