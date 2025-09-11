"""Context assembler for position-aware message assembly."""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from ..core.tokenizer_service import TokenizerService
from ..core.budget_manager import BudgetManager, BudgetAllocation


@dataclass
class ContextSection:
    """Represents a section of context with metadata."""
    name: str
    content: str
    priority: float
    placement: str = "middle"
    token_count: int = 0
    allocated_tokens: int = 0  # 预算分配的token数量，用于精确压缩目标
    # 移除compression_needed字段，通过比较token_count和allocated_tokens判断是否需要压缩


@dataclass
class AssembledContext:
    """Result of context assembly."""
    full_context: str
    sections: List[ContextSection]
    total_tokens: int
    placement_map: Dict[str, List[str]]
    dropped_sections: List[str]


class ContextAssembler:
    """Assembles context according to position strategies to avoid "Lost in the Middle"."""
    
    def __init__(self, tokenizer_service: Optional[TokenizerService] = None, compressor_service: Optional['Compressor'] = None):
        """
        Initialize context assembler.
        
        Args:
            tokenizer_service: TokenizerService instance for token counting
            compressor_service: Compressor service for content compression
        """
        self.tokenizer = tokenizer_service or TokenizerService()
        self.compressor = compressor_service
        self.placement_strategy = {
            "head": [],
            "middle": [],
            "tail": []
        }
    
    def assemble_context(self,
                        content_sections: Dict[str, str],
                        budget_allocations: List[BudgetAllocation],
                        placement_policy: Optional[Dict[str, List[str]]] = None,
                        bucket_configs: Optional[Dict[str, Any]] = None) -> AssembledContext:
        """
        Assemble context sections according to placement strategy.
        
        Args:
            content_sections: Dictionary of section names to content
            budget_allocations: Budget allocations for each section
            placement_policy: Placement policy (head/middle/tail positions)
            bucket_configs: Optional bucket configurations to access compression methods
            
        Returns:
            AssembledContext with position-aware arrangement
        """
        # Create context sections with metadata
        sections = self._create_sections(content_sections, budget_allocations)
        
        # Apply placement policy
        if placement_policy:
            self._apply_placement_policy(sections, placement_policy)
        
        # Sort sections by placement and priority
        sections = self._sort_sections(sections)
        
        # Apply token limits and compression
        sections = self._apply_token_limits(sections, bucket_configs)
        
        # Build final context with position strategy
        assembled_context, placement_map, dropped_sections = self._build_context(sections)
        
        # Calculate total tokens
        total_tokens = self.tokenizer.count_tokens(assembled_context)
        
        return AssembledContext(
            full_context=assembled_context,
            sections=sections,
            total_tokens=total_tokens,
            placement_map=placement_map,
            dropped_sections=dropped_sections
        )
    
    def _create_sections(self,
                        content_sections: Dict[str, str],
                        budget_allocations: List[BudgetAllocation]) -> List[ContextSection]:
        """Create context sections with metadata from budget allocations."""
        sections = []
        allocation_map = {alloc.bucket_name: alloc for alloc in budget_allocations}
        
        for section_name, content in content_sections.items():
            allocation = allocation_map.get(section_name)
            if not allocation:
                continue
            
            section = ContextSection(
                name=section_name,
                content=content,
                priority=allocation.priority,
                token_count=self.tokenizer.count_tokens(content),
                allocated_tokens=allocation.allocated_tokens  # 传递预算分配的token数量
            )
            sections.append(section)
        
        return sections
    
    def _apply_placement_policy(self,
                               sections: List[ContextSection],
                               placement_policy: Dict[str, List[str]]):
        """Apply placement policy to determine section positions."""
        for position, section_names in placement_policy.items():
            for section_name in section_names:
                for section in sections:
                    if section.name == section_name:
                        section.placement = position
                        break
    
    def _sort_sections(self, sections: List[ContextSection]) -> List[ContextSection]:
        """Sort sections by placement and priority."""
        # Define placement priority order
        placement_order = {"head": 0, "middle": 1, "tail": 2}
        
        return sorted(sections, key=lambda s: (placement_order.get(s.placement, 1), -s.priority))
    
    def _apply_token_limits(self, sections: List[ContextSection], bucket_configs: Optional[Dict[str, Any]] = None) -> List[ContextSection]:
        """Apply token limits and compression to sections.
        
        通过比较token_count和allocated_tokens来判断是否需要压缩，
        完全移除了compression_needed标记的依赖。
        """
        processed_sections = []
        
        for section in sections:
            # 通过比较实际token数和分配预算来判断是否需要压缩
            if section.token_count > section.allocated_tokens and section.token_count > 0:
                # 内容超过预算，需要压缩
                # Determine compression method from bucket configuration
                compression_method = None
                if bucket_configs and section.name in bucket_configs:
                    bucket_config = bucket_configs[section.name]
                    compression_method = getattr(bucket_config, 'compress', None)
                
                # Apply compression with specified method
                compressed_content = self._compress_section(section, compression_method)
                section.content = compressed_content
                section.token_count = self.tokenizer.count_tokens(compressed_content)
            # 如果内容符合预算或为空，无需任何处理
            
            processed_sections.append(section)
        
        return processed_sections
    
    def _compress_section(self, section: ContextSection, compression_method: Optional[str] = None) -> str:
        """Apply compression to a section based on its needs and configuration.
        
        完全按照allocated_tokens进行精确压缩，移除50%启发式规则。
        """
        if section.token_count <= 0:
            return section.content
        
        # 精确压缩：直接按照预算分配的allocated_tokens作为目标
        target_tokens = section.allocated_tokens
        
        # 如果实际内容已经符合预算要求，无需压缩
        if section.token_count <= target_tokens:
            return section.content
            
        # 如果压缩机服务可用，使用指定方法进行精确压缩
        if self.compressor and compression_method:
            try:
                result = self.compressor.compress(
                    content=section.content,
                    target_tokens=target_tokens,
                    method=compression_method
                )
                return result.compressed_content
            except Exception as e:
                # 压缩方法失败，回退到简单截断
                print(f"Warning: Compression method '{compression_method}' failed: {e}")
        
        # 回退到简单截断，精确到allocated_tokens
        from ..utils.token_utils import truncate_to_tokens
        return truncate_to_tokens(section.content, target_tokens, self.tokenizer)
    
    def _build_context(self, sections: List[ContextSection]) -> Tuple[str, Dict[str, List[str]], List[str]]:
        """Build final context string with position strategy."""
        placement_map = {"head": [], "middle": [], "tail": []}
        dropped_sections = []
        context_parts = []
        
        # Group sections by placement
        for section in sections:
            if section.content:  # Only include non-empty sections
                placement_map[section.placement].append(section.name)
            else:
                dropped_sections.append(section.name)
        
        # Build context in order: head -> middle -> tail
        for placement in ["head", "middle", "tail"]:
            for section_name in placement_map[placement]:
                section = next(s for s in sections if s.name == section_name)
                if section.content:
                    context_parts.append(section.content)
        
        # Join with appropriate separators
        assembled_context = self._join_context_parts(context_parts)
        
        return assembled_context, placement_map, dropped_sections
    
    def _join_context_parts(self, context_parts: List[str]) -> str:
        """Join context parts with appropriate separators."""
        if not context_parts:
            return ""
        
        # Use double newlines to separate major sections
        separator = "\n\n"
        return separator.join(part.strip() for part in context_parts if part.strip())
    
    def apply_lost_in_middle_mitigation(self,
                                       content_sections: Dict[str, str],
                                       key_sections: List[str]) -> Dict[str, str]:
        """
        Apply strategies to mitigate "Lost in the Middle" effect.
        
        Args:
            content_sections: Original content sections
            key_sections: List of section names that should be prioritized
            
        Returns:
            Modified content sections with mitigation applied
        """
        mitigated_sections = content_sections.copy()
        
        # Strategy 1: Place key sections at the beginning or end
        for section_name in key_sections:
            if section_name in mitigated_sections:
                content = mitigated_sections[section_name]
                
                # Add emphasis markers for key sections
                if not content.startswith("IMPORTANT:"):
                    mitigated_sections[section_name] = f"IMPORTANT: {content}"
        
        return mitigated_sections
    
    def create_excerpts_with_summary(self,
                                   long_content: str,
                                   excerpt_ratio: float = 0.3) -> Tuple[str, str]:
        """
        Create excerpts from long content with summary.
        
        Args:
            long_content: Long content to excerpt
            excerpt_ratio: Ratio of content to excerpt (head + tail)
            
        Returns:
            Tuple of (excerpt, summary)
        """
        total_tokens = self.tokenizer.count_tokens(long_content)
        target_excerpt_tokens = int(total_tokens * excerpt_ratio)
        
        # Split content into paragraphs
        paragraphs = long_content.split('\n\n')
        
        # Take beginning and end portions
        head_tokens = target_excerpt_tokens // 2
        tail_tokens = target_excerpt_tokens // 2
        
        # Build head excerpt
        head_excerpt = ""
        head_token_count = 0
        for paragraph in paragraphs:
            para_tokens = self.tokenizer.count_tokens(paragraph)
            if head_token_count + para_tokens <= head_tokens:
                head_excerpt += paragraph + "\n\n"
                head_token_count += para_tokens
            else:
                break
        
        # Build tail excerpt
        tail_excerpt = ""
        tail_token_count = 0
        for paragraph in reversed(paragraphs):
            para_tokens = self.tokenizer.count_tokens(paragraph)
            if tail_token_count + para_tokens <= tail_tokens:
                tail_excerpt = paragraph + "\n\n" + tail_excerpt
                tail_token_count += para_tokens
            else:
                break
        
        # Combine excerpts
        excerpt = head_excerpt.strip() + "\n\n... [content truncated] ...\n\n" + tail_excerpt.strip()
        
        # Create summary (placeholder - can be enhanced with actual summarization)
        summary = f"Content summary: {len(paragraphs)} paragraphs, {total_tokens} tokens total."
        
        return excerpt, summary
    
    def get_context_stats(self, assembled_context: AssembledContext) -> Dict[str, Any]:
        """Get statistics about the assembled context."""
        stats = {
            "total_tokens": assembled_context.total_tokens,
            "total_sections": len(assembled_context.sections),
            "sections_by_placement": {
                placement: len(sections)
                for placement, sections in assembled_context.placement_map.items()
            },
            "dropped_sections": len(assembled_context.dropped_sections),
            "section_details": []
        }
        
        for section in assembled_context.sections:
            # 通过比较token_count和allocated_tokens来判断是否需要压缩
            needs_compression = section.token_count > section.allocated_tokens
            stats["section_details"].append({
                "name": section.name,
                "tokens": section.token_count,
                "placement": section.placement,
                "priority": section.priority,
                "allocated_tokens": section.allocated_tokens,
                "needs_compression": needs_compression,
                "budget_utilization": section.token_count / section.allocated_tokens if section.allocated_tokens > 0 else 0
            })
        
        return stats