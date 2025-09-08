"""Tests for ContextAssembler."""

import pytest
from context_engineer.core.context_assembler import ContextAssembler, ContextSection, AssembledContext
from context_engineer.core.budget_manager import BudgetAllocation
from context_engineer.core.tokenizer_service import TokenizerService


class TestContextAssembler:
    """Test cases for ContextAssembler."""
    
    def test_context_section_creation(self):
        """Test creating ContextSection."""
        section = ContextSection(
            name="system",
            content="You are a helpful assistant.",
            priority=2.0,
            placement="head",
            compression_needed=False
        )
        
        assert section.name == "system"
        assert section.content == "You are a helpful assistant."
        assert section.priority == 2.0
        assert section.placement == "head"
        assert section.compression_needed is False
    
    def test_context_assembler_initialization(self):
        """Test ContextAssembler initialization."""
        assembler = ContextAssembler()
        assert assembler.tokenizer is not None
        assert assembler.placement_strategy == {"head": [], "middle": [], "tail": []}
    
    def test_assemble_context_basic(self):
        """Test basic context assembly."""
        assembler = ContextAssembler()
        
        content_sections = {
            "system": "You are a helpful assistant.",
            "task": "Help the user with their question.",
            "history": "Previous conversation about weather."
        }
        
        budget_allocations = [
            BudgetAllocation("system", 50, 2.0),
            BudgetAllocation("task", 40, 1.5),
            BudgetAllocation("history", 30, 1.0)
        ]
        
        placement_policy = {
            "head": ["system", "task"],
            "middle": ["history"],
            "tail": []
        }
        
        result = assembler.assemble_context(content_sections, budget_allocations, placement_policy)
        
        assert isinstance(result, AssembledContext)
        assert result.full_context is not None
        assert len(result.sections) == 3
        assert result.total_tokens > 0
        assert "head" in result.placement_map
        assert "middle" in result.placement_map
        assert "tail" in result.placement_map
    
    def test_apply_placement_policy(self):
        """Test applying placement policy to sections."""
        assembler = ContextAssembler()
        
        sections = [
            ContextSection("system", "System prompt", 2.0),
            ContextSection("task", "Task description", 1.5),
            ContextSection("history", "Conversation history", 1.0)
        ]
        
        placement_policy = {
            "head": ["system", "task"],
            "middle": ["history"],
            "tail": []
        }
        
        assembler._apply_placement_policy(sections, placement_policy)
        
        # Check that sections have correct placement
        system_section = next(s for s in sections if s.name == "system")
        task_section = next(s for s in sections if s.name == "task")
        history_section = next(s for s in sections if s.name == "history")
        
        assert system_section.placement == "head"
        assert task_section.placement == "head"
        assert history_section.placement == "middle"
    
    def test_sort_sections(self):
        """Test section sorting by placement and priority."""
        assembler = ContextAssembler()
        
        sections = [
            ContextSection("history", "History", 1.0, "middle"),
            ContextSection("system", "System", 2.0, "head"),
            ContextSection("task", "Task", 1.5, "head"),
            ContextSection("scratchpad", "Scratchpad", 0.5, "tail")
        ]
        
        sorted_sections = assembler._sort_sections(sections)
        
        # Should be ordered: head -> middle -> tail
        # Within same placement, higher priority first
        assert sorted_sections[0].name == "system"  # head, priority 2.0
        assert sorted_sections[1].name == "task"    # head, priority 1.5
        assert sorted_sections[2].name == "history" # middle, priority 1.0
        assert sorted_sections[3].name == "scratchpad" # tail, priority 0.5
    
    def test_apply_token_limits(self):
        """Test applying token limits and compression."""
        assembler = ContextAssembler()
        
        sections = [
            ContextSection("long_section", "This is a very long section that needs compression. " * 20, 1.0, compression_needed=True),
            ContextSection("short_section", "Short content", 1.0, compression_needed=False)
        ]
        
        # Set token counts
        for section in sections:
            section.token_count = assembler.tokenizer.count_tokens(section.content)
        
        processed_sections = assembler._apply_token_limits(sections)
        
        assert len(processed_sections) == 2
        
        # Long section should be compressed
        long_section = next(s for s in processed_sections if s.name == "long_section")
        assert long_section.token_count < sections[0].token_count
        
        # Short section should remain unchanged
        short_section = next(s for s in processed_sections if s.name == "short_section")
        assert short_section.content == "Short content"
    
    def test_build_context(self):
        """Test building final context string."""
        assembler = ContextAssembler()
        
        sections = [
            ContextSection("system", "System prompt", 2.0, "head"),
            ContextSection("task", "Task description", 1.5, "head"),
            ContextSection("history", "History content", 1.0, "middle")
        ]
        
        context, placement_map, dropped = assembler._build_context(sections)
        
        assert context is not None
        assert len(context) > 0
        assert "system" in placement_map["head"]
        assert "task" in placement_map["head"]
        assert "history" in placement_map["middle"]
        assert len(dropped) == 0
    
    def test_apply_lost_in_middle_mitigation(self):
        """Test applying Lost in the Middle mitigation strategies."""
        assembler = ContextAssembler()
        
        content_sections = {
            "system": "System instructions",
            "task": "Task description",
            "evidence": "Supporting evidence and documentation"
        }
        
        key_sections = ["system", "task"]
        
        mitigated = assembler.apply_lost_in_middle_mitigation(content_sections, key_sections)
        
        # Key sections should have emphasis markers
        assert mitigated["system"].startswith("IMPORTANT:")
        assert mitigated["task"].startswith("IMPORTANT:")
        
        # Non-key sections should remain unchanged
        assert mitigated["evidence"] == "Supporting evidence and documentation"
    
    def test_create_excerpts_with_summary(self):
        """Test creating excerpts from long content."""
        assembler = ContextAssembler()
        
        long_content = "Paragraph 1. " * 10 + "\n\n" + "Paragraph 2. " * 10 + "\n\n" + "Paragraph 3. " * 10
        
        excerpt, summary = assembler.create_excerpts_with_summary(long_content, excerpt_ratio=0.5)
        
        assert excerpt is not None
        assert "... [content truncated] ..." in excerpt
        assert summary is not None
        assert "paragraphs" in summary.lower()
    
    def test_get_context_stats(self):
        """Test getting context statistics."""
        assembler = ContextAssembler()
        
        # Create a mock assembled context
        sections = [
            ContextSection("system", "System", 2.0, "head", token_count=10),
            ContextSection("task", "Task", 1.5, "head", token_count=8),
            ContextSection("history", "History", 1.0, "middle", token_count=15)
        ]
        
        mock_context = AssembledContext(
            full_context="System\n\nTask\n\nHistory",
            sections=sections,
            total_tokens=33,
            placement_map={"head": ["system", "task"], "middle": ["history"], "tail": []},
            dropped_sections=[]
        )
        
        stats = assembler.get_context_stats(mock_context)
        
        assert stats["total_tokens"] == 33
        assert stats["total_sections"] == 3
        assert stats["sections_by_placement"]["head"] == 2
        assert stats["sections_by_placement"]["middle"] == 1
        assert stats["sections_by_placement"]["tail"] == 0
        assert len(stats["section_details"]) == 3
    
    def test_empty_content_handling(self):
        """Test handling of empty content sections."""
        assembler = ContextAssembler()
        
        content_sections = {
            "system": "System prompt",
            "empty_section": "",  # Empty content
            "task": "Task description"
        }
        
        budget_allocations = [
            BudgetAllocation("system", 30, 2.0),
            BudgetAllocation("empty_section", 0, 1.0),  # Zero allocation
            BudgetAllocation("task", 25, 1.5)
        ]
        
        result = assembler.assemble_context(content_sections, budget_allocations)
        
        assert result.full_context is not None
        # Empty section should be handled gracefully
        assert len(result.sections) >= 2  # At least system and task
    
    def test_compression_needed_flag(self):
        """Test that compression_needed flag is properly handled."""
        assembler = ContextAssembler()
        
        # Create sections with compression needed
        sections = [
            ContextSection("long_section", "This is very long content " * 50, 1.0, compression_needed=True),
            ContextSection("normal_section", "Normal content", 1.0, compression_needed=False)
        ]
        
        # Set initial token counts
        for section in sections:
            section.token_count = assembler.tokenizer.count_tokens(section.content)
        
        processed = assembler._apply_token_limits(sections)
        
        # Long section should be compressed
        long_processed = next(s for s in processed if s.name == "long_section")
        assert long_processed.token_count < sections[0].token_count
        
        # Normal section should remain the same
        normal_processed = next(s for s in processed if s.name == "normal_section")
        assert normal_processed.content == "Normal content"