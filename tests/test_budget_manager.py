"""Tests for BudgetManager."""

import pytest
from context_engineer.core.budget_manager import BudgetManager, BucketConfig, BudgetAllocation
from context_engineer.core.tokenizer_service import TokenizerService


class TestBudgetManager:
    """Test cases for BudgetManager."""
    
    def test_bucket_config_creation(self):
        """Test creating bucket configuration."""
        bucket = BucketConfig(
            name="test_bucket",
            min_tokens=100,
            max_tokens=500,
            weight=1.5,
            sticky=True,
            compress="signature_only"
        )
        
        assert bucket.name == "test_bucket"
        assert bucket.min_tokens == 100
        assert bucket.max_tokens == 500
        assert bucket.weight == 1.5
        assert bucket.sticky is True
        assert bucket.compress == "signature_only"
    
    def test_budget_manager_initialization(self):
        """Test BudgetManager initialization."""
        manager = BudgetManager()
        assert manager.tokenizer is not None
        assert len(manager.buckets) == 0
        assert len(manager.drop_order) == 0
    
    def test_add_bucket(self):
        """Test adding a bucket to the manager."""
        manager = BudgetManager()
        bucket = BucketConfig(
            name="system",
            min_tokens=100,
            max_tokens=300,
            weight=2.0,
            sticky=True
        )
        
        manager.add_bucket(bucket)
        assert len(manager.buckets) == 1
        assert "system" in manager.buckets
        assert manager.buckets["system"] == bucket
    
    def test_configure_buckets(self):
        """Test configuring multiple buckets from dictionary."""
        manager = BudgetManager()
        
        bucket_configs = {
            "system": {
                "min_tokens": 100,
                "max_tokens": 300,
                "weight": 2.0,
                "sticky": True
            },
            "task": {
                "min_tokens": 50,
                "max_tokens": 200,
                "weight": 1.5
            }
        }
        
        manager.configure_buckets(bucket_configs)
        
        assert len(manager.buckets) == 2
        assert "system" in manager.buckets
        assert "task" in manager.buckets
        assert manager.buckets["system"].min_tokens == 100
        assert manager.buckets["task"].weight == 1.5
    
    def test_calculate_budget(self):
        """Test budget calculation."""
        manager = BudgetManager()
        
        model_limit = 8000
        output_budget = 1200
        system_overhead = 200
        
        available = manager.calculate_budget(model_limit, output_budget, system_overhead)
        
        expected = model_limit - output_budget - system_overhead
        assert available == expected
    
    def test_initial_allocation_normal_case(self):
        """Test initial allocation with sufficient budget."""
        manager = BudgetManager()
        
        # Add buckets
        manager.add_bucket(BucketConfig("system", 100, 300, 2.0))
        manager.add_bucket(BucketConfig("task", 50, 200, 1.5))
        manager.add_bucket(BucketConfig("tools", 30, 150, 1.0))
        
        available_budget = 500  # Sufficient for minimums
        allocations = manager._initial_allocation(available_budget)
        
        assert len(allocations) == 3
        
        # Check that minimums are met
        total_allocated = sum(alloc.allocated_tokens for alloc in allocations)
        total_min = sum(bucket.min_tokens for bucket in manager.buckets.values())
        
        assert total_allocated >= total_min
        assert total_allocated <= available_budget
    
    def test_initial_allocation_insufficient_budget(self):
        """Test initial allocation with insufficient budget."""
        manager = BudgetManager()
        
        # Add buckets with high minimums
        manager.add_bucket(BucketConfig("system", 300, 500, 2.0))
        manager.add_bucket(BucketConfig("task", 200, 400, 1.5))
        
        available_budget = 200  # Less than minimum required
        allocations = manager._initial_allocation(available_budget)
        
        assert len(allocations) == 2
        
        # Should allocate proportionally to minimums
        total_allocated = sum(alloc.allocated_tokens for alloc in allocations)
        assert total_allocated == available_budget
    
    def test_allocate_budget(self):
        """Test full budget allocation process."""
        manager = BudgetManager()
        
        # Configure buckets
        manager.configure_buckets({
            "system": {"min_tokens": 100, "max_tokens": 300, "weight": 2.0},
            "task": {"min_tokens": 50, "max_tokens": 200, "weight": 1.5},
            "history": {"min_tokens": 0, "max_tokens": 400, "weight": 1.0}
        })
        
        # Test allocation
        model_limit = 1000
        output_budget = 200
        content_scores = {
            "system": 0.8,
            "task": 0.9,
            "history": 0.6
        }
        
        allocations = manager.allocate_budget(model_limit, output_budget, content_scores)
        
        assert len(allocations) == 3
        
        # Check constraints
        total_allocated = sum(alloc.allocated_tokens for alloc in allocations)
        available_budget = manager.calculate_budget(model_limit, output_budget)
        
        assert total_allocated <= available_budget
        
        # Check individual allocations
        for allocation in allocations:
            bucket = manager.buckets[allocation.bucket_name]
            assert allocation.allocated_tokens >= bucket.min_tokens
            assert allocation.allocated_tokens <= bucket.max_tokens
    
    def test_handle_overflow(self):
        """Test overflow handling."""
        manager = BudgetManager()
        
        # Create allocations that exceed budget
        allocations = [
            BudgetAllocation("system", 200, 2.0),
            BudgetAllocation("task", 150, 1.5),
            BudgetAllocation("tools", 100, 1.0)
        ]
        
        # Configure buckets with droppable flag
        manager.add_bucket(BucketConfig("system", 50, 300, 2.0, sticky=True))
        manager.add_bucket(BucketConfig("task", 30, 200, 1.5))
        manager.add_bucket(BucketConfig("tools", 20, 150, 1.0, droppable=True))
        
        available_budget = 300  # Less than total allocated
        result = manager._handle_overflow(allocations, available_budget)
        
        total_after = sum(alloc.allocated_tokens for alloc in result)
        assert total_after <= available_budget
    
    def test_set_drop_order(self):
        """Test setting drop order."""
        manager = BudgetManager()
        drop_order = ["fewshot", "history", "tools", "rag"]
        
        manager.set_drop_order(drop_order)
        assert manager.drop_order == drop_order
    
    def test_validate_configuration(self):
        """Test configuration validation."""
        manager = BudgetManager()
        
        # Valid configuration
        manager.add_bucket(BucketConfig("system", 100, 300, 2.0))
        manager.add_bucket(BucketConfig("task", 50, 200, 1.5))
        manager.set_drop_order(["task", "system"])
        
        assert manager.validate_configuration() is True
        
        # Invalid configuration - min > max
        manager.buckets.clear()
        manager.add_bucket(BucketConfig("system", 300, 100, 2.0))  # Invalid
        
        assert manager.validate_configuration() is False
    
    def test_get_bucket_config(self):
        """Test getting bucket configuration."""
        manager = BudgetManager()
        bucket = BucketConfig("system", 100, 300, 2.0)
        manager.add_bucket(bucket)
        
        retrieved = manager.get_bucket_config("system")
        assert retrieved == bucket
        
        missing = manager.get_bucket_config("nonexistent")
        assert missing is None
    
    def test_get_total_min_max_tokens(self):
        """Test getting total minimum and maximum tokens."""
        manager = BudgetManager()
        
        manager.add_bucket(BucketConfig("system", 100, 300, 2.0))
        manager.add_bucket(BucketConfig("task", 50, 200, 1.5))
        manager.add_bucket(BucketConfig("tools", 30, 150, 1.0))
        
        total_min = manager.get_total_min_tokens()
        total_max = manager.get_total_max_tokens()
        
        assert total_min == 180  # 100 + 50 + 30
        assert total_max == 650  # 300 + 200 + 150