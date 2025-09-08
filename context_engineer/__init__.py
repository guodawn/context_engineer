"""
ContextEngineer: A system for optimizing context management in large language models.

This package provides tools for token budgeting, context optimization, and dynamic
information retrieval to maximize signal density while minimizing noise in LLM contexts.
"""

__version__ = "0.1.0"
__author__ = "ContextEngineer Team"

from .core.budget_manager import BudgetManager
from .core.tokenizer_service import TokenizerService
from .core.context_assembler import ContextAssembler
from .services.compressor import Compressor
from .services.policy_engine import PolicyEngine
from .config.settings import ContextConfig

__all__ = [
    "BudgetManager",
    "TokenizerService", 
    "ContextAssembler",
    "Compressor",
    "PolicyEngine",
    "ContextConfig"
]