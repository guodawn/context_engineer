"""Tokenizer service for unified tokenization and length estimation."""

import re
from typing import Dict, Optional, Union, List
from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    """Abstract base class for tokenizers."""
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text."""
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (faster but less accurate)."""
        pass


class SimpleTokenizer(BaseTokenizer):
    """Simple tokenizer using regex-based word splitting."""
    
    def __init__(self, avg_chars_per_token: float = 4.0):
        self.avg_chars_per_token = avg_chars_per_token
        self.word_pattern = re.compile(r'\w+|[^\w\s]')
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using word-based splitting."""
        if not text:
            return 0
        words = self.word_pattern.findall(text)
        return len(words)
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count based on character length."""
        if not text:
            return 0
        return int(len(text) / self.avg_chars_per_token)


class TiktokenTokenizer(BaseTokenizer):
    """Tokenizer using tiktoken (if available)."""
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        try:
            import tiktoken
            self.encoding = tiktoken.get_encoding(encoding_name)
            self.available = True
        except ImportError:
            self.available = False
            self.fallback = SimpleTokenizer()
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken encoding."""
        if not self.available:
            return self.fallback.count_tokens(text)
        return len(self.encoding.encode(text))
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (same as count for tiktoken)."""
        return self.count_tokens(text)


class TokenizerService:
    """Unified tokenizer service supporting multiple backends."""
    
    def __init__(self, backend: str = "auto", **kwargs):
        """
        Initialize tokenizer service.
        
        Args:
            backend: Tokenizer backend ('simple', 'tiktoken', 'auto')
            **kwargs: Additional arguments for specific backends
        """
        if backend == "auto":
            self.tokenizer = TiktokenTokenizer(**kwargs)
            if not self.tokenizer.available:
                self.tokenizer = SimpleTokenizer(**kwargs)
        elif backend == "tiktoken":
            self.tokenizer = TiktokenTokenizer(**kwargs)
        elif backend == "simple":
            self.tokenizer = SimpleTokenizer(**kwargs)
        else:
            raise ValueError(f"Unknown tokenizer backend: {backend}")
    
    def count_tokens(self, text: Union[str, List[str], Dict[str, str]]) -> int:
        """
        Count tokens in text.
        
        Args:
            text: String, list of strings, or dict of strings
            
        Returns:
            Total token count
        """
        if isinstance(text, str):
            return self.tokenizer.count_tokens(text)
        elif isinstance(text, list):
            return sum(self.tokenizer.count_tokens(item) for item in text)
        elif isinstance(text, dict):
            total = 0
            for key, value in text.items():
                total += self.tokenizer.count_tokens(str(key))
                total += self.tokenizer.count_tokens(str(value))
            return total
        else:
            return self.tokenizer.count_tokens(str(text))
    
    def estimate_tokens(self, text: Union[str, List[str], Dict[str, str]]) -> int:
        """
        Estimate tokens in text (faster but less accurate).
        
        Args:
            text: String, list of strings, or dict of strings
            
        Returns:
            Estimated token count
        """
        if isinstance(text, str):
            return self.tokenizer.estimate_tokens(text)
        elif isinstance(text, list):
            return sum(self.tokenizer.estimate_tokens(item) for item in text)
        elif isinstance(text, dict):
            total = 0
            for key, value in text.items():
                total += self.tokenizer.estimate_tokens(str(key))
                total += self.tokenizer.estimate_tokens(str(value))
            return total
        else:
            return self.tokenizer.estimate_tokens(str(text))
    
    def count_tokens_with_breakdown(self, text: Dict[str, str]) -> Dict[str, int]:
        """
        Count tokens for each section in a dictionary.
        
        Args:
            text: Dictionary of text sections
            
        Returns:
            Dictionary with token counts for each section
        """
        breakdown = {}
        total = 0
        
        for key, value in text.items():
            count = self.count_tokens(value)
            breakdown[key] = count
            total += count
        
        breakdown["total"] = total
        return breakdown
    
    def get_tokenizer_info(self) -> Dict[str, Union[str, bool]]:
        """Get information about the current tokenizer."""
        info = {
            "backend": type(self.tokenizer).__name__,
            "available": True
        }
        
        if hasattr(self.tokenizer, 'available'):
            info["available"] = self.tokenizer.available
            if hasattr(self.tokenizer, 'encoding'):
                info["encoding_name"] = self.tokenizer.encoding.name
        
        return info