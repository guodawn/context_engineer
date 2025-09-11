# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ContextEngineer is a system for designing, building, and optimizing dynamic automation systems that provide large language models with the right information and tools at the right time to reliably and scalably complete complex tasks.

## Architecture

The system implements a token budgeting and context optimization framework with these key components:

1. **Context Architecture Definition**: SystemPrompt / UserPrompt / History / Tools / Memory / RAG
2. **Token Budget Management**: Proportional token allocation across 8 semantic buckets
3. **Context Injection Engine**: Unified context assembly logic
4. **Dynamic Optimization**: Select/Compress/Rerank pipeline for context optimization

## Key Technical Components

- **TokenizerService**: Unified tokenization and length estimation
- **BudgetManager**: Token allocation and fallback management
- **Compressor**: Multi-level filtering/extraction/summarization pipeline
- **ContextAssembler**: Position-aware message assembly (head/middle/tail)
- **Policy Engine**: Task-specific budget strategy templates

## Context Buckets

The system divides context into 8 buckets with configurable min/max/weight parameters:
- System & Safety Constraints
- Task Instructions
- Tools & Schema
- History (summarizable)
- Long-term Memory
- RAG Evidence
- Few-shot Examples
- Scratchpad

## Development Notes

This appears to be a documentation/specification repository for a context engineering system. When implementing code:
- Follow the token budgeting algorithm outlined in context_engineer.md
- Implement the 8-bucket context architecture
- Support both static quotas and dynamic ROI-based allocation
- Ensure "Lost in the Middle" mitigation through strategic positioning
- 请记住使用使用 python 虚拟环境运行代码，虚拟环境位于项目根目录venv