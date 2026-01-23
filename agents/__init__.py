"""
Multi-Agent Backend System for Financial Data Extraction

This package contains all agent implementations:
- Master Orchestrator Agent
- Ingestion Agent (with sub-agents)
- Query Agent (with sub-agents)
- Maintenance Agent (with sub-agents)
"""

from .base_agent import BaseAgent

__all__ = ['BaseAgent']
