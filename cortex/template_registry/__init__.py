"""Template Registry System for Cortex.

This module provides intelligent template management for different model types,
handling various chat formats and output processing requirements.
"""

from cortex.template_registry.registry import TemplateRegistry
from cortex.template_registry.auto_detector import TemplateDetector
from cortex.template_registry.template_profiles.base import BaseTemplateProfile

__all__ = [
    'TemplateRegistry',
    'TemplateDetector',
    'BaseTemplateProfile'
]