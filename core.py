"""Global configuration for Super AI Scraper"""
from datetime import datetime

# Current timestamp and user
TIMESTAMP = "2025-07-14 17:25:34"
USER = "Muhammad-Fauzan22"

# System configuration
CONFIG = {
    "max_retries": 3,
    "batch_size": 1024 * 1024,  # 1MB
    "timeout": 30,
    "workers": 4,
    "max_targets": 20
}

# AI Configuration
AI_CONFIG = {
    "DEEPSEEK": {
        "endpoint": "https://api.deepseek.com/v1/chat/completions",
        "model": "deepseek-chat",
        "key": "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"
    },
    "CLAUDE": {
        "endpoint": "https://api.anthropic.com/v1/messages",
        "model": "claude-3-opus-20240229",
        "key": "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"
    },
    "PERPLEXITY": {
        "endpoint": "https://api.perplexity.ai/chat/completions",
        "model": "pplx-70b-online",
        "key": "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"
    },
    "CYPHER": {
        "endpoint": "https://api.cypher.ai/v1/completions",
        "model": "cypher-alpha",
        "key": "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"
    },
    "GEMMA": {
        "endpoint": "https://api.gemma.ai/v1/generate",
        "model": "gemma-7b-it",
        "key": "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"
    }
}

# Alert thresholds
ALERT_THRESHOLDS = {
    "cpu_percent": 80,
    "memory_percent": 85,
    "disk_usage": 90,
    "healing_ratio": 0.3
}

# Auto-scaling configuration
SCALING_PROFILE = {
    "min_workers": 2,
    "max_workers": 8,
    "scale_up_threshold": 70,  # CPU usage %
    "scale_down_threshold": 30
}
