import os
from typing import Dict, Any, Optional
from pydantic import BaseSettings, validator
import json
from datetime import timedelta

class Settings(BaseSettings):
    # App config
    app_name: str = "Biometric Fraud Prevention System"
    version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Database
    db_path: str = "data/biometrics.db"
    db_timeout: int = 30
    
    # Security
    secret_key: str = "your-secret-key-change-in-prod"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # ML
    model_path: str = "ml/models/fraud_model.pkl"
    contamination_rate: float = 0.1
    embedding_dim: int = 128
    fraud_threshold: float = 0.5
    
    # Logging
    log_level: str = "INFO"
    log_dir: str = "data/logs"
    max_log_size_mb: int = 10
    log_backup_count: int = 5
    
    # API
    api_prefix: str = "/api/v1"
    cors_origins: list = ["http://localhost:3000"]
    
    # Email/SMS (stubs)
    smtp_server: str = "smtp.example.com"
    smtp_port: int = 587
    email_from: str = "noreply@biometric-system.com"
    twilio_sid: str = ""
    twilio_token: str = ""
    
    # External integrations
    redis_url: str = "redis://localhost:6379"
    celery_broker: str = "redis://localhost:6379/0"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    @validator("cors_origins", pre=True)
    def assemble_cors_origins(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    @validator("secret_key")
    def secret_key_must_be_set(cls, v):
        if not v or v == "your-secret-key-change-in-prod":
            raise ValueError("secret_key must be set in environment or .env file")
        return v

# Load settings
settings = Settings()

# Extended config loader with JSON support
def load_config(file_path: str = "config/full_config.json") -> Dict[str, Any]:
    """Load additional config from JSON."""
    default_config = {
        "features": {
            "enable_mfa": True,
            "real_time_alerts": True,
            "audit_logging": True,
            "multi_tenant": False
        },
        "performance": {
            "max_concurrent_requests": 1000,
            "cache_ttl": 300,
            "batch_size": 100
        },
        "monitoring": {
            "metrics_enabled": True,
            "prometheus_port": 9090,
            "sentry_dsn": ""
        },
        "data_retention": {
            "alerts_days": 90,
            "biometrics_days": 365,
            "logs_days": 30
        }
    }
    
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
                default_config.update(config)
        except Exception as e:
            print(f"Config load warning: {e}")
    
    return default_config

# Global config
full_config = load_config()

# Validation functions
def validate_embedding_dim(dim: int) -> bool:
    """Validate embedding dimension."""
    return dim > 0 and dim <= 512

def validate_fraud_threshold(threshold: float) -> bool:
    """Validate fraud threshold."""
    return 0.0 < threshold < 1.0

# Env-specific settings
def get_env_specific_config(env: str = os.getenv("ENV", "development")) -> Dict:
    """Get environment-specific overrides."""
    configs = {
        "development": {
            "debug": True,
            "log_level": "DEBUG",
            "db_path": "data/dev_biometrics.db"
        },
        "production": {
            "debug": False,
            "log_level": "INFO",
            "db_path": "/var/app/data/prod_biometrics.db",
            "max_concurrent_requests": 5000
        },
        "testing": {
            "debug": True,
            "log_level": "WARNING",
            "db_path": ":memory:"
        }
    }
    return configs.get(env, {})

env_config = get_env_specific_config()
settings.dict().update(env_config)

# Export for use
__all__ = ["settings", "full_config", "validate_embedding_dim", "validate_fraud_threshold", "get_env_specific_config"]
