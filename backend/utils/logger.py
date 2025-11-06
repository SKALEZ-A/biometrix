import logging
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime
import os

class BiometricLogger:
    def __init__(self, name: str, log_dir: str = "data/logs"):
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{name}.log")
        
        # Formatter for structured logs
        json_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s", "extra": %(extra)s}'
        )
        
        # Rotating handler (max 10MB, keep 5 backups)
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setFormatter(json_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False
    
    def info(self, message: str, extra: dict = None):
        self.logger.info(message, extra=extra or {})
    
    def warning(self, message: str, extra: dict = None):
        self.logger.warning(message, extra=extra or {})
    
    def error(self, message: str, extra: dict = None):
        self.logger.error(message, extra=extra or {})
    
    def critical(self, message: str, extra: dict = None):
        self.logger.critical(message, extra=extra or {})

# Usage example and global loggers
app_logger = BiometricLogger("app")
db_logger = BiometricLogger("database")
ml_logger = BiometricLogger("ml")

# Extended logging functions with context
def log_biometric_event(event_type: str, user_id: str, details: dict):
    """Log biometric-specific events."""
    extra = {"event_type": event_type, "user_id": user_id, **details}
    app_logger.info(f"Biometric event: {event_type}", extra=extra)

def log_fraud_detection(user_id: str, score: float, threshold: float = 0.5):
    """Log fraud detection outcomes."""
    level = "warning" if score > threshold else "info"
    extra = {"user_id": user_id, "score": score, "threshold": threshold}
    if score > threshold:
        app_logger.warning("Fraud detected", extra=extra)
    else:
        app_logger.info("Normal verification", extra=extra)
