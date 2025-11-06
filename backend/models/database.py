import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class BiometricDatabase:
    def __init__(self, db_path: str = "data/biometrics.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.init_db()
    
    def init_db(self):
        """Initialize database schema with tables for users, biometrics, alerts."""
        cursor = self.conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                enrolled BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Biometrics table (embeddings as JSON for simplicity)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS biometrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                face_embedding TEXT,  -- JSON serialized list[float]
                fingerprint_hash TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                score REAL NOT NULL,
                reason TEXT,
                severity TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_biometrics_user ON biometrics(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_user ON alerts(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)')
        
        self.conn.commit()
        logger.info("Database initialized")
    
    def create_user(self, user_id: str, name: str, email: str) -> bool:
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO users (id, name, email) VALUES (?, ?, ?)",
                (user_id, name, email)
            )
            self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.IntegrityError as e:
            logger.error(f"User creation error: {e}")
            return False
    
    def get_user(self, user_id: str) -> Optional[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        if row:
            return {
                "id": row[0], "name": row[1], "email": row[2],
                "enrolled": bool(row[3]), "created_at": row[4]
            }
        return None
    
    def enroll_biometric(self, user_id: str, face_embedding: List[float], fingerprint_hash: str) -> bool:
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO biometrics (user_id, face_embedding, fingerprint_hash) VALUES (?, ?, ?)",
                (user_id, json.dumps(face_embedding), fingerprint_hash)
            )
            # Update user enrolled status
            cursor.execute("UPDATE users SET enrolled = TRUE WHERE id = ?", (user_id,))
            self.conn.commit()
            logger.info(f"Biometric enrolled for {user_id}")
            return True
        except Exception as e:
            logger.error(f"Enrollment error: {e}")
            return False
    
    def detect_fraud(self, user_id: str, input_embedding: List[float]) -> Dict:
        cursor = self.conn.cursor()
        cursor.execute("SELECT face_embedding FROM biometrics WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1", (user_id,))
        row = cursor.fetchone()
        if not row:
            return {"fraud_detected": True, "reason": "No enrollment"}
        
        stored_embedding = json.loads(row[0])
        # Simple Euclidean distance for anomaly (extendable to cosine/ML)
        distance = sum((a - b) ** 2 for a, b in zip(input_embedding, stored_embedding[:len(input_embedding)])) ** 0.5
        fraud_score = min(distance / len(input_embedding), 1.0)
        
        if fraud_score > 0.5:  # Threshold
            cursor.execute(
                "INSERT INTO alerts (user_id, score, reason, severity) VALUES (?, ?, ?, ?)",
                (user_id, fraud_score, "Anomaly detected", "high" if fraud_score > 0.8 else "medium")
            )
            self.conn.commit()
            return {"fraud_detected": True, "score": fraud_score, "distance": distance}
        
        return {"fraud_detected": False, "score": fraud_score}
    
    def get_alerts(self, user_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
        cursor = self.conn.cursor()
        if user_id:
            cursor.execute("SELECT * FROM alerts WHERE user_id = ? ORDER BY created_at DESC LIMIT ?", (user_id, limit))
        else:
            cursor.execute("SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?", (limit,))
        
        alerts = []
        for row in cursor.fetchall():
            alerts.append({
                "id": row[0], "user_id": row[1], "score": row[2], "reason": row[3],
                "severity": row[4], "created_at": row[5]
            })
        return alerts
    
    def close(self):
        self.conn.close()

# Migration stub for schema updates
def run_migrations(db: BiometricDatabase):
    """Stub for future migrations (e.g., add columns)."""
    logger.info("Running migrations - schema up to date.")
    pass

# Global instance
db = BiometricDatabase()
