from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import logging
from datetime import datetime
import json
import os

# Configure logging for production-grade monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Biometric Fraud Prevention API", version="1.0.0", description="Secure biometric fraud detection system")

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class UserBiometric(BaseModel):
    user_id: str
    face_embedding: List[float]  # Simulated 128-dim vector
    fingerprint_hash: str
    timestamp: datetime

class FraudAlert(BaseModel):
    user_id: str
    score: float
    reason: str
    severity: str  # low/medium/high

class User(BaseModel):
    id: str
    name: str
    email: str
    enrolled: bool = False

# In-memory storage for demo (replace with DB in prod)
users_db = {}
biometrics_db = {}
alerts_db = []

# Dependency for auth (simplified JWT stub)
def get_current_user(user_id: str):
    if user_id not in users_db:
        raise HTTPException(status_code=401, detail="Invalid user")
    return users_db[user_id]

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Biometric Fraud Prevention System API", "status": "running"}

# User management endpoints
@app.post("/users/", response_model=User)
async def create_user(user: User):
    if user.id in users_db:
        raise HTTPException(status_code=400, detail="User exists")
    users_db[user.id] = user.dict()
    logger.info(f"Created user {user.id}")
    return user

@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: str):
    user = users_db.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# Biometric enrollment
@app.post("/biometrics/enroll")
async def enroll_biometric(bio: UserBiometric, current_user: str = Depends(get_current_user)):
    if bio.user_id != current_user:
        raise HTTPException(status_code=403, detail="Unauthorized")
    biometrics_db[bio.user_id] = bio.dict()
    logger.info(f"Enrolled biometrics for {bio.user_id}")
    return {"status": "enrolled", "user_id": bio.user_id}

# Fraud detection endpoint (simulates ML inference)
@app.post("/fraud/detect")
async def detect_fraud(bio: UserBiometric, current_user: Optional[str] = None):
    # Simulate fraud score calculation (cosine similarity stub)
    stored = biometrics_db.get(bio.user_id)
    if not stored:
        raise HTTPException(status_code=404, detail="No enrolled biometrics")
    
    # Dummy anomaly detection logic
    similarity = sum(a * b for a, b in zip(bio.face_embedding[:10], stored['face_embedding'][:10])) / (len(bio.face_embedding) or 1)
    fraud_score = 1.0 - similarity  # Inverse for fraud likelihood
    
    if fraud_score > 0.7:
        alert = FraudAlert(user_id=bio.user_id, score=fraud_score, reason="Biometric mismatch", severity="high")
        alerts_db.append(alert.dict())
        logger.warning(f"Fraud alert for {bio.user_id}: score {fraud_score}")
        return {"fraud_detected": True, "alert": alert.dict()}
    
    return {"fraud_detected": False, "score": fraud_score}

# Get alerts
@app.get("/alerts/", response_model=List[FraudAlert])
async def get_alerts(limit: int = 100):
    return alerts_db[-limit:]

# Health check
@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Run server (for dev)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
