#!/usr/bin/env python3
import os
import subprocess
import sys
import yaml
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeploymentManager:
    def __init__(self, config_path: str = "deploy/config.yaml"):
        self.config = self._load_config(config_path)
        self.workspace = Path("/Users/mac/Desktop/CODES/LUMEN_CODES/2-NOV/biometric-fraud-prevention-system")
    
    def _load_config(self, path: str) -> Dict:
        """Load deployment config."""
        default_config = {
            "docker": {
                "image_name": "biometric-fraud-system",
                "tag": "latest",
                "ports": {"backend": "8000:8000", "frontend": "3000:3000"},
                "volumes": [".:/app"]
            },
            "cloud": {
                "provider": "aws",  # or gcp, azure
                "region": "us-east-1",
                "instance_type": "t3.medium"
            },
            "ci_cd": {
                "github_actions": True,
                "gitlab_ci": False
            },
            "environment": {
                "dev": {"debug": True},
                "prod": {"debug": False, "db_path": "/var/app/data/db.sqlite"}
            }
        }
        
        if os.path.exists(path):
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
                default_config.update(config)
        
        logger.info("Deployment config loaded")
        return default_config
    
    def create_docker_compose(self, env: str = "dev") -> None:
        """Generate docker-compose.yml."""
        compose_content = f"""version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "{self.config['docker']['ports']['backend']}"
    volumes:
      - .:/app
    environment:
      - ENV={env}
      - DEBUG={self.config['environment'][env]['debug']}
    depends_on:
      - redis
      - db
  
  frontend:
    build: ./frontend
    ports:
      - "{self.config['docker']['ports']['frontend']}"
    volumes:
      - .:/app
    depends_on:
      - backend
  
  db:
    image: sqlite  # Custom or postgres
    volumes:
      - data/biometrics.db:/data/db.sqlite
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
  
  celery:
    build: ./backend
    command: celery -A tasks worker --loglevel=info
    volumes:
      - .:/app
    depends_on:
      - redis
      - backend
"""
        
        with open("docker-compose.yml", "w") as f:
            f.write(compose_content)
        
        # Create Dockerfile for backend
        backend_dockerfile = """FROM python:3.9-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        os.makedirs("backend/Dockerfile", exist_ok=True)
        with open("backend/Dockerfile", "w") as f:
            f.write(backend_dockerfile)
        
        logger.info("Docker Compose and Dockerfiles generated")
    
    def build_docker_images(self) -> None:
        """Build Docker images."""
        cmd = ["docker-compose", "build"]
        subprocess.run(cmd, cwd=self.workspace, check=True)
        logger.info("Docker images built successfully")
    
    def deploy_local(self) -> None:
        """Deploy locally with docker-compose."""
        self.create_docker_compose()
        self.build_docker_images()
        
        cmd = ["docker-compose", "up", "-d"]
        subprocess.run(cmd, cwd=self.workspace, check=True)
        logger.info("Local deployment started")
    
    def generate_ci_config(self, platform: str = "github") -> None:
        """Generate CI/CD config."""
        if platform == "github":
            ci_content = """name: CI/CD Pipeline
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -r backend/requirements.txt
        cd frontend && npm install
    - name: Run backend tests
      run: pytest backend/tests/
    - name: Run frontend tests
      run: cd frontend && npm test
    - name: Build Docker
      run: docker-compose build
  deploy:
    if: github.ref == 'refs/heads/main'
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Deploy to production
      run: |
        echo "Deploying to AWS/EC2..."
        # Add deployment commands
"""
            with open(".github/workflows/ci-cd.yml", "w") as f:
                f.write(ci_content)
            os.makedirs(".github/workflows", exist_ok=True)
            logger.info("GitHub Actions CI/CD generated")
    
    def setup_cloud(self, provider: str = "aws") -> None:
        """Generate cloud deployment configs."""
        if provider == "aws":
            # Terraform config stub
            tf_content = """
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

resource "aws_instance" "biometric_server" {
  ami           = "ami-0abcdef1234567890"
  instance_type = var.instance_type
  tags = {
    Name = "biometric-fraud-server"
  }
}

variable "region" {
  default = "us-east-1"
}

variable "instance_type" {
  default = "t3.medium"
}
"""
            os.makedirs("deploy/terraform", exist_ok=True)
            with open("deploy/terraform/main.tf", "w") as f:
                f.write(tf_content)
            
            # CloudFormation alternative
            cf_content = """AWSTemplateFormatVersion: '2010-09-09'
Description: Biometric Fraud System Infrastructure
Resources:
  EC2Instance:
    Type: AWS::EC2::Instance
    Properties:
      ImageId: ami-0abcdef1234567890
      InstanceType: t3.medium
"""
            with open("deploy/cloudformation/stack.yaml", "w") as f:
                f.write(cf_content)
            
            logger.info("AWS cloud configs generated (Terraform + CloudFormation)")
        elif provider == "gcp":
            # GCP deployment manager stub
            logger.info("GCP deployment configs would be generated here")
            pass
    
    def generate_env_files(self, env: str = "dev") -> None:
        """Generate .env files for different environments."""
        env_content = f"""# Biometric Fraud System - {env.upper()} Environment
APP_NAME=Biometric Fraud Prevention System
DEBUG={'True' if self.config['environment'][env]['debug'] else 'False'}
DB_PATH={self.config['environment'][env].get('db_path', 'data/biometrics.db')}
SECRET_KEY=super-secret-key-change-me-{env}
FRAUD_THRESHOLD=0.5
LOG_LEVEL={'DEBUG' if env == 'dev' else 'INFO'}

# SMTP for alerts
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
EMAIL_FROM=noreply@biometric-system.com

# Redis/Celery
REDIS_URL=redis://localhost:6379

# ML Config
MODEL_PATH=ml/models/fraud_model.pkl
EMBEDDING_DIM=128

# CORS
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
"""
        
        with open(f".env.{env}", "w") as f:
            f.write(env_content)
        
        # Copy to .env for current
        with open(".env", "w") as f:
            f.write(env_content)
        
        logger.info(f".env files generated for {env}")
    
    def run_full_deployment(self, env: str = "dev", cloud_provider: str = None, ci_platform: str = None):
        """Full deployment workflow."""
        logger.info(f"Starting deployment for {env} environment")
        
        self.generate_env_files(env)
        self.create_docker_compose(env)
        
        if ci_platform:
            self.generate_ci_config(ci_platform)
        
        if cloud_provider:
            self.setup_cloud(cloud_provider)
        
        self.deploy_local()
        
        # Post-deployment checks
        health_check = subprocess.run(["curl", "-f", "http://localhost:8000/health"], capture_output=True)
        if health_check.returncode == 0:
            logger.info("Deployment successful - health check passed")
        else:
            logger.error("Deployment warning - health check failed")
    
    def cleanup(self):
        """Cleanup deployment resources."""
        subprocess.run(["docker-compose", "down"], cwd=self.workspace)
        logger.info("Cleanup completed")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python deploy.py [local|cloud|ci] [env] [provider]")
        sys.exit(1)
    
    action = sys.argv[1]
    env = sys.argv[2] if len(sys.argv) > 2 else "dev"
    provider = sys.argv[3] if len(sys.argv) > 3 else None
    
    dm = DeploymentManager()
    
    if action == "local":
        dm.run_full_deployment(env)
    elif action == "cloud":
        dm.setup_cloud(provider)
    elif action == "ci":
        dm.generate_ci_config()
    elif action == "cleanup":
        dm.cleanup()
    else:
        print("Unknown action")
