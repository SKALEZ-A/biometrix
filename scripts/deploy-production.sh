#!/bin/bash

set -e

ENVIRONMENT=${1:-production}
VERSION=${2:-latest}

echo "Deploying Fraud Prevention System to $ENVIRONMENT..."

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(staging|production)$ ]]; then
    echo "Error: Environment must be 'staging' or 'production'"
    exit 1
fi

# Check kubectl access
if ! kubectl cluster-info &> /dev/null; then
    echo "Error: Cannot connect to Kubernetes cluster"
    exit 1
fi

# Set namespace
NAMESPACE="fraud-prevention"
if [ "$ENVIRONMENT" = "staging" ]; then
    NAMESPACE="fraud-prevention-staging"
fi

echo "Deploying to namespace: $NAMESPACE"

# Apply Kubernetes configurations
echo "Applying Kubernetes configurations..."
kubectl apply -f infrastructure/kubernetes/ -n $NAMESPACE

# Update image versions
echo "Updating service images to version: $VERSION..."
kubectl set image deployment/biometric-service \
    biometric-service=fraud-prevention/biometric-service:$VERSION \
    -n $NAMESPACE

kubectl set image deployment/fraud-detection-service \
    fraud-detection-service=fraud-prevention/fraud-detection-service:$VERSION \
    -n $NAMESPACE

kubectl set image deployment/transaction-service \
    transaction-service=fraud-prevention/transaction-service:$VERSION \
    -n $NAMESPACE

# Wait for rollout
echo "Waiting for rollout to complete..."
kubectl rollout status deployment/biometric-service -n $NAMESPACE
kubectl rollout status deployment/fraud-detection-service -n $NAMESPACE
kubectl rollout status deployment/transaction-service -n $NAMESPACE

# Run smoke tests
echo "Running smoke tests..."
./scripts/smoke-tests.sh $ENVIRONMENT

echo "Deployment to $ENVIRONMENT completed successfully!"
