#!/bin/bash

set -e

echo "Setting up Fraud Prevention Development Environment..."

# Check prerequisites
command -v node >/dev/null 2>&1 || { echo "Node.js is required but not installed. Aborting." >&2; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed. Aborting." >&2; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "Docker Compose is required but not installed. Aborting." >&2; exit 1; }

# Install Node dependencies
echo "Installing Node.js dependencies..."
npm install

# Start infrastructure services
echo "Starting infrastructure services (PostgreSQL, MongoDB, Redis)..."
docker-compose -f infrastructure/docker-compose.dev.yml up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 10

# Run database migrations
echo "Running database migrations..."
npm run migrate:up

# Seed development data
echo "Seeding development data..."
npm run seed:dev

# Build TypeScript
echo "Building TypeScript..."
npm run build

echo "Development environment setup complete!"
echo ""
echo "Available commands:"
echo "  npm run dev:biometric       - Start biometric service"
echo "  npm run dev:fraud-detection - Start fraud detection service"
echo "  npm run dev:transaction     - Start transaction service"
echo "  npm run dev:all             - Start all services"
echo "  npm test                    - Run tests"
echo ""
echo "Services running:"
echo "  PostgreSQL: localhost:5432"
echo "  MongoDB: localhost:27017"
echo "  Redis: localhost:6379"
