#!/bin/bash
# scripts/deploy.sh

set -e

# Ensure we are in the project root (where docker-compose.yml is)
cd "$(dirname "$0")/.."

echo "🚀 Deploying LingoBot Pro..."

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Stop existing containers
echo "🛑 Stopping existing containers..."
sudo docker compose down

# Pull latest changes
echo "📥 Pulling latest changes..."
git pull origin main

# Build new images
echo "🔨 Building new images..."
sudo docker compose build --no-cache

# Start services
echo "🚀 Starting services..."
sudo docker compose up -d

# Run database migrations
echo "📊 Running database setup..."
sudo docker compose exec app python -c "
from server import Base, engine
Base.metadata.create_all(bind=engine)
print('✅ Database setup complete')
"

# Clean up old images (Save disk space on AWS)
echo "🧹 Cleaning up old images..."
sudo docker image prune -f

echo "✅ Deployment complete!"
echo ""
echo "🌐 Services are live!"
echo "📋 Check logs: sudo docker compose logs -f"