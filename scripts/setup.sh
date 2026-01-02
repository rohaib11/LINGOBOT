#!/bin/bash
# scripts/setup.sh

set -e

# Ensure we are in the project root
cd "$(dirname "$0")/.."

echo "🚀 Setting up LingoBot Pro..."

# Check if .env exists
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo "📝 Creating .env file from template..."
        cp .env.example .env
        echo "⚠️  Please edit .env file with your configuration"
    else
        echo "⚠️  .env.example not found. Creating empty .env..."
        touch .env
    fi
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data logs models static dashboard nginx

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Build and start services
echo "🔨 Building Docker images..."
sudo docker compose build

echo "🚀 Starting services..."
sudo docker compose up -d

echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are running
echo "🔍 Checking service status..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ LingoBot Pro is running!"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Edit .env file with your Groq API key"
    echo "   2. Restart services: sudo docker compose restart"
    echo "   3. View logs: sudo docker compose logs -f"
else
    echo "❌ Services failed to start. Check logs: sudo docker compose logs"
    exit 1
fi