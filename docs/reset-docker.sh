#!/bin/bash

# Print message about what's happening
echo "⚠️  WARNING: This will remove ALL Docker containers, images, volumes, and networks ⚠️"
echo "This action cannot be undone. Data in volumes will be permanently deleted."
echo "Waiting 5 seconds before proceeding. Press Ctrl+C to cancel."
sleep 5

echo "🛑 Stopping all running containers..."
docker stop $(docker ps -aq) 2>/dev/null || echo "No containers to stop"

echo "🗑️  Removing all containers..."
docker rm $(docker ps -aq) 2>/dev/null || echo "No containers to remove"

echo "🗑️  Removing all images..."
docker rmi $(docker images -q) --force 2>/dev/null || echo "No images to remove"

echo "🗑️  Removing all volumes..."
docker volume rm $(docker volume ls -q) 2>/dev/null || echo "No volumes to remove"

echo "🗑️  Removing all networks..."
docker network rm $(docker network ls -q) 2>/dev/null || echo "No default networks to remove"

echo "🧹 Cleaning up any dangling resources..."
docker system prune -af --volumes

echo "✅ Docker environment has been completely reset."
echo "All containers, images, volumes, and networks have been removed."