#!/bin/bash
# deploy-neural-generator.sh - Deployment script for Neural Code Generator

  set -e
  
  # Configuration
  REGISTRY=${REGISTRY:-"registry.example.com"}
  TAG=${TAG:-"latest"}
  NAMESPACE=${NAMESPACE:-"program-synthesis"}
  EMBEDDING_MODEL=${EMBEDDING_MODEL:-"all-mpnet-base-v2"}
  BATCH_SIZE=${BATCH_SIZE:-"32"}
  KUBECTL=${KUBECTL:-"kubectl"}
  
  # Print header
  echo "========================================"
  echo "Neural Code Generator Deployment Script"
  echo "========================================"
echo "Registry: $REGISTRY"
echo "Tag: $TAG"
echo "Namespace: $NAMESPACE"
  echo "========================================"
  
  # Check if namespace exists, create if it doesn't
  if ! $KUBECTL get namespace "$NAMESPACE" &> /dev/null; then
echo "Creating namespace: $NAMESPACE"
  $KUBECTL create namespace "$NAMESPACE"
  fi
  
  # Apply PVCs
  echo "Creating persistent volume claims..."
  $KUBECTL apply -f kubernetes/neural-pvc.yaml
  
  # Wait for PVCs to be bound
  echo "Waiting for PVCs to be bound..."
  $KUBECTL wait --for=condition=Bound pvc/neural-models-pvc -n "$NAMESPACE" --timeout=60s
  $KUBECTL wait --for=condition=Bound pvc/knowledge-base-pvc -n "$NAMESPACE" --timeout=60s
  
  # Apply ConfigMap with code examples (if needed)
  if [ -d "code-examples" ]; then
  echo "Creating ConfigMap with code examples..."
  $KUBECTL create configmap code-examples-config \
  --from-file=code-examples/ \
  -n "$NAMESPACE" \
  --dry-run=client -o yaml | $KUBECTL apply -f -
  fi
  
  # Apply the knowledge base initializer job
  echo "Creating knowledge base initializer job..."
  cat kubernetes/knowledge-base-initializer-job.yaml | \
  sed "s|\${REGISTRY}|$REGISTRY|g" | \
  sed "s|\${TAG}|$TAG|g" | \
  sed "s|\${EMBEDDING_MODEL}|$EMBEDDING_MODEL|g" | \
  sed "s|\${BATCH_SIZE}|$BATCH_SIZE|g" | \
  $KUBECTL apply -f -
  
  # Wait for knowledge base initialization to complete
  echo "Waiting for knowledge base initialization to complete..."
  $KUBECTL wait --for=condition=complete job/knowledge-base-initializer -n "$NAMESPACE" --timeout=600s
  
  # Apply the deployment and service
  echo "Deploying Neural Code Generator..."
  cat kubernetes/neural-code-generator-deployment.yaml | \
  sed "s|\${REGISTRY}|$REGISTRY|g" | \
  sed "s|\${TAG}|$TAG|g" | \
  $KUBECTL apply -f -
  
  $KUBECTL apply -f kubernetes/neural-code-generator-service.yaml
  
  # Wait for deployment to be ready
  echo "Waiting for deployment to be ready..."
  $KUBECTL rollout status deployment/neural-code-generator -n "$NAMESPACE" --timeout=300s
  
  echo "========================================"
  echo "Neural Code Generator deployed successfully!"
  echo "========================================"
  echo "To check the status:"
  echo "$KUBECTL get pods -n $NAMESPACE -l app=neural-code-generator"
  echo ""
  echo "To view logs:"
  echo "$KUBECTL logs -f -n $NAMESPACE -l app=neural-code-generator"
  echo ""
  echo "To forward the health check port:"
  echo "$KUBECTL port-forward -n $NAMESPACE svc/neural-code-generator 8000:8000"
  echo "Then open http://localhost:8000/health in your browser"
  echo "========================================"