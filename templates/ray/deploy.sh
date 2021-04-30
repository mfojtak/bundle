#!/bin/bash
kubectl apply -f cluster.yaml
kubectl wait --for=condition=Available deployment/ray-head
kubectl wait --for=condition=Available deployment/ray-worker
POD=$(kubectl get pod -l component=ray-head -o jsonpath="{.items[0].metadata.name}")
kubectl cp . $POD:/{{ name }}
kubectl exec $POD -- python /{{ name }}/project/main.py