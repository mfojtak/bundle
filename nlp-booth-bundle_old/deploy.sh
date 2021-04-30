#!/bin/bash
kubectl create -k ./
kubectl wait --for=condition=complete job/nlp-booth-bundle-build --timeout 2h
kubectl apply -f deployment.yaml