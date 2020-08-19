#!/bin/bash
kubectl create -k ./
kubectl wait --for=condition=complete job/{{ name }}-build --timeout 2h
kubectl apply -f deployment.yaml