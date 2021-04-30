#!/bin/bash
kubectl create -k ./

retval_complete=1
retval_failed=1
while [[ $retval_complete -ne 0 ]] && [[ $retval_failed -ne 0 ]]; do
  sleep 5
  output=$(kubectl wait --for=condition=failed job/{{ name }}-build --timeout=0 2>&1)
  retval_failed=$?
  output=$(kubectl wait --for=condition=complete job/{{ name }}-build --timeout=0 2>&1)
  retval_complete=$?
  kubectl logs -f job/{{ name }}-build
done

if [ $retval_failed -eq 0 ]; then
    echo "Job failed. Please check logs."
    exit 1
fi
echo "Image {{ image }} built successfully"
kubectl apply -f job.yaml