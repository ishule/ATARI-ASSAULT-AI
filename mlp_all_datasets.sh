#!/bin/bash

echo "=========================================="
echo "Ejecutando TODOS los datasets MLP"
echo "=========================================="

mkdir -p results
> results/mlp_results.txt

datasets=("iris" "cancer" "wine" "mnist")

for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"
    num=$((i+1))
    total=${#datasets[@]}
    
    echo ""
    echo "[$num/$total] Ejecutando $dataset..."
    ./RunMLP --dataset $dataset --results results/mlp_results.txt
    
    if [ $? -eq 0 ]; then
        echo "✓ $dataset completado"
    else
        echo "✗ Error en $dataset"
    fi
done

echo ""
echo "=========================================="
echo "TODOS LOS EXPERIMENTOS COMPLETADOS"
echo "=========================================="
echo ""
echo "Resumen:"
grep "MEJOR MODELO PARA" results/mlp_results.txt