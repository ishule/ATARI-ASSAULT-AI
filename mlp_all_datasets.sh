#!/bin/bash

echo "=========================================="
echo "Ejecutando TODOS los datasets MLP"
echo "=========================================="

mkdir -p results models/mlp

# Limpiar archivo de resultados
> results/mlp_results.txt

datasets=("iris" "cancer" "wine" "mnist")

for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"
    num=$((i+1))
    total=${#datasets[@]}
    
    echo ""
    echo "=========================================="
    echo "[$num/$total] Ejecutando $dataset..."
    echo "=========================================="
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
echo "Resumen de mejores modelos:"
grep -A7 "MEJOR MODELO PARA" results/mlp_results.txt

echo ""
echo "Modelos guardados en models/mlp/:"
ls -lh models/mlp/