# 🧬 Algoritmo Genético para Redes Neuronales

## Requisitos del Proyecto

Este módulo implementa dos niveles de complejidad:

| Nota | Requisito | Implementación |
|------|-----------|----------------|
| **1.75** | Red Neuronal con pesos entrenados con Algoritmos Genéticos | ✅ `--mode weights` |
| **2.00** | Neuroevolución: Red Neuronal con evolución de arquitectura Y pesos | ✅ `--mode neuro` |

---

## 📁 Archivos Creados

```
include/GeneticAlgorithm/
├── Individual.hpp       # Representa un individuo (red neuronal)
└── GeneticAlgorithm.hpp # Algoritmo genético principal

src/GeneticAlgorithm/
├── Individual.cpp       # Implementación del individuo
└── GeneticAlgorithm.cpp # Implementación del GA

src/Main/
└── RunGA.cpp            # Programa principal
```

---

## 🔧 Compilación

Añade al Makefile:

```makefile
# Algoritmo Genético
GA_SRC = src/GeneticAlgorithm/Individual.cpp \
         src/GeneticAlgorithm/GeneticAlgorithm.cpp \
         src/ActivationFunctions.cpp

RunGA: src/Main/RunGA.cpp $(GA_SRC)
	$(CXX) $(CXXFLAGS) -I include -o RunGA $^ -std=c++17
```

O compila manualmente:

```bash
g++ -std=c++17 -O2 -I include \
    src/Main/RunGA.cpp \
    src/GeneticAlgorithm/Individual.cpp \
    src/GeneticAlgorithm/GeneticAlgorithm.cpp \
    src/ActivationFunctions.cpp \
    -o RunGA
```

---

## 🚀 Uso

### Modo 1: Evolución de Pesos (Nota 1.75)

La arquitectura es **fija**, solo se evolucionan los pesos.

```bash
# Ejemplo con Iris (4 entradas, 10 neuronas ocultas, 3 salidas)
./RunGA --mode weights \
        --arch 4-10-3 \
        --dataset data/Iris.csv \
        --num-classes 3 \
        --pop 50 \
        --gen 100 \
        --mutation 0.1
```

### Modo 2: Neuroevolución (Nota 2.00)

Se evoluciona **arquitectura Y pesos** simultáneamente.

```bash
# Ejemplo con Iris
./RunGA --mode neuro \
        --input 4 \
        --output 3 \
        --dataset data/Iris.csv \
        --num-classes 3 \
        --pop 50 \
        --gen 100 \
        --mutation 0.1 \
        --arch-mutation 0.05 \
        --min-layers 1 \
        --max-layers 4 \
        --min-neurons 4 \
        --max-neurons 64
```

---

## ⚙️ Parámetros

### Configuración General
| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--mode` | `weights` o `neuro` | `weights` |
| `--dataset` | Ruta al CSV | Requerido |
| `--num-classes` | Clases para one-hot | - |
| `--activation` | RELU, SIGMOID, TANH | RELU |
| `--save` | Guardar modelo | - |

### Parámetros del GA
| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--pop` | Tamaño de población | 50 |
| `--gen` | Generaciones máximas | 100 |
| `--mutation` | Tasa mutación pesos | 0.1 |
| `--elite` | Ratio de élite | 0.1 |

### Parámetros de Neuroevolución
| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--arch-mutation` | Tasa mutación arquitectura | 0.05 |
| `--min-layers` | Mínimo capas ocultas | 1 |
| `--max-layers` | Máximo capas ocultas | 4 |
| `--min-neurons` | Mínimo neuronas/capa | 4 |
| `--max-neurons` | Máximo neuronas/capa | 64 |

---

## 📊 Diferencias entre Modos

### Modo `weights` (1.75)
```
Generación 0:  [4-10-3]  [4-10-3]  [4-10-3]  [4-10-3]
                  ↓         ↓         ↓         ↓
                Solo cambian los valores de los pesos
                  ↓         ↓         ↓         ↓
Generación N:  [4-10-3]  [4-10-3]  [4-10-3]  [4-10-3]
               (misma arquitectura siempre)
```

### Modo `neuro` (2.00)
```
Generación 0:  [4-10-3]  [4-8-8-3]  [4-20-3]  [4-5-5-5-3]
                  ↓          ↓          ↓          ↓
              Cambian pesos Y arquitectura (capas/neuronas)
                  ↓          ↓          ↓          ↓
Generación N:  [4-15-8-3]  [4-12-3]  [4-20-10-3]  [4-8-3]
               (arquitecturas evolucionadas)
```

---

## 🔬 Operadores Genéticos

### Selección
- **Torneo**: Competencia entre k individuos
- **Ruleta**: Proporcional al fitness
- **Ranking**: Basado en posición
- **Elitismo**: Los mejores pasan directamente

### Crossover (Cruce)
```cpp
// Para cada peso:
if (random() < mutationRate) {
    // Mutación: peso aleatorio nuevo
    child.weight = random(-1, 1);
} else {
    // Crossover: hereda de padre o madre
    child.weight = random() < 0.5 ? father.weight : mother.weight;
}
```

### Mutación de Arquitectura (solo modo `neuro`)
1. **Añadir neurona** a una capa
2. **Eliminar neurona** de una capa
3. **Añadir capa oculta**
4. **Eliminar capa oculta**

---

## 📈 Ejemplo de Salida

```
========================================
MODO: NEUROEVOLUCIÓN (Nota 2.00)
========================================

Entradas: 4
Salidas: 3
Capas ocultas: 1-4
Neuronas/capa: 4-64

Iniciando evolución...
Generaciones máximas: 100
Fitness objetivo: 1e+09

Gen 0 | Best: 45.23 | Avg: 32.15 | Arch: 4-32-18-3 | Params: 826
Gen 1 | Best: 52.10 | Avg: 41.23 | Arch: 4-32-18-3 | Params: 826
Gen 2 | Best: 61.45 | Avg: 48.92 | Arch: 4-45-3 | Params: 318
...
Gen 50 | Best: 96.67 | Avg: 89.23 | Arch: 4-20-10-3 | Params: 293

========================================
RESULTADOS FINALES
========================================
Mejor arquitectura: 4-20-10-3
Parámetros totales: 293
Fitness final: 96.67
Accuracy Train: 97.50%
Accuracy Test: 96.67%
```

---

## 🎮 Integración con Atari (Opcional)

Para usar el GA con el juego Atari Assault, necesitas:

1. Incluir la librería ALE (Arcade Learning Environment)
2. Implementar `calculateGameFitness()` en `Individual.cpp`

```cpp
double Individual::calculateGameFitness(int maxSteps) {
    ALEInterface ale;
    ale.loadROM("assets/assault.bin");
    
    double totalReward = 0;
    int steps = 0;
    
    while (!ale.game_over() && steps < maxSteps) {
        // Obtener estado de la RAM
        auto state = getStateFromRAM(ale);
        
        // Predecir acción
        auto output = predict(state);
        
        // Ejecutar acción
        totalReward += ale.act(actionFromOutput(output));
        steps++;
    }
    
    fitness_ = totalReward;
    return fitness_;
}
```

---

## 📚 Referencias

- **NEAT**: NeuroEvolution of Augmenting Topologies (Stanley & Miikkulainen, 2002)
- **Genetic Algorithms**: Holland, 1975
- Tu implementación original en `Atari Assault/atariAssault_IA-master/`
