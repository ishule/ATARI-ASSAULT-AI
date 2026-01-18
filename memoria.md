**Resumen**
- **Propósito:** Resumen técnico y estado operativo de las tecnologías implementadas en este repositorio: perceptrón y algoritmos genéticos.

**Tecnologías Implementadas**
- **Perceptrón:** Implementación clásica en `src/Perceptron.cpp` y `include/Perceptron.hpp`. Entrenamiento y ejecución mediante el binario/ejecutables en `bin/RunPerceptron` y scripts relacionados.
- **Algoritmos Genéticos (GA):** Implementación para optimización/neuroevolución en `src/GeneticAlgorithm/GeneticAlgorithm.cpp` y `include/GeneticAlgorithm/GeneticAlgorithm.hpp`. Utilizado para búsqueda de pesos/mejores configuraciones y generación de individuos (`include/GeneticAlgorithm/Individual.hpp`).

**Estado Operativo (resumen por dataset — análisis muy por encima)**
- **Iris:** Perceptrón: resultado esperado: alto rendimiento (dataset pequeño y linealmente separable en gran medida). Ver: [results/perceptron_iris.txt](results/perceptron_iris.txt). Estado: operativo y útil como baseline rápido.
- **Breast Cancer (Breast Cancer Wisconsin):** Perceptrón: suele obtener buenas métricas (clasificación binaria con separabilidad razonable). Ver: [results/perceptron_cancer.txt](results/perceptron_cancer.txt). Estado: operativo; podría beneficiarse de preprocesado adicional y balanceo (`include/Utils/Balance.hpp`).
- **Wine (red wine quality):** Perceptrón: rendimiento moderado; problema más ruidoso y con más clases/umbralización. Ver: [results/perceptron_wine.txt](results/perceptron_wine.txt). Estado: operativo pero limitado — considerar MLP o tuning.
- **MNIST:** Perceptrón simple: limitada capacidad en clasificación de dígitos (alto error esperado). Ver: [results/perceptron_mnist.txt](results/perceptron_mnist.txt). Estado: funcional pero insuficiente para producción; utilizar MLP/MLP con GA o redes convolucionales para mejoras.

- **Algoritmos Genéticos (aplicaciones observadas):** GA empleados para optimizar parámetros/pesos en experimentos MLP/neurales. Resultados y logs: [results/ga_neuro_iris.txt](results/ga_neuro_iris.txt), [results/ga_neuro_wine.txt](results/ga_neuro_wine.txt). Modelos finales guardados en `models/` (p. ej. [models/ga/best_individual.txt](models/ga/best_individual.txt)). Estado: operativo; producen soluciones útiles pero dependen fuertemente de la configuración (tasa de mutación, población, número de generaciones).

**Archivos y código relevantes**
- **Implementación perceptrón:** [src/Perceptron.cpp](src/Perceptron.cpp), [include/Perceptron.hpp](include/Perceptron.hpp)
- **Implementación GA:** [src/GeneticAlgorithm/GeneticAlgorithm.cpp](src/GeneticAlgorithm/GeneticAlgorithm.cpp), [include/GeneticAlgorithm/GeneticAlgorithm.hpp](include/GeneticAlgorithm/GeneticAlgorithm.hpp)
- **Resultados y modelos:**
  - [results/perceptron_iris.txt](results/perceptron_iris.txt)
  - [results/perceptron_cancer.txt](results/perceptron_cancer.txt)
  - [results/perceptron_wine.txt](results/perceptron_wine.txt)
  - [results/perceptron_mnist.txt](results/perceptron_mnist.txt)
  - [results/ga_neuro_iris.txt](results/ga_neuro_iris.txt)
  - [models/ga/best_individual.txt](models/ga/best_individual.txt)

**Observaciones y recomendaciones (breves)**
- El perceptrón es útil como línea base rápida; para problemas no lineales (MNIST, wine) conviene usar MLP o técnicas más sofisticadas.
- Los GA funcionan como optimizadores globales pero requieren tuning; usarlos para inicializar pesos o hiperparámetros puede mejorar resultados cuando el espacio es multimodal.
- Recomendado: añadir métricas estándar (accuracy, precision/recall, confusion matrix) en los scripts de salida para comparar experimentos más fácilmente.

**Próximos pasos sugeridos**
- Ejecutar experimentos automáticos (`RunPerceptron`, `RunGA`) para obtener métricas consolidadas.
- Probar MLP/Backprop para MNIST y Wine; comparar con GA+MLP.

---
*Memoria generada automáticamente: análisis muy por encima basado en los resultados y archivos disponibles en el repositorio.*

**Descripción detallada de implementaciones**

**Perceptrón**
- Estructura: `Perceptron` hereda de `Network` y almacena `weights` como una matriz `numOutputs x (numInputs + 1)`; el último elemento de cada fila actúa como bias. Implementación: [src/Perceptron.cpp](src/Perceptron.cpp) y `include/Perceptron.hpp`.
- Entrenamiento: implementa la regla clásica del perceptrón por muestra con learning rate fijo (`learningRate = 0.1`) y número de épocas fijo (100). Para cada muestra y cada unidad de salida: calcula producto punto, añade bias, aplica función escalón (step) produciendo `1.0` o `-1.0`, calcula `error = target - pred` y actualiza pesos y bias según `w += lr * error * x`.
- Soporta tanto salida binaria (un valor) como multiclase (one-hot). Durante el entrenamiento se calcula `accuracy` sobre train/val y se guarda el mejor conjunto de pesos si existe conjunto de validación.
- I/O: `load` y `save` leen/escriben pesos en texto plano.
- Observación crítica: los loaders en `src/Main/RunPerceptron.cpp` añaden explícitamente una columna `1.0` como bias a cada fila de entrada y el `Perceptron` también mantiene un peso extra (`weights[j].back()`) que se suma como bias — esto puede duplicar el efecto del bias. Recomiendo armonizar (quitar la suma extra en `Perceptron` o no añadir `1.0` en los loaders) para evitar inconsistencia.

**Algoritmos Genéticos (GA)**
- Estructura general: implementado en `include/GeneticAlgorithm/GeneticAlgorithm.hpp` y `src/GeneticAlgorithm/GeneticAlgorithm.cpp`. `GAConfig` centraliza parámetros (población, élite, selección, tasas de mutación, neuroevolución, generaciones, `saveDir`).
- Flujo: inicializa población (pesos aleatorios con Xavier o topologías aleatorias en neuroevolución), itera generaciones llamando `runGeneration()` hasta `maxGenerations` o alcanzar `targetFitness`. Cada generación: evaluar fitness, ordenar población, selección, cruce (`crossover` / `crossoverNeuroevolution`) y mutación, formar nueva población.
- Selección: torneo, ruleta y ranking implementados. Además se mantiene élite según `eliteRatio`.
- Guardado: al final guarda el mejor individuo en `config.saveDir` (por defecto `models/ga/`).

**Individual (representación de red neuronal en GA)**
- Archivo: `include/GeneticAlgorithm/Individual.hpp` y `src/GeneticAlgorithm/Individual.cpp`.
- Contenido: `topology_` (`vector<int>`), `weights_` (VecWeights: [capa][neurona][peso]), `biases_`, `activation_`, `fitness_`.
- Inicialización: `initializeRandomWeights()` usa Xavier/Glorot; constructor para neuroevolución genera topologías aleatorias (`generateRandomTopology`).
- Cruce y mutación: `crossover` mezcla pesos entre padres; `crossoverNeuroevolution` permite cambios de arquitectura y mezcla con re-inicialización cuando es necesario; `mutateWeights` aplica ruido gaussiano con clipping; `mutateArchitecture` añade/quita neuronas o capas y ajusta matrices de pesos.
- Forward / predicción: `forwardPass` aplica `ActivationFunctions::apply` en capas ocultas y `sigmoide` en la capa de salida; `predict` envuelve `forwardPass`.
- Fitness: `calculateFitness` combina accuracy y MSE: `fitness = accuracy*100 - mse` (mayor es mejor).

**Preprocesado y utilidades**
- Loaders dataset: `src/Main/RunPerceptron.cpp` contiene `loadIris`, `loadCancer`, `loadWine`, `loadMNIST`, `loadCreditCard`. Cada loader realiza parsing, normalizado/estandarizado donde aplica y agrega `1.0` como bias al final de cada fila.
- Normalize: `include/Utils/Normalize.hpp` + `src/Utils/Normalize.cpp` implementan `fit`/`transform` y guardar/cargar scaler.
- Balance: `include/Utils/Balance.hpp` + `src/Utils/Balance.cpp` implementan oversampling simple por clase.
- Data: `include/Utils/Data.hpp` + `src/Utils/Data.cpp` contienen un loader CSV genérico y un método `splitData`.

**Puntos importantes / Observaciones**
- Bias: armonizar tratamiento del bias entre loaders y `Perceptron` para evitar duplicación.
- Métricas: `RunPerceptron` imprime accuracy train/val/test, pero es recomendable añadir precision/recall, F1 y matriz de confusión para análisis más fino.
- MNIST y problemas no lineales: Perceptrón simple es limitado; usar MLP (o CNN para imágenes) y/o combinar GA para búsqueda de topologías/hiperparámetros.
- GA: solución flexible pero sensible al tuning (población, mutación, generaciones); útil para iniciar pesos o buscar arquitecturas en espacios multimodales.

**Archivos clave (rápido)**
- Perceptrón: [src/Perceptron.cpp](src/Perceptron.cpp), [include/Perceptron.hpp](include/Perceptron.hpp)
- GA: [src/GeneticAlgorithm/GeneticAlgorithm.cpp](src/GeneticAlgorithm/GeneticAlgorithm.cpp), [include/GeneticAlgorithm/GeneticAlgorithm.hpp](include/GeneticAlgorithm/GeneticAlgorithm.hpp)
- Individual: [src/GeneticAlgorithm/Individual.cpp](src/GeneticAlgorithm/Individual.cpp), [include/GeneticAlgorithm/Individual.hpp](include/GeneticAlgorithm/Individual.hpp)
- Loaders / ejecución: [src/Main/RunPerceptron.cpp](src/Main/RunPerceptron.cpp)
- Utils: [src/Utils/Normalize.cpp](src/Utils/Normalize.cpp), [src/Utils/Balance.cpp](src/Utils/Balance.cpp), [src/Utils/Data.cpp](src/Utils/Data.cpp)

**Recomendaciones de próximos pasos**
- Corregir la inconsistencia de bias en `Perceptron` o en los loaders.
- Añadir métricas más completas en `RunPerceptron`.
- Para MNIST: entrenar MLP y comparar con GA+MLP; considerar CNN para mejor rendimiento.

---
*Contenido añadido automáticamente: descripción técnica y observaciones sobre la implementación de Perceptrón y Algoritmos Genéticos en el repositorio.*
