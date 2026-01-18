# Instrucciones para Compilar y Ejecutar

## Recomendación
Es importante revisar los videos de clase, especialmente el del **24 de noviembre**, para entender el proceso completo.

## Compilación
Para compilar el archivo `minimal_agent.cpp`, utiliza el siguiente comando:

```bash
g++ minimal_agent.cpp ./lib/ale/libale.so -o agent
```

## Ejecución
Para ejecutar el programa, usa el siguiente comando:

```bash
LD_LIBRARY_PATH=. ./agent ./supported/assault.bin
```

## Nota
Debes crear la carpeta `build` siguiendo las instrucciones proporcionadas en el video mencionado.

# Compilación y ejecución del proyecto

Este proyecto implementa algoritmos de aprendizaje automático para Atari Assault utilizando la Arcade Learning Environment (ALE).

## Cómo usar y ejecutar el proyecto

### 1. Recolección de datos de Atari Assault

Compila el **Agent Manual**:
```bash
make
./bin/AgentManual <rom_file>
```

Esto generará un archivo `data_manual.csv` con los estados de la RAM y las acciones tomadas durante el juego.

### 2. Procesamiento de datos

Usa las clases de utilidades:
- `Data.hpp`: Para leer y dividir datos.
- `Normalize.hpp`: Para normalizar las entradas.

### 3. Perceptrón
Compilamos el perceptrón de la siguiente forma:
```bash
make RunPerceptron
./bin/RunPerceptron --dataset <nombre_del_dataset>
```

Podemos probar con los datasets: iris, cancer, wine, mnist y atari..

**Función que realiza:** entrena un modelo de perceptrón (clasificador lineal) sobre el dataset indicado.

#### 3.1 Prueba con Atari
Si el perceptrón se ha entrenado con el dataset atari, podrá usarse con el agente compilado para perceptrón.

Compilamos y ejecutamos el agente perceptrón:
```bash
make AgentPerceptron
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/lib/ale
/bin/AgentPerceptron supported/assault.bin models/perceptron/<modelo_perceptron>.txt
```

El primer argumento es la ROM de Atari Assault y el segundo es el fichero con los pesos/hiperparámetros guardados. Asegurarse de exportar libale.so en LD_LIBRARY_PATH para que la librería se encuentre en tiempo de ejecución.

### 4. MLP
Compilamos la red MLP de la siguiente forma:
```bash
make RunMLP
./bin/RunMLP --dataset <nombre_del_dataset>
```

Podemos probar con los datasets: iris, cancer, wine y atari.
Consultar función usage para ver qué más parámetros podemos pasar a RunMLP.

**Función que realiza:** se encarga de ejecutar todos los experimentos para un determinado dataset. Prueba varias combinaciones (combinando estructura, añadiendo dropout o early stop, etc) y se queda con el mejor modelo que lo guarda en models/mlp.

#### 4.1 Prueba con Atari Assault
Si el MLP se ha entrenado con el dataset atari, después podrá probarse con la función AgentMLPAuto.cpp.

Esta función se encarga de ejecutar el modelo entrenado en la Atari de forma visual fijándonos en las posiciones de la RAM que hemos considerado importantes a la hora de recolectar los datos a mano y contabiliza la puntuación obtenida en cada ejecución. 

Compilamos y ejecutamos:
```bash
make AgentMLPAuto
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/lib/ale
/bin/AgentMLPAuto supported/assault.bin models/mlp/atari_exp2_80-128-64-3_RELU_backprop_vbuena_1.txt
```

El primer argumento es la ROM de Atari Assault y el segundo es el modelo con la arquitectura y los pesos.

Nos aseguramos que primero hemos exportado la librería libale.so para que se pueda encontrar en tiempo de ejecución, si no, no se ejecutará.

### 5. Algoritmos genéticos
Compilamos los ejecutables relacionados con algoritmos genéticos:
```bash
make RunGA
make AgentGA
make TestAgentGA
```

- RunGA: ejecuta experimentos de entrenamiento usando un algoritmo genético para optimizar pesos o arquitecturas. Usar:
```bash
./bin/RunGA --dataset <nombre_del_dataset> [--population N] [--generations G] [--mutation-rate M]
```
Consultar la función usage para ver todos los parámetros disponibles.

- TestAgentGA: sirve para probar y evaluar candidatos resultantes del algoritmo genético en un entorno controlado sin exigir la ROM.

- AgentGA: ejecuta un agente que utiliza el modelo generado por el algoritmo genético sobre la ROM de Atari.

RunGA

Fichero: src/Main/RunGA.cpp
Propósito: ejecutar el Algoritmo Genético para optimizar pesos (modo weights) o topologías+pesos (modo neuro) sobre un dataset.
Parámetro mínimo necesario: --dataset <nombre|ruta> (ej.: iris, cancer, wine, mnist o un CSV). Es el único parámetro obligatorio para una ejecución básica.
Comportamiento actual: el runner crea particiones train/val/test pero concatena val a train y evoluciona sobre train+val. La evaluación final se hace sobre test.
Fitness mostrado: fitness = accuracy*100 - MSE (no confundir con accuracy pura).
Ejemplo (Windows):
bin\RunGA --dataset iris
5.1 Prueba con Atari Assault

AgentGA (entrenamiento): src/Main/AgentGA.cpp entrena un agente en Atari usando GA. Ejecuta episodios en el emulador, evalúa individuos (redes) y aplica operadores genéticos para evolucionar políticas.
Cómo entrenar:
make AgentGA
bin\AgentGA supported/assault.bin
TestAgentGA (evaluación): src/Main/TestAgentGA.cpp carga un modelo guardado y ejecuta el agente para medir rendimiento sin volver a entrenar.
Cómo probar un bot:
make TestAgentGA
bin\TestAgentGA
Cambiar modelo a cargar: edita src/Main/TestAgentGA.cpp y modifica la ruta usada para cargar el modelo (por ejemplo models/ga/best_individual.txt). Por defecto el código suele apuntar al mejor modelo guardado.
Notas:

Los modelos resultantes se almacenan en models/ga/ y los logs/resultados en results/.
Si quieres que la validación influya en la selección, puedo modificar RunGA para evaluar por val cada generación o para hacer early-stopping (actualmente la evolución se realiza sobre train+val).


