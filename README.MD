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


