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

---

### 2. Procesamiento de datos

Usa las clases de utilidades:
- `Data.hpp`: Para leer y dividir datos.
- `Normalize.hpp`: Para normalizar las entradas.

---

### 3. Perceptrón

#### 3.1 Entrenamiento del Perceptrón
Compilamos el perceptrón de la siguiente forma:
```bash
make RunPerceptron
./bin/RunPerceptron --dataset <nombre_del_dataset>
```

Podemos probar con los datasets: `iris`, `cancer`, `wine`, `mnist` y `atari`.

**Función que realiza:** entrena un modelo de perceptrón (clasificador lineal) sobre el dataset indicado.

#### 3.2 Prueba con Atari
Si el perceptrón se ha entrenado con el dataset `atari`, podrá usarse con el agente compilado para perceptrón.

Compilamos y ejecutamos el agente perceptrón:
```bash
make AgentPerceptron
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/lib/ale
/bin/AgentPerceptron models/perceptron/<modelo_perceptron>.txt
```

- El primer argumento es la ROM de Atari Assault.
- El segundo es el fichero con los pesos/hiperparámetros guardados.

**Nota:** Asegúrate de exportar `libale.so` en `LD_LIBRARY_PATH` para que la librería se encuentre en tiempo de ejecución.

---

### 4. MLP

#### 4.1 Entrenamiento de la red MLP
Compilamos la red MLP de la siguiente forma:
```bash
make RunMLP
./bin/RunMLP --dataset <nombre_del_dataset>
```

Podemos probar con los datasets: `iris`, `cancer`, `wine` y `atari`.

**Función que realiza:** ejecuta experimentos para un dataset determinado, probando varias combinaciones (estructura, dropout, early stop, etc.) y guarda el mejor modelo en `models/mlp`.

#### 4.2 Prueba con Atari Assault
Si el MLP se ha entrenado con el dataset `atari`, después podrá probarse con la función `AgentMLPAuto.cpp`.

Esta función ejecuta el modelo entrenado en Atari de forma visual, analizando las posiciones de la RAM consideradas importantes y contabilizando la puntuación obtenida en cada ejecución.

Compilamos y ejecutamos:
```bash
make AgentMLPAuto
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/lib/ale
/bin/AgentMLPAuto supported/assault.bin models/mlp/atari_exp2_80-128-64-3_RELU_backprop_vbuena_1.txt
```

- El primer argumento es la ROM de Atari Assault.
- El segundo es el modelo con la arquitectura y los pesos.

**Nota:** Asegúrate de exportar la librería `libale.so` para que se pueda encontrar en tiempo de ejecución.

---

### 5. Algoritmos genéticos

#### 5.1 Compilación
Compilamos los ejecutables relacionados con algoritmos genéticos:
```bash
make RunGA
make AgentGA
make TestAgentGA
```

#### 5.2 Descripción de los ejecutables

- **RunGA**: Ejecuta experimentos de entrenamiento usando un algoritmo genético para optimizar pesos o arquitecturas.
    ```bash
    ./bin/RunGA --dataset <nombre_del_dataset> --mode neuro|weights [--population N] [--generations G] [--mutation-rate M]
    ```
    Consulta la función `usage` para ver todos los parámetros disponibles.

- **AgentGA**: Entrena un agente que utiliza el modelo generado por el algoritmo genético sobre la ROM de Atari, puedes configurar en este mismo archivo la config y cambiar para que solo se entrene con una topología fija (actualmente es pura neuroevolución), también se puede cambiar población, ratio de mutación entre muchas cosas más.
    ```bash
    make AgentGA
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/lib/ale
    ./bin/AgentGA
    ```

- **TestAgentGA**: Prueba y evalúa agentes entrenados en AgentGA.
    ```bash
    make TestAgentGA
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/lib/ale
    ./bin/TestAgentGA
    ```
    Para cambiar el modelo a cargar, edita `src/Main/TestAgentGA.cpp` y modifica la ruta del modelo (por ejemplo, `models/ga/best_individual.txt`) dentro del propio código.

#### 6 Nota importante
- Los modelos resultantes se almacenan en `models/` y los logs/resultados en `results/`.
- Asegúrate de tener instaladas las dependencias de SDL necesarias para ejecutar los binarios que utilizan la librería `libale.so`.
