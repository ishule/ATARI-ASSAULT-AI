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
*** End Patch
Usa las clases de utilidades:

- `Data.hpp`: Para leer y dividir datos.

- `Normalize.hpp`: Para normalizar las entradas.
