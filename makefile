# Define el compilador
CXX = g++

# Opciones del compilador
CXXFLAGS = -Wall -Wextra -std=c++17

# Directorios específicos
ALE_DIR = lib/ale

# Rutas de inclusión y vinculadores necesarios
CXXFLAGS += -Iinclude -Iinclude/Utils -I$(ALE_DIR)/src -I/usr/include/SDL
LDFLAGS += -L$(ALE_DIR) -lale -lSDL

# Directorios del proyecto
SRC_DIR = src
MAIN_DIR = $(SRC_DIR)/Main
UTILS_DIR = $(SRC_DIR)/Utils
BUILD_DIR = build
BIN_DIR = bin

# Archivos fuente para cada ejecutable
AGENT_MANUAL_SRCS = $(MAIN_DIR)/AgentManual.cpp $(UTILS_DIR)/Balance.cpp $(UTILS_DIR)/Data.cpp $(UTILS_DIR)/Normalize.cpp
RUN_MLP_SRCS = $(MAIN_DIR)/RunMLP.cpp $(UTILS_DIR)/Balance.cpp $(UTILS_DIR)/Data.cpp $(UTILS_DIR)/Normalize.cpp src/ActivationFunctions.cpp src/MLP.cpp
RUN_PERCEPTRON_SRCS = $(MAIN_DIR)/RunPerceptron.cpp $(UTILS_DIR)/Balance.cpp $(UTILS_DIR)/Data.cpp $(UTILS_DIR)/Normalize.cpp src/Perceptron.cpp

# Ejecutables generados
AGENT_MANUAL_EXE = $(BIN_DIR)/AgentManual
RUN_MLP_EXE = $(BIN_DIR)/RunMLP
RUN_PERCEPTRON_EXE = $(BIN_DIR)/RunPerceptron

# Regla por defecto: compilar todos los ejecutables
all: $(AGENT_MANUAL_EXE) $(RUN_MLP_EXE) $(RUN_PERCEPTRON_EXE)

# Regla para compilar AgentManual
$(AGENT_MANUAL_EXE): $(AGENT_MANUAL_SRCS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Regla para compilar RunMLP
$(RUN_MLP_EXE): $(RUN_MLP_SRCS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Regla para compilar RunPerceptron
$(RUN_PERCEPTRON_EXE): $(RUN_PERCEPTRON_SRCS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Regla para limpiar los archivos generados
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)