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
RUN_MLP_SRCS = $(MAIN_DIR)/RunMLP.cpp \
               $(SRC_DIR)/ActivationFunctions.cpp \
               $(SRC_DIR)/MLP.cpp
RUN_PERCEPTRON_SRCS = $(MAIN_DIR)/RunPerceptron.cpp $(UTILS_DIR)/Balance.cpp $(UTILS_DIR)/Data.cpp $(UTILS_DIR)/Normalize.cpp src/Perceptron.cpp
RUN_GA_SRCS = $(MAIN_DIR)/RunGA.cpp $(SRC_DIR)/GeneticAlgorithm/GeneticAlgorithm.cpp $(SRC_DIR)/GeneticAlgorithm/Individual.cpp $(SRC_DIR)/ActivationFunctions.cpp $(UTILS_DIR)/Balance.cpp $(UTILS_DIR)/Data.cpp $(UTILS_DIR)/Normalize.cpp $(UTILS_DIR)/GlobalSeed.cpp
AGENT_GA_SRCS = $(MAIN_DIR)/AgentGA.cpp $(SRC_DIR)/GeneticAlgorithm/GeneticAlgorithm.cpp $(SRC_DIR)/GeneticAlgorithm/Individual.cpp $(SRC_DIR)/ActivationFunctions.cpp $(UTILS_DIR)/Balance.cpp $(UTILS_DIR)/Data.cpp $(UTILS_DIR)/Normalize.cpp $(UTILS_DIR)/GlobalSeed.cpp
AGENT_PERCEPTRON_SRCS = $(MAIN_DIR)/AgentPerceptron.cpp $(UTILS_DIR)/Balance.cpp $(UTILS_DIR)/Data.cpp $(UTILS_DIR)/Normalize.cpp $(SRC_DIR)/Perceptron.cpp
TEST_AGENT_GA_SRCS = $(MAIN_DIR)/TestAgentGA.cpp $(SRC_DIR)/GeneticAlgorithm/GeneticAlgorithm.cpp $(SRC_DIR)/GeneticAlgorithm/Individual.cpp $(SRC_DIR)/ActivationFunctions.cpp $(UTILS_DIR)/Balance.cpp $(UTILS_DIR)/Data.cpp $(UTILS_DIR)/Normalize.cpp $(UTILS_DIR)/GlobalSeed.cpp
TEST_LOGIC_GATES_SRCS = $(MAIN_DIR)/TestLogicGates.cpp $(UTILS_DIR)/Balance.cpp $(UTILS_DIR)/Data.cpp $(UTILS_DIR)/Normalize.cpp $(SRC_DIR)/Perceptron.cpp

# Ejecutables generados
AGENT_MANUAL_EXE = $(BIN_DIR)/AgentManual
RUN_MLP_EXE = $(BIN_DIR)/RunMLP
RUN_PERCEPTRON_EXE = $(BIN_DIR)/RunPerceptron
RUN_GA_EXE = $(BIN_DIR)/RunGA
AGENT_GA_EXE = $(BIN_DIR)/AgentGA
AGENT_PERCEPTRON_EXE = $(BIN_DIR)/AgentPerceptron
AGENT_MLP_EXE = $(BIN_DIR)/AgentMLP
TEST_AGENT_GA_EXE = $(BIN_DIR)/TestAgentGA
TEST_LOGIC_GATES_EXE = $(BIN_DIR)/TestLogicGates

# Regla por defecto: compilar todos los ejecutables
all: $(AGENT_MANUAL_EXE) $(RUN_MLP_EXE) $(RUN_PERCEPTRON_EXE) $(RUN_GA_EXE) $(AGENT_PERCEPTRON_EXE) $(AGENT_MLP_EXE) $(AGENT_GA_EXE) $(TEST_AGENT_GA_EXE) $(TEST_LOGIC_GATES_EXE)

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

# Regla para compilar RunGA
$(RUN_GA_EXE): $(RUN_GA_SRCS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Regla para compilar AgentGA
$(AGENT_GA_EXE): $(AGENT_GA_SRCS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Regla para compilar AgentPerceptron
$(AGENT_PERCEPTRON_EXE): $(AGENT_PERCEPTRON_SRCS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Regla para compilar AgentMLPAuto (genera bin/AgentMLPAuto)
$(AGENT_MLP_EXE): $(MAIN_DIR)/AgentMLPAuto.cpp $(SRC_DIR)/MLP.cpp $(SRC_DIR)/ActivationFunctions.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -Iinclude -Ilib/ale/src $^ -o $@ $(LDFLAGS)

# Regla para compilar TestAgentGA
$(TEST_AGENT_GA_EXE): $(TEST_AGENT_GA_SRCS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Regla para compilar TestLogicGates
$(TEST_LOGIC_GATES_EXE): $(TEST_LOGIC_GATES_SRCS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Regla para limpiar los archivos generados
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Targets convenientes para compilar cada ejecutable individualmente
AgentManual: $(AGENT_MANUAL_EXE)

RunMLP: $(RUN_MLP_EXE)

RunPerceptron: $(RUN_PERCEPTRON_EXE)

RunGA: $(RUN_GA_EXE)

AgentGA: $(AGENT_GA_EXE)

AgentPerceptron: $(AGENT_PERCEPTRON_EXE)

TestAgentGA: $(TEST_AGENT_GA_EXE)

TestLogicGates: $(TEST_LOGIC_GATES_EXE)

TrainMLPAtari: src/Main/TrainMLPAtari.cpp src/MLP.cpp src/ActivationFunctions.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -Iinclude $^ -o bin/TrainMLPAtari

AgentMLP: $(AGENT_MLP_EXE)

# Asegurar que los objetivos no colisionen con archivos del sistema
.PHONY: all clean AgentManual RunMLP RunPerceptron TrainMLPAtari AgentMLPAuto RunGA AgentPerceptron AgentGA TestAgentGA TestLogicGates
