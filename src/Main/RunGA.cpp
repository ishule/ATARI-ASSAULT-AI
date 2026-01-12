/**
 * =============================================================================
 * RunGA.cpp - Programa principal para Algoritmo Genético con Redes Neuronales
 * =============================================================================
 * 
 * Este programa permite entrenar redes neuronales usando:
 * 
 * 1. EVOLUCIÓN DE PESOS (--mode weights) - Nota 1.75
 *    - Arquitectura fija definida por el usuario
 *    - Solo se evolucionan los pesos mediante GA
 *    
 * 2. NEUROEVOLUCIÓN (--mode neuro) - Nota 2.00
 *    - Se evoluciona tanto la arquitectura como los pesos
 *    - Mutaciones pueden añadir/eliminar neuronas y capas
 * 
 * Uso:
 *   ./RunGA --mode weights --arch 4-10-3 --dataset iris --generations 100
 *   ./RunGA --mode neuro --input 4 --output 3 --dataset iris --generations 100
 */

#include "GeneticAlgorithm/GeneticAlgorithm.hpp"
#include "ActivationFunctions.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cstring>

// =============================================================================
// FUNCIONES DE UTILIDAD
// =============================================================================

/**
 * Lee un dataset CSV
 * @param filepath Ruta al archivo
 * @param X Matriz de características (salida)
 * @param Y Matriz de etiquetas (salida)
 * @param hasHeader Si el CSV tiene cabecera
 * @param labelCol Columna de etiqueta (-1 para última)
 */
void loadCSV(const std::string& filepath,
             std::vector<std::vector<double>>& X,
             std::vector<std::vector<double>>& Y,
             bool hasHeader = true,
             int labelCol = -1) {
    
    std::ifstream file(filepath);
    if (!file) {
        throw std::runtime_error("No se pudo abrir: " + filepath);
    }
    
    std::string line;
    bool isFirst = true;
    int numCols = 0;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        // Saltar cabecera
        if (hasHeader && isFirst) {
            isFirst = false;
            continue;
        }
        
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;
        
        while (std::getline(ss, cell, ',')) {
            try {
                row.push_back(std::stod(cell));
            } catch (...) {
                row.push_back(0.0);
            }
        }
        
        if (row.empty()) continue;
        
        if (numCols == 0) numCols = row.size();
        
        // Determinar columna de etiqueta
        int lc = (labelCol < 0) ? numCols - 1 : labelCol;
        
        // Separar características y etiqueta
        std::vector<double> features;
        for (int i = 0; i < static_cast<int>(row.size()); ++i) {
            if (i != lc) {
                features.push_back(row[i]);
            }
        }
        
        X.push_back(features);
        Y.push_back({row[lc]});
    }
}

/**
 * Normaliza datos al rango [0,1]
 */
void normalize(std::vector<std::vector<double>>& X) {
    if (X.empty()) return;
    
    size_t numFeatures = X[0].size();
    
    for (size_t f = 0; f < numFeatures; ++f) {
        double minVal = X[0][f];
        double maxVal = X[0][f];
        
        for (const auto& row : X) {
            if (row[f] < minVal) minVal = row[f];
            if (row[f] > maxVal) maxVal = row[f];
        }
        
        double range = maxVal - minVal;
        if (range < 1e-8) range = 1.0;
        
        for (auto& row : X) {
            row[f] = (row[f] - minVal) / range;
        }
    }
}

/**
 * Convierte etiquetas a one-hot encoding
 */
void toOneHot(std::vector<std::vector<double>>& Y, int numClasses) {
    std::vector<std::vector<double>> oneHot;
    
    for (const auto& y : Y) {
        int label = static_cast<int>(y[0]);
        std::vector<double> encoded(numClasses, 0.0);
        if (label >= 0 && label < numClasses) {
            encoded[label] = 1.0;
        }
        oneHot.push_back(encoded);
    }
    
    Y = oneHot;
}

/**
 * Divide datos en train/test
 */
void trainTestSplit(const std::vector<std::vector<double>>& X,
                    const std::vector<std::vector<double>>& Y,
                    std::vector<std::vector<double>>& Xtrain,
                    std::vector<std::vector<double>>& Ytrain,
                    std::vector<std::vector<double>>& Xtest,
                    std::vector<std::vector<double>>& Ytest,
                    double testRatio = 0.2) {
    
    std::vector<size_t> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);
    
    size_t testSize = static_cast<size_t>(X.size() * testRatio);
    
    for (size_t i = 0; i < indices.size(); ++i) {
        if (i < testSize) {
            Xtest.push_back(X[indices[i]]);
            Ytest.push_back(Y[indices[i]]);
        } else {
            Xtrain.push_back(X[indices[i]]);
            Ytrain.push_back(Y[indices[i]]);
        }
    }
}

/**
 * Parsea una arquitectura del formato "4-10-5-3"
 */
std::vector<int> parseArchitecture(const std::string& arch) {
    std::vector<int> topology;
    std::stringstream ss(arch);
    std::string token;
    
    while (std::getline(ss, token, '-')) {
        topology.push_back(std::stoi(token));
    }
    
    return topology;
}

/**
 * Evalúa accuracy de un individuo
 */
double evaluateAccuracy(const Individual& ind,
                       const std::vector<std::vector<double>>& X,
                       const std::vector<std::vector<double>>& Y) {
    int correct = 0;
    
    for (size_t i = 0; i < X.size(); ++i) {
        auto pred = ind.predict(X[i]);
        
        if (Y[i].size() == 1) {
            // Binario
            double predicted = (pred[0] >= 0.5) ? 1.0 : 0.0;
            if (std::abs(predicted - Y[i][0]) < 0.1) correct++;
        } else {
            // Multiclase
            auto predMax = std::max_element(pred.begin(), pred.end());
            auto trueMax = std::max_element(Y[i].begin(), Y[i].end());
            if (std::distance(pred.begin(), predMax) == 
                std::distance(Y[i].begin(), trueMax)) {
                correct++;
            }
        }
    }
    
    return 100.0 * correct / X.size();
}

void printUsage(const char* programName) {
    std::cout << "\nUso: " << programName << " [opciones]\n\n";
    std::cout << "MODOS:\n";
    std::cout << "  --mode weights    Evolución solo de pesos (Nota 1.75)\n";
    std::cout << "  --mode neuro      Neuroevolución completa (Nota 2.00)\n\n";
    
    std::cout << "ARQUITECTURA (modo weights):\n";
    std::cout << "  --arch <topo>     Arquitectura, ej: 4-10-5-3\n\n";
    
    std::cout << "ARQUITECTURA (modo neuro):\n";
    std::cout << "  --input <n>       Número de entradas\n";
    std::cout << "  --output <n>      Número de salidas\n";
    std::cout << "  --min-layers <n>  Mínimo capas ocultas (default: 1)\n";
    std::cout << "  --max-layers <n>  Máximo capas ocultas (default: 4)\n";
    std::cout << "  --min-neurons <n> Mínimo neuronas/capa (default: 4)\n";
    std::cout << "  --max-neurons <n> Máximo neuronas/capa (default: 64)\n\n";
    
    std::cout << "DATOS:\n";
    std::cout << "  --dataset <path>  Ruta al CSV\n";
    std::cout << "  --no-header       CSV sin cabecera\n";
    std::cout << "  --label-col <n>   Columna de etiqueta (-1 para última)\n";
    std::cout << "  --num-classes <n> Número de clases (para one-hot)\n\n";
    
    std::cout << "PARÁMETROS GA:\n";
    std::cout << "  --pop <n>         Tamaño de población (default: 50)\n";
    std::cout << "  --gen <n>         Generaciones (default: 100)\n";
    std::cout << "  --mutation <f>    Tasa mutación pesos (default: 0.1)\n";
    std::cout << "  --arch-mutation <f> Tasa mutación arquitectura (default: 0.05)\n";
    std::cout << "  --elite <f>       Ratio élite (default: 0.1)\n\n";
    
    std::cout << "OTROS:\n";
    std::cout << "  --activation <act> Activación: RELU, SIGMOID, TANH (default: RELU)\n";
    std::cout << "  --save <path>     Guardar mejor modelo\n";
    std::cout << "  --quiet           Sin mensajes de progreso\n";
    std::cout << "  --help            Mostrar esta ayuda\n\n";
    
    std::cout << "EJEMPLOS:\n";
    std::cout << "  # Evolución de pesos con Iris dataset\n";
    std::cout << "  " << programName << " --mode weights --arch 4-10-3 --dataset data/Iris.csv --num-classes 3\n\n";
    std::cout << "  # Neuroevolución con Iris dataset\n";
    std::cout << "  " << programName << " --mode neuro --input 4 --output 3 --dataset data/Iris.csv --num-classes 3\n";
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[]) {
    // Valores por defecto
    std::string mode = "weights";
    std::string archStr = "4-10-3";
    std::string datasetPath = "";
    std::string savePath = "";
    std::string activationStr = "RELU";
    
    int inputSize = 0;
    int outputSize = 0;
    int numClasses = 0;
    int labelCol = -1;
    bool hasHeader = true;
    
    GAConfig config;
    config.populationSize = 50;
    config.maxGenerations = 100;
    config.mutationRate = 0.1;
    config.archMutationRate = 0.05;
    config.eliteRatio = 0.1;
    config.minHiddenLayers = 1;
    config.maxHiddenLayers = 4;
    config.minNeuronsPerLayer = 4;
    config.maxNeuronsPerLayer = 64;
    config.verbose = true;
    
    // Parsear argumentos
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (arg == "--arch" && i + 1 < argc) archStr = argv[++i];
        else if (arg == "--dataset" && i + 1 < argc) datasetPath = argv[++i];
        else if (arg == "--save" && i + 1 < argc) savePath = argv[++i];
        else if (arg == "--input" && i + 1 < argc) inputSize = std::stoi(argv[++i]);
        else if (arg == "--output" && i + 1 < argc) outputSize = std::stoi(argv[++i]);
        else if (arg == "--num-classes" && i + 1 < argc) numClasses = std::stoi(argv[++i]);
        else if (arg == "--label-col" && i + 1 < argc) labelCol = std::stoi(argv[++i]);
        else if (arg == "--pop" && i + 1 < argc) config.populationSize = std::stoi(argv[++i]);
        else if (arg == "--gen" && i + 1 < argc) config.maxGenerations = std::stoi(argv[++i]);
        else if (arg == "--mutation" && i + 1 < argc) config.mutationRate = std::stod(argv[++i]);
        else if (arg == "--arch-mutation" && i + 1 < argc) config.archMutationRate = std::stod(argv[++i]);
        else if (arg == "--elite" && i + 1 < argc) config.eliteRatio = std::stod(argv[++i]);
        else if (arg == "--min-layers" && i + 1 < argc) config.minHiddenLayers = std::stoi(argv[++i]);
        else if (arg == "--max-layers" && i + 1 < argc) config.maxHiddenLayers = std::stoi(argv[++i]);
        else if (arg == "--min-neurons" && i + 1 < argc) config.minNeuronsPerLayer = std::stoi(argv[++i]);
        else if (arg == "--max-neurons" && i + 1 < argc) config.maxNeuronsPerLayer = std::stoi(argv[++i]);
        else if (arg == "--activation" && i + 1 < argc) activationStr = argv[++i];
        else if (arg == "--no-header") hasHeader = false;
        else if (arg == "--quiet") config.verbose = false;
    }
    
    // Validar argumentos
    if (datasetPath.empty()) {
        std::cerr << "Error: Se requiere --dataset\n";
        printUsage(argv[0]);
        return 1;
    }
    
    // Parsear activación
    ActivationType activation = ActivationFunctions::fromString(activationStr);
    
    // Cargar datos
    std::vector<std::vector<double>> X, Y;
    std::cout << "Cargando dataset: " << datasetPath << "\n";
    loadCSV(datasetPath, X, Y, hasHeader, labelCol);
    std::cout << "Muestras cargadas: " << X.size() << "\n";
    
    if (X.empty()) {
        std::cerr << "Error: Dataset vacío\n";
        return 1;
    }
    
    // Normalizar características
    normalize(X);
    
    // Convertir a one-hot si es multiclase
    if (numClasses > 0) {
        toOneHot(Y, numClasses);
    }
    
    // Dividir en train/test
    std::vector<std::vector<double>> Xtrain, Ytrain, Xtest, Ytest;
    trainTestSplit(X, Y, Xtrain, Ytrain, Xtest, Ytest, 0.2);
    
    std::cout << "Train: " << Xtrain.size() << " muestras\n";
    std::cout << "Test: " << Xtest.size() << " muestras\n\n";
    
    // Crear algoritmo genético según modo
    Individual best({1});  // Placeholder
    
    if (mode == "weights") {
        // =====================================================================
        // MODO 1: EVOLUCIÓN DE PESOS SOLAMENTE (Nota 1.75)
        // =====================================================================
        std::cout << "========================================\n";
        std::cout << "MODO: Evolución de PESOS (Nota 1.75)\n";
        std::cout << "========================================\n\n";
        
        std::vector<int> topology = parseArchitecture(archStr);
        
        // Verificar/ajustar arquitectura
        if (topology.front() != static_cast<int>(X[0].size())) {
            std::cout << "Ajustando entrada a " << X[0].size() << " features\n";
            topology[0] = X[0].size();
        }
        if (topology.back() != static_cast<int>(Y[0].size())) {
            std::cout << "Ajustando salida a " << Y[0].size() << " clases\n";
            topology.back() = Y[0].size();
        }
        
        std::cout << "Arquitectura: ";
        for (int t : topology) std::cout << t << " ";
        std::cout << "\n\n";
        
        config.evolveArchitecture = false;
        GeneticAlgorithm ga(config, topology, activation);
        best = ga.evolve(Xtrain, Ytrain);
        
    } else if (mode == "neuro") {
        // =====================================================================
        // MODO 2: NEUROEVOLUCIÓN (Nota 2.00)
        // =====================================================================
        std::cout << "========================================\n";
        std::cout << "MODO: NEUROEVOLUCIÓN (Nota 2.00)\n";
        std::cout << "========================================\n\n";
        
        if (inputSize == 0) inputSize = X[0].size();
        if (outputSize == 0) outputSize = Y[0].size();
        
        std::cout << "Entradas: " << inputSize << "\n";
        std::cout << "Salidas: " << outputSize << "\n";
        std::cout << "Capas ocultas: " << config.minHiddenLayers << "-" 
                  << config.maxHiddenLayers << "\n";
        std::cout << "Neuronas/capa: " << config.minNeuronsPerLayer << "-" 
                  << config.maxNeuronsPerLayer << "\n\n";
        
        config.evolveArchitecture = true;
        GeneticAlgorithm ga(config, inputSize, outputSize, activation);
        best = ga.evolve(Xtrain, Ytrain);
        
    } else {
        std::cerr << "Error: Modo desconocido: " << mode << "\n";
        return 1;
    }
    
    // Evaluar en test
    std::cout << "\n========================================\n";
    std::cout << "RESULTADOS FINALES\n";
    std::cout << "========================================\n";
    
    double trainAcc = evaluateAccuracy(best, Xtrain, Ytrain);
    double testAcc = evaluateAccuracy(best, Xtest, Ytest);
    
    std::cout << "Mejor arquitectura: " << best.getArchitectureString() << "\n";
    std::cout << "Parámetros totales: " << best.getTotalParameters() << "\n";
    std::cout << "Fitness final: " << best.getFitness() << "\n";
    std::cout << "Accuracy Train: " << trainAcc << "%\n";
    std::cout << "Accuracy Test: " << testAcc << "%\n";
    
    // Guardar modelo
    if (!savePath.empty()) {
        best.save(savePath);
        std::cout << "\nModelo guardado en: " << savePath << "\n";
    }
    
    return 0;
}
