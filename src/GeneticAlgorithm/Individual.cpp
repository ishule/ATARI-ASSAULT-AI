/*
 * INDIVIDUAL.CPP - Implementación del individuo (red neuronal)
 */

#include "GeneticAlgorithm/Individual.hpp"
#include "ActivationFunctions.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <sstream>

// Generador de números aleatorios (compartido)
std::mt19937& Individual::getRandomEngine() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return gen;
}

// ============================================
// CONSTRUCTORES
// ============================================

// Constructor básico: arquitectura fija, pesos random
Individual::Individual(const std::vector<int>& topology, ActivationType act)
    : fitness_(0.0), topology_(topology), activation_(act) {
    initializeRandomWeights();
}

// Constructor con pesos dados (para hijos del crossover)
Individual::Individual(const std::vector<int>& topology,
                       const VecWeights& weights,
                       const std::vector<std::vector<double>>& biases,
                       ActivationType act)
    : fitness_(0.0), topology_(topology), weights_(weights), 
      biases_(biases), activation_(act) {
}

// Constructor para neuroevolución: genera arquitectura random
Individual::Individual(int inputSize, int outputSize,
                       int minHiddenLayers, int maxHiddenLayers,
                       int minNeurons, int maxNeurons,
                       ActivationType act)
    : fitness_(0.0), activation_(act) {
    
    topology_ = generateRandomTopology(inputSize, outputSize,
                                        minHiddenLayers, maxHiddenLayers,
                                        minNeurons, maxNeurons);
    initializeRandomWeights();
}

// Constructor copia
Individual::Individual(const Individual& other)
    : fitness_(other.fitness_), topology_(other.topology_),
      weights_(other.weights_), biases_(other.biases_),
      activation_(other.activation_) {
}

// Operador de asignación
Individual& Individual::operator=(const Individual& other) {
    if (this != &other) {
        fitness_ = other.fitness_;
        topology_ = other.topology_;
        weights_ = other.weights_;
        biases_ = other.biases_;
        activation_ = other.activation_;
    }
    return *this;
}

// ============================================
// INICIALIZACIÓN DE PESOS (Xavier/Glorot)
// ============================================

void Individual::initializeRandomWeights() {
    weights_.clear();
    biases_.clear();
    
    auto& gen = getRandomEngine();
    
    // Para cada capa (menos la última)
    for (size_t i = 0; i < topology_.size() - 1; ++i) {
        int inputSize = topology_[i];
        int outputSize = topology_[i + 1];
        
        // Xavier: pesos en rango [-limit, limit]
        double limit = std::sqrt(6.0 / (inputSize + outputSize));
        std::uniform_real_distribution<> dis(-limit, limit);
        
        // Crear matriz de pesos para esta capa
        std::vector<std::vector<double>> layerWeights(outputSize);
        for (int j = 0; j < outputSize; ++j) {
            layerWeights[j].resize(inputSize);
            for (int k = 0; k < inputSize; ++k) {
                layerWeights[j][k] = dis(gen);
            }
        }
        weights_.push_back(layerWeights);
        
        // Biases a 0
        biases_.push_back(std::vector<double>(outputSize, 0.0));
    }
}

// Genera arquitectura random para neuroevolución
std::vector<int> Individual::generateRandomTopology(int inputSize, int outputSize,
                                                     int minHiddenLayers, int maxHiddenLayers,
                                                     int minNeurons, int maxNeurons) {
    auto& gen = getRandomEngine();
    std::uniform_int_distribution<> layerDist(minHiddenLayers, maxHiddenLayers);
    std::uniform_int_distribution<> neuronDist(minNeurons, maxNeurons);
    
    std::vector<int> topology;
    topology.push_back(inputSize);  // Entrada fija
    
    // Capas ocultas random
    int numHiddenLayers = layerDist(gen);
    for (int i = 0; i < numHiddenLayers; ++i) {
        topology.push_back(neuronDist(gen));
    }
    
    topology.push_back(outputSize);  // Salida fija
    return topology;
}

// ============================================
// CROSSOVER
// ============================================

// Crossover normal: mezcla pesos de dos padres
// Requiere misma arquitectura
Individual Individual::crossover(const Individual& other, double mutationRate) const {
    if (topology_ != other.topology_) {
        throw std::runtime_error("Crossover requiere misma arquitectura");
    }
    
    auto& gen = getRandomEngine();
    std::uniform_real_distribution<> prob(0.0, 1.0);
    std::normal_distribution<> mutation(0.0, 0.5);
    
    VecWeights childWeights;
    std::vector<std::vector<double>> childBiases;
    
    // Para cada capa
    for (size_t layer = 0; layer < weights_.size(); ++layer) {
        std::vector<std::vector<double>> layerWeights(weights_[layer].size());
        std::vector<double> layerBiases(biases_[layer].size());
        
        for (size_t j = 0; j < weights_[layer].size(); ++j) {
            layerWeights[j].resize(weights_[layer][j].size());
            
            for (size_t k = 0; k < weights_[layer][j].size(); ++k) {
                // Mutación o elegir de un padre
                if (prob(gen) < mutationRate) {
                    layerWeights[j][k] = mutation(gen);  // Peso nuevo random
                } else {
                    // 50/50 de cada padre
                    layerWeights[j][k] = (prob(gen) < 0.5) ? 
                        weights_[layer][j][k] : other.weights_[layer][j][k];
                }
            }
            
            // Lo mismo para bias
            if (prob(gen) < mutationRate) {
                layerBiases[j] = mutation(gen) * 0.1;
            } else {
                layerBiases[j] = (prob(gen) < 0.5) ? 
                    biases_[layer][j] : other.biases_[layer][j];
            }
        }
        
        childWeights.push_back(layerWeights);
        childBiases.push_back(layerBiases);
    }
    
    return Individual(topology_, childWeights, childBiases, activation_);
}

// Crossover para neuroevolución: puede cambiar arquitectura
Individual Individual::crossoverNeuroevolution(const Individual& other,
                                               double mutationRate,
                                               double archMutationRate) const {
    auto& gen = getRandomEngine();
    std::uniform_real_distribution<> prob(0.0, 1.0);
    std::uniform_real_distribution<> weightDist(-0.5, 0.5);
    
    // El mejor padre "manda" más
    std::vector<int> newTopology;
    const Individual* betterParent;
    const Individual* worseParent;
    
    if (fitness_ > other.fitness_) {
        newTopology = topology_;
        betterParent = this;
        worseParent = &other;
    } else {
        newTopology = other.topology_;
        betterParent = &other;
        worseParent = this;
    }
    
    // Posible mutación de arquitectura
    if (prob(gen) < archMutationRate) {
        std::uniform_int_distribution<> choice(0, 3);
        int mutation = choice(gen);
        
        if (mutation == 0 && newTopology.size() > 3) {
            // Quitar una capa oculta
            std::uniform_int_distribution<> layerDist(1, newTopology.size() - 2);
            int layerToRemove = layerDist(gen);
            newTopology.erase(newTopology.begin() + layerToRemove);
        } else if (mutation == 1 && newTopology.size() < 6) {
            // Añadir una capa oculta
            std::uniform_int_distribution<> layerDist(1, newTopology.size() - 1);
            std::uniform_int_distribution<> neuronDist(16, 64);
            int insertPos = layerDist(gen);
            newTopology.insert(newTopology.begin() + insertPos, neuronDist(gen));
        } else if (mutation == 2 && newTopology.size() > 2) {
            // Añadir neuronas a una capa
            std::uniform_int_distribution<> layerDist(1, newTopology.size() - 2);
            std::uniform_int_distribution<> addNeurons(4, 16);
            int layerToModify = layerDist(gen);
            newTopology[layerToModify] = std::min(128, newTopology[layerToModify] + addNeurons(gen));
        } else if (newTopology.size() > 2) {
            // Quitar neuronas de una capa
            std::uniform_int_distribution<> layerDist(1, newTopology.size() - 2);
            std::uniform_int_distribution<> removeNeurons(2, 8);
            int layerToModify = layerDist(gen);
            newTopology[layerToModify] = std::max(4, newTopology[layerToModify] - removeNeurons(gen));
        }
    }
    
    // Construir pesos heredando de los padres
    VecWeights childWeights;
    std::vector<std::vector<double>> childBiases;
    
    for (size_t layer = 0; layer < newTopology.size() - 1; ++layer) {
        int outputSize = newTopology[layer + 1];
        int inputSize = newTopology[layer];
        
        std::vector<std::vector<double>> layerWeights(outputSize);
        std::vector<double> layerBiases(outputSize, 0.0);
        
        for (int j = 0; j < outputSize; ++j) {
            layerWeights[j].resize(inputSize);
            
            for (int k = 0; k < inputSize; ++k) {
                bool inherited = false;
                
                // Si el peso existe en el mejor padre, heredar
                if (layer < betterParent->weights_.size() &&
                    static_cast<size_t>(j) < betterParent->weights_[layer].size() &&
                    static_cast<size_t>(k) < betterParent->weights_[layer][j].size()) {
                    
                    if (prob(gen) < mutationRate) {
                        // Mutar: añadir ruido
                        layerWeights[j][k] = betterParent->weights_[layer][j][k] + weightDist(gen) * 0.3;
                    } else if (prob(gen) < 0.7) {
                        // 70% del mejor padre
                        layerWeights[j][k] = betterParent->weights_[layer][j][k];
                    } else if (layer < worseParent->weights_.size() &&
                               static_cast<size_t>(j) < worseParent->weights_[layer].size() &&
                               static_cast<size_t>(k) < worseParent->weights_[layer][j].size()) {
                        // 30% del otro padre
                        layerWeights[j][k] = worseParent->weights_[layer][j][k];
                    } else {
                        layerWeights[j][k] = betterParent->weights_[layer][j][k];
                    }
                    inherited = true;
                }
                
                // Si no existe, inicializar con Xavier
                if (!inherited) {
                    double limit = std::sqrt(6.0 / (inputSize + outputSize));
                    std::uniform_real_distribution<> xavier(-limit, limit);
                    layerWeights[j][k] = xavier(gen);
                }
            }
            
            // Heredar bias si existe
            if (layer < betterParent->biases_.size() &&
                static_cast<size_t>(j) < betterParent->biases_[layer].size()) {
                layerBiases[j] = betterParent->biases_[layer][j];
            }
        }
        
        childWeights.push_back(layerWeights);
        childBiases.push_back(layerBiases);
    }
    
    return Individual(newTopology, childWeights, childBiases, activation_);
}

// ============================================
// MUTACIÓN
// ============================================

// Muta los pesos añadiendo ruido gaussiano
void Individual::mutateWeights(double mutationRate, double mutationStrength) {
    auto& gen = getRandomEngine();
    std::uniform_real_distribution<> prob(0.0, 1.0);
    std::normal_distribution<> mutation(0.0, mutationStrength);
    
    for (size_t layer = 0; layer < weights_.size(); ++layer) {
        for (size_t j = 0; j < weights_[layer].size(); ++j) {
            for (size_t k = 0; k < weights_[layer][j].size(); ++k) {
                if (prob(gen) < mutationRate) {
                    weights_[layer][j][k] += mutation(gen);
                    // Limitar para que no explote
                    weights_[layer][j][k] = std::max(-5.0, std::min(5.0, weights_[layer][j][k]));
                }
            }
            
            if (prob(gen) < mutationRate) {
                biases_[layer][j] += mutation(gen) * 0.1;
                biases_[layer][j] = std::max(-2.0, std::min(2.0, biases_[layer][j]));
            }
        }
    }
}

// Muta la arquitectura: añade/quita neuronas o capas
void Individual::mutateArchitecture(double mutationRate, int minNeurons, int maxNeurons) {
    auto& gen = getRandomEngine();
    std::uniform_real_distribution<> prob(0.0, 1.0);
    std::uniform_real_distribution<> weightDist(-0.5, 0.5);
    
    if (prob(gen) >= mutationRate) return;  // No mutar
    
    std::uniform_int_distribution<> choice(0, 3);
    int mutation = choice(gen);
    
    switch (mutation) {
        case 0:  // Añadir neurona
            if (topology_.size() > 2) {
                std::uniform_int_distribution<> layerDist(1, topology_.size() - 2);
                int layerIdx = layerDist(gen);
                if (topology_[layerIdx] < maxNeurons) {
                    topology_[layerIdx]++;
                    
                    // Añadir pesos para la nueva neurona
                    if (layerIdx > 0 && static_cast<size_t>(layerIdx - 1) < weights_.size()) {
                        std::vector<double> newWeights(weights_[layerIdx - 1][0].size());
                        for (auto& w : newWeights) w = weightDist(gen);
                        weights_[layerIdx - 1].push_back(newWeights);
                        biases_[layerIdx - 1].push_back(0.0);
                    }
                    
                    // Añadir columna en la capa siguiente
                    if (static_cast<size_t>(layerIdx) < weights_.size()) {
                        for (auto& neuronWeights : weights_[layerIdx]) {
                            neuronWeights.push_back(weightDist(gen));
                        }
                    }
                }
            }
            break;
            
        case 1:  // Eliminar neurona
            if (topology_.size() > 2) {
                std::uniform_int_distribution<> layerDist(1, topology_.size() - 2);
                int layerIdx = layerDist(gen);
                if (topology_[layerIdx] > minNeurons) {
                    std::uniform_int_distribution<> neuronDist(0, topology_[layerIdx] - 1);
                    int neuronToRemove = neuronDist(gen);
                    
                    topology_[layerIdx]--;
                    
                    // Eliminar fila de pesos
                    if (layerIdx > 0 && static_cast<size_t>(layerIdx - 1) < weights_.size()) {
                        if (static_cast<size_t>(neuronToRemove) < weights_[layerIdx - 1].size()) {
                            weights_[layerIdx - 1].erase(weights_[layerIdx - 1].begin() + neuronToRemove);
                            biases_[layerIdx - 1].erase(biases_[layerIdx - 1].begin() + neuronToRemove);
                        }
                    }
                    
                    // Eliminar columna en capa siguiente
                    if (static_cast<size_t>(layerIdx) < weights_.size()) {
                        for (auto& neuronWeights : weights_[layerIdx]) {
                            if (static_cast<size_t>(neuronToRemove) < neuronWeights.size()) {
                                neuronWeights.erase(neuronWeights.begin() + neuronToRemove);
                            }
                        }
                    }
                }
            }
            break;
            
        case 2:  // Añadir capa
            if (topology_.size() < 8) {
                std::uniform_int_distribution<> posDist(1, topology_.size() - 1);
                std::uniform_int_distribution<> neuronDist(minNeurons, maxNeurons);
                int pos = posDist(gen);
                int newNeurons = neuronDist(gen);
                
                int prevSize = topology_[pos - 1];
                
                topology_.insert(topology_.begin() + pos, newNeurons);
                
                // Crear pesos para la nueva capa
                std::vector<std::vector<double>> newLayerWeights(newNeurons);
                for (int i = 0; i < newNeurons; ++i) {
                    newLayerWeights[i].resize(prevSize);
                    for (int j = 0; j < prevSize; ++j) {
                        newLayerWeights[i][j] = weightDist(gen);
                    }
                }
                weights_.insert(weights_.begin() + (pos - 1), newLayerWeights);
                biases_.insert(biases_.begin() + (pos - 1), std::vector<double>(newNeurons, 0.0));
                
                // Ajustar pesos de la capa siguiente
                if (static_cast<size_t>(pos) < weights_.size()) {
                    for (auto& neuronWeights : weights_[pos]) {
                        neuronWeights.resize(newNeurons);
                        for (int i = prevSize; i < newNeurons; ++i) {
                            neuronWeights[i] = weightDist(gen);
                        }
                    }
                }
            }
            break;
            
        case 3:  // Eliminar capa
            if (topology_.size() > 3) {
                std::uniform_int_distribution<> posDist(1, topology_.size() - 2);
                int pos = posDist(gen);
                
                topology_.erase(topology_.begin() + pos);
                
                if (static_cast<size_t>(pos - 1) < weights_.size()) {
                    weights_.erase(weights_.begin() + (pos - 1));
                    biases_.erase(biases_.begin() + (pos - 1));
                }
                
                // Reconectar
                if (static_cast<size_t>(pos - 1) < weights_.size() && pos >= 1) {
                    int newInputSize = topology_[pos - 1];
                    for (auto& neuronWeights : weights_[pos - 1]) {
                        neuronWeights.resize(newInputSize);
                        for (auto& w : neuronWeights) {
                            if (w == 0.0) w = weightDist(gen);
                        }
                    }
                }
            }
            break;
    }
}

// ============================================
// PREDICCIÓN (Forward Pass)
// ============================================

std::vector<double> Individual::predict(const std::vector<double>& input) const {
    return forwardPass(input);
}

std::vector<double> Individual::forwardPass(const std::vector<double>& input) const {
    std::vector<double> current = input;
    
    for (size_t layer = 0; layer < weights_.size(); ++layer) {
        std::vector<double> output(weights_[layer].size());
        
        for (size_t j = 0; j < weights_[layer].size(); ++j) {
            double sum = biases_[layer][j];
            for (size_t k = 0; k < current.size(); ++k) {
                sum += weights_[layer][j][k] * current[k];
            }
            
            // Última capa: sigmoid. Otras: la que toque
            bool isLastLayer = (layer == weights_.size() - 1);
            if (isLastLayer) {
                output[j] = 1.0 / (1.0 + std::exp(-sum));  // Sigmoid
            } else {
                output[j] = ActivationFunctions::apply(sum, activation_);
            }
        }
        
        current = output;
    }
    
    return current;
}

// ============================================
// FITNESS
// ============================================

// Calcula fitness con dataset (accuracy)
double Individual::calculateFitness(const std::vector<std::vector<double>>& X,
                                    const std::vector<std::vector<double>>& Y) {
    if (X.empty() || Y.empty() || X.size() != Y.size()) {
        fitness_ = 0.0;
        return fitness_;
    }
    
    int correct = 0;
    double totalError = 0.0;
    
    for (size_t i = 0; i < X.size(); ++i) {
        std::vector<double> pred = predict(X[i]);
        
        if (pred.size() != Y[i].size()) continue;
        
        // Calcular error
        for (size_t j = 0; j < pred.size(); ++j) {
            double error = pred[j] - Y[i][j];
            totalError += error * error;
        }
        
        // ¿Clasificó bien?
        if (Y[i].size() == 1) {
            // Binaria
            double predicted = (pred[0] >= 0.5) ? 1.0 : 0.0;
            if (std::abs(predicted - Y[i][0]) < 0.1) correct++;
        } else {
            // Multiclase: comparar argmax
            auto predMax = std::max_element(pred.begin(), pred.end());
            auto trueMax = std::max_element(Y[i].begin(), Y[i].end());
            if (std::distance(pred.begin(), predMax) == std::distance(Y[i].begin(), trueMax)) {
                correct++;
            }
        }
    }
    
    double accuracy = static_cast<double>(correct) / X.size();
    double mse = totalError / X.size();
    
    // Fitness = accuracy*100 - error
    fitness_ = accuracy * 100.0 - mse;
    
    return fitness_;
}

// Fitness para juegos (placeholder)
double Individual::calculateGameFitness(int maxSteps) {
    // Aquí iría la lógica del juego (ALE etc)
    fitness_ = 0.0;
    return fitness_;
}

// ============================================
// UTILIDADES
// ============================================

int Individual::getTotalParameters() const {
    int total = 0;
    for (size_t layer = 0; layer < weights_.size(); ++layer) {
        for (const auto& row : weights_[layer]) {
            total += row.size();
        }
        total += biases_[layer].size();
    }
    return total;
}

std::string Individual::getArchitectureString() const {
    std::stringstream ss;
    for (size_t i = 0; i < topology_.size(); ++i) {
        ss << topology_[i];
        if (i < topology_.size() - 1) ss << "-";
    }
    return ss.str();
}

// ============================================
// GUARDAR / CARGAR
// ============================================

void Individual::save(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file) throw std::runtime_error("No se pudo crear: " + filepath);
    
    // Topología
    file << topology_.size() << "\n";
    for (int t : topology_) file << t << " ";
    file << "\n";
    
    // Activación
    file << static_cast<int>(activation_) << "\n";
    
    // Fitness
    file << fitness_ << "\n";
    
    // Pesos y biases
    for (size_t layer = 0; layer < weights_.size(); ++layer) {
        for (const auto& row : weights_[layer]) {
            for (double w : row) file << w << " ";
            file << "\n";
        }
        for (double b : biases_[layer]) file << b << " ";
        file << "\n";
    }
}

void Individual::load(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file) throw std::runtime_error("No se pudo abrir: " + filepath);
    
    // Leer topología
    size_t numLayers;
    file >> numLayers;
    topology_.resize(numLayers);
    for (size_t i = 0; i < numLayers; ++i) {
        file >> topology_[i];
    }
    
    // Leer activación
    int act;
    file >> act;
    activation_ = static_cast<ActivationType>(act);
    
    // Leer fitness
    file >> fitness_;
    
    // Leer pesos
    weights_.clear();
    biases_.clear();
    
    for (size_t layer = 0; layer < numLayers - 1; ++layer) {
        int outputSize = topology_[layer + 1];
        int inputSize = topology_[layer];
        
        std::vector<std::vector<double>> layerWeights(outputSize);
        for (int j = 0; j < outputSize; ++j) {
            layerWeights[j].resize(inputSize);
            for (int k = 0; k < inputSize; ++k) {
                file >> layerWeights[j][k];
            }
        }
        weights_.push_back(layerWeights);
        
        std::vector<double> layerBiases(outputSize);
        for (int j = 0; j < outputSize; ++j) {
            file >> layerBiases[j];
        }
        biases_.push_back(layerBiases);
    }
}
