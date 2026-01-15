#include "GeneticAlgorithm/Individual.hpp"
#include "ActivationFunctions.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <sstream>
using namespace std;

// Generador de números aleatorios
mt19937& Individual::getRandomEngine() {
    static random_device rd;
    static mt19937 gen(rd());
    return gen;
}

// Constructor básico: arquitectura fija, pesos random
Individual::Individual(const vector<int>& topology, ActivationType act)
    : fitness_(0.0), topology_(topology), activation_(act) { initializeRandomWeights(); }

// Constructor con pesos dados (para hijos del crossover)
Individual::Individual(const vector<int>& topology, const VecWeights& weights, const vector<vector<double>>& biases, ActivationType act)
    : fitness_(0.0), topology_(topology), weights_(weights), biases_(biases), activation_(act) {}

// Constructor para neuroevolución: genera arquitectura random
Individual::Individual(int inputSize, int outputSize, int minHiddenLayers, int maxHiddenLayers, int minNeurons, int maxNeurons, ActivationType act)
    : fitness_(0.0), activation_(act) {
    topology_ = generateRandomTopology(inputSize, outputSize, minHiddenLayers, maxHiddenLayers, minNeurons, maxNeurons);
    initializeRandomWeights();
}

// Constructor copia
Individual::Individual(const Individual& other)
    : fitness_(other.fitness_), topology_(other.topology_), weights_(other.weights_), biases_(other.biases_), activation_(other.activation_) {}

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
    for (size_t i = 0; i < topology_.size() - 1; ++i) {
        int inputSize = topology_[i];
        int outputSize = topology_[i + 1];
        double limit = sqrt(6.0 / (inputSize + outputSize));
        uniform_real_distribution<> dis(-limit, limit);
        vector<vector<double>> layerWeights(outputSize);
        for (int j = 0; j < outputSize; ++j) {
            layerWeights[j].resize(inputSize);
            for (int k = 0; k < inputSize; ++k) {
                layerWeights[j][k] = dis(gen);
            }
        }
        weights_.push_back(layerWeights);
        biases_.push_back(vector<double>(outputSize, 0.0));
    }
}

// Genera arquitectura random para neuroevolución
vector<int> Individual::generateRandomTopology(int inputSize, int outputSize, int minHiddenLayers, int maxHiddenLayers, int minNeurons, int maxNeurons) {
    auto& gen = getRandomEngine();
    uniform_int_distribution<> layerDist(minHiddenLayers, maxHiddenLayers);
    uniform_int_distribution<> neuronDist(minNeurons, maxNeurons);
    vector<int> topology;
    topology.push_back(inputSize);
    int numHiddenLayers = layerDist(gen);
    for (int i = 0; i < numHiddenLayers; ++i) {
        topology.push_back(neuronDist(gen));
    }
    topology.push_back(outputSize);
    return topology;
}

// Cruza dos individuos (con la misma arquitectura) para crear un hijo con pesos mezclados y posibles mutaciones
Individual Individual::crossover(const Individual& other, double mutationRate) const {
    if (topology_ != other.topology_) {
        throw runtime_error("Crossover requiere misma arquitectura");
    }
    auto& gen = getRandomEngine();
    uniform_real_distribution<> prob(0.0, 1.0);
    normal_distribution<> mutation(0.0, 0.5);
    VecWeights childWeights;
    vector<vector<double>> childBiases;
    for (size_t layer = 0; layer < weights_.size(); ++layer) {
        vector<vector<double>> layerWeights(weights_[layer].size());
        vector<double> layerBiases(biases_[layer].size());
        for (size_t j = 0; j < weights_[layer].size(); ++j) {
            layerWeights[j].resize(weights_[layer][j].size());
            for (size_t k = 0; k < weights_[layer][j].size(); ++k) {
                // Con probabilidad mutationRate, muta el peso; si no, hereda aleatoriamente de uno de los padres
                if (prob(gen) < mutationRate) {
                    layerWeights[j][k] = mutation(gen);
                } else {
                    layerWeights[j][k] = (prob(gen) < 0.5) ? weights_[layer][j][k] : other.weights_[layer][j][k];
                }
            }
            // Con probabilidad mutationRate, muta el bias; si no, hereda aleatoriamente de uno de los padres
            if (prob(gen) < mutationRate) {
                layerBiases[j] = mutation(gen) * 0.1;
            } else {
                layerBiases[j] = (prob(gen) < 0.5) ? biases_[layer][j] : other.biases_[layer][j];
            }
        }
        childWeights.push_back(layerWeights);
        childBiases.push_back(layerBiases);
    }
    // Devuelve un nuevo individuo con la arquitectura y pesos/biases resultantes
    return Individual(topology_, childWeights, childBiases, activation_);
}

// Cruza dos individuos permitiendo mutación de arquitectura y pesos (neuroevolución)
Individual Individual::crossoverNeuroevolution(const Individual& other, double mutationRate, double archMutationRate) const {
    auto& gen = getRandomEngine();
    uniform_real_distribution<> prob(0.0, 1.0);
    uniform_real_distribution<> weightDist(-0.5, 0.5);
    vector<int> newTopology;
    const Individual* betterParent;
    const Individual* worseParent;
    // Selecciona el padre con mejor fitness como base
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
        uniform_int_distribution<> choice(0, 3);
        int mutation = choice(gen);
        // Elimina una capa oculta
        if (mutation == 0 && newTopology.size() > 3) {
            uniform_int_distribution<> layerDist(1, newTopology.size() - 2);
            int layerToRemove = layerDist(gen);
            newTopology.erase(newTopology.begin() + layerToRemove);
        // Inserta una nueva capa oculta
        } else if (mutation == 1 && newTopology.size() < 6) {
            uniform_int_distribution<> layerDist(1, newTopology.size() - 1);
            uniform_int_distribution<> neuronDist(16, 64);
            int insertPos = layerDist(gen);
            newTopology.insert(newTopology.begin() + insertPos, neuronDist(gen));
        // Añade neuronas a una capa existente
        } else if (mutation == 2 && newTopology.size() > 2) {
            uniform_int_distribution<> layerDist(1, newTopology.size() - 2);
            uniform_int_distribution<> addNeurons(4, 16);
            int layerToModify = layerDist(gen);
            newTopology[layerToModify] = min(128, newTopology[layerToModify] + addNeurons(gen));
        // Quita neuronas de una capa existente
        } else if (newTopology.size() > 2) {
            uniform_int_distribution<> layerDist(1, newTopology.size() - 2);
            uniform_int_distribution<> removeNeurons(2, 8);
            int layerToModify = layerDist(gen);
            newTopology[layerToModify] = max(4, newTopology[layerToModify] - removeNeurons(gen));
        }
    }
    VecWeights childWeights;
    vector<vector<double>> childBiases;
    // Para cada capa, mezcla pesos y biases de los padres
    for (size_t layer = 0; layer < newTopology.size() - 1; ++layer) {
        int outputSize = newTopology[layer + 1];
        int inputSize = newTopology[layer];
        vector<vector<double>> layerWeights(outputSize);
        vector<double> layerBiases(outputSize, 0.0);
        for (int j = 0; j < outputSize; ++j) {
            layerWeights[j].resize(inputSize);
            for (int k = 0; k < inputSize; ++k) {
                bool inherited = false;
                // Si el padre tiene ese peso, lo hereda o lo muta
                if (layer < betterParent->weights_.size() && static_cast<size_t>(j) < betterParent->weights_[layer].size() && static_cast<size_t>(k) < betterParent->weights_[layer][j].size()) {
                    if (prob(gen) < mutationRate) {
                        // Mutación del peso
                        layerWeights[j][k] = betterParent->weights_[layer][j][k] + weightDist(gen) * 0.3;
                    } else if (prob(gen) < 0.7) {
                        // Hereda del mejor padre
                        layerWeights[j][k] = betterParent->weights_[layer][j][k];
                    } else if (layer < worseParent->weights_.size() && static_cast<size_t>(j) < worseParent->weights_[layer].size() && static_cast<size_t>(k) < worseParent->weights_[layer][j].size()) {
                        // Hereda del peor padre
                        layerWeights[j][k] = worseParent->weights_[layer][j][k];
                    } else {
                        layerWeights[j][k] = betterParent->weights_[layer][j][k];
                    }
                    inherited = true;
                }
                // Si no existe ese peso en los padres, inicializa aleatoriamente (Xavier)
                if (!inherited) {
                    double limit = sqrt(6.0 / (inputSize + outputSize));
                    uniform_real_distribution<> xavier(-limit, limit);
                    layerWeights[j][k] = xavier(gen);
                }
            }
            // Hereda bias si existe
            if (layer < betterParent->biases_.size() && static_cast<size_t>(j) < betterParent->biases_[layer].size()) {
                layerBiases[j] = betterParent->biases_[layer][j];
            }
        }
        childWeights.push_back(layerWeights);
        childBiases.push_back(layerBiases);
    }
    // Devuelve el nuevo individuo hijo
    return Individual(newTopology, childWeights, childBiases, activation_);
}

void Individual::mutateWeights(double mutationRate, double mutationStrength) {
    auto& gen = getRandomEngine();
    uniform_real_distribution<> prob(0.0, 1.0);
    normal_distribution<> mutation(0.0, mutationStrength);
    for (size_t layer = 0; layer < weights_.size(); ++layer) {
        for (size_t j = 0; j < weights_[layer].size(); ++j) {
            for (size_t k = 0; k < weights_[layer][j].size(); ++k) {
                // Con probabilidad mutationRate, modifica el peso con una perturbación aleatoria
                if (prob(gen) < mutationRate) {
                    weights_[layer][j][k] += mutation(gen);
                    weights_[layer][j][k] = max(-5.0, min(5.0, weights_[layer][j][k]));
                }
            }
            // Con probabilidad mutationRate, modifica el bias
            if (prob(gen) < mutationRate) {
                biases_[layer][j] += mutation(gen) * 0.1;
                biases_[layer][j] = max(-2.0, min(2.0, biases_[layer][j]));
            }
        }
    }
}

void Individual::mutateArchitecture(double mutationRate, int minNeurons, int maxNeurons) {
    auto& gen = getRandomEngine();
    uniform_real_distribution<> prob(0.0, 1.0);
    uniform_real_distribution<> weightDist(-0.5, 0.5);
    if (prob(gen) >= mutationRate) return;
    uniform_int_distribution<> choice(0, 3);
    int mutation = choice(gen);
    switch (mutation) {
        case 0:
            // Añade una neurona a una capa oculta aleatoria
            if (topology_.size() > 2) {
                uniform_int_distribution<> layerDist(1, topology_.size() - 2);
                int layerIdx = layerDist(gen);
                if (topology_[layerIdx] < maxNeurons) {
                    topology_[layerIdx]++;
                    if (layerIdx > 0 && static_cast<size_t>(layerIdx - 1) < weights_.size()) {
                        vector<double> newWeights(weights_[layerIdx - 1][0].size());
                        for (auto& w : newWeights) w = weightDist(gen);
                        weights_[layerIdx - 1].push_back(newWeights);
                        biases_[layerIdx - 1].push_back(0.0);
                    }
                    if (static_cast<size_t>(layerIdx) < weights_.size()) {
                        for (auto& neuronWeights : weights_[layerIdx]) {
                            neuronWeights.push_back(weightDist(gen));
                        }
                    }
                }
            }
            break;
        case 1:
            // Elimina una neurona de una capa oculta aleatoria
            if (topology_.size() > 2) {
                uniform_int_distribution<> layerDist(1, topology_.size() - 2);
                int layerIdx = layerDist(gen);
                if (topology_[layerIdx] > minNeurons) {
                    uniform_int_distribution<> neuronDist(0, topology_[layerIdx] - 1);
                    int neuronToRemove = neuronDist(gen);
                    topology_[layerIdx]--;
                    if (layerIdx > 0 && static_cast<size_t>(layerIdx - 1) < weights_.size()) {
                        if (static_cast<size_t>(neuronToRemove) < weights_[layerIdx - 1].size()) {
                            weights_[layerIdx - 1].erase(weights_[layerIdx - 1].begin() + neuronToRemove);
                            biases_[layerIdx - 1].erase(biases_[layerIdx - 1].begin() + neuronToRemove);
                        }
                    }
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
        case 2:
            // Inserta una nueva capa oculta en una posición aleatoria
            if (topology_.size() < 8) {
                uniform_int_distribution<> posDist(1, topology_.size() - 1);
                uniform_int_distribution<> neuronDist(minNeurons, maxNeurons);
                int pos = posDist(gen);
                int newNeurons = neuronDist(gen);
                int prevSize = topology_[pos - 1];
                topology_.insert(topology_.begin() + pos, newNeurons);
                vector<vector<double>> newLayerWeights(newNeurons);
                for (int i = 0; i < newNeurons; ++i) {
                    newLayerWeights[i].resize(prevSize);
                    for (int j = 0; j < prevSize; ++j) {
                        newLayerWeights[i][j] = weightDist(gen);
                    }
                }
                weights_.insert(weights_.begin() + (pos - 1), newLayerWeights);
                biases_.insert(biases_.begin() + (pos - 1), vector<double>(newNeurons, 0.0));
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
        case 3:
            // Elimina una capa oculta aleatoria
            if (topology_.size() > 3) {
                uniform_int_distribution<> posDist(1, topology_.size() - 2);
                int pos = posDist(gen);
                topology_.erase(topology_.begin() + pos);
                if (static_cast<size_t>(pos - 1) < weights_.size()) {
                    weights_.erase(weights_.begin() + (pos - 1));
                    biases_.erase(biases_.begin() + (pos - 1));
                }
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

vector<double> Individual::predict(const vector<double>& input) const { return forwardPass(input); }

vector<double> Individual::forwardPass(const vector<double>& input) const {
    vector<double> current = input;
    for (size_t layer = 0; layer < weights_.size(); ++layer) {
        vector<double> output(weights_[layer].size());
        for (size_t j = 0; j < weights_[layer].size(); ++j) {
            double sum = biases_[layer][j];
            for (size_t k = 0; k < current.size(); ++k) {
                sum += weights_[layer][j][k] * current[k];
            }
            bool isLastLayer = (layer == weights_.size() - 1);
            // Si es la última capa, aplica sigmoide; si no, la función de activación seleccionada
            if (isLastLayer) {
                output[j] = 1.0 / (1.0 + exp(-sum));
            } else {
                output[j] = ActivationFunctions::apply(sum, activation_);
            }
        }
        current = output;
    }
    return current;
}

double Individual::calculateFitness(const vector<vector<double>>& X, const vector<vector<double>>& Y) {
    if (X.empty() || Y.empty() || X.size() != Y.size()) {
        fitness_ = 0.0;
        return fitness_;
    }
    int correct = 0;
    double totalError = 0.0;
    for (size_t i = 0; i < X.size(); ++i) {
        vector<double> pred = predict(X[i]);
        if (pred.size() != Y[i].size()) continue;
        for (size_t j = 0; j < pred.size(); ++j) {
            double error = pred[j] - Y[i][j];
            totalError += error * error;
        }
        // Si es clasificación binaria
        if (Y[i].size() == 1) {
            double predicted = (pred[0] >= 0.5) ? 1.0 : 0.0;
            if (abs(predicted - Y[i][0]) < 0.1) correct++;
        } else {
            // Si es clasificación multiclase
            auto predMax = max_element(pred.begin(), pred.end());
            auto trueMax = max_element(Y[i].begin(), Y[i].end());
            if (distance(pred.begin(), predMax) == distance(Y[i].begin(), trueMax)) {
                correct++;
            }
        }
    }
    double accuracy = static_cast<double>(correct) / X.size();
    double mse = totalError / X.size();
    // Fitness combina precisión y error cuadrático medio
    fitness_ = accuracy * 100.0 - mse;
    return fitness_;
}

double Individual::calculateGameFitness(int maxSteps) {
    fitness_ = 0.0;
    return fitness_;
}


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

string Individual::getArchitectureString() const {
    stringstream ss;
    for (size_t i = 0; i < topology_.size(); ++i) {
        ss << topology_[i];
        if (i < topology_.size() - 1) ss << "-";
    }
    return ss.str();
}

void Individual::save(const string& filepath) const {
    ofstream file(filepath);
    if (!file) throw runtime_error("No se pudo crear: " + filepath);
    file << topology_.size() << "\n";
    for (int t : topology_) file << t << " ";
    file << "\n";
    file << static_cast<int>(activation_) << "\n";
    file << fitness_ << "\n";
    for (size_t layer = 0; layer < weights_.size(); ++layer) {
        for (const auto& row : weights_[layer]) {
            for (double w : row) file << w << " ";
            file << "\n";
        }
        for (double b : biases_[layer]) file << b << " ";
        file << "\n";
    }
}

void Individual::load(const string& filepath) {
    ifstream file(filepath);
    if (!file) throw runtime_error("No se pudo abrir: " + filepath);
    size_t numLayers;
    file >> numLayers;
    topology_.resize(numLayers);
    for (size_t i = 0; i < numLayers; ++i) {
        file >> topology_[i];
    }
    int act;
    file >> act;
    activation_ = static_cast<ActivationType>(act);
    file >> fitness_;
    weights_.clear();
    biases_.clear();
    for (size_t layer = 0; layer < numLayers - 1; ++layer) {
        int outputSize = topology_[layer + 1];
        int inputSize = topology_[layer];
        vector<vector<double>> layerWeights(outputSize);
        for (int j = 0; j < outputSize; ++j) {
            layerWeights[j].resize(inputSize);
            for (int k = 0; k < inputSize; ++k) {
                file >> layerWeights[j][k];
            }
        }
        weights_.push_back(layerWeights);
        vector<double> layerBiases(outputSize);
        for (int j = 0; j < outputSize; ++j) {
            file >> layerBiases[j];
        }
        biases_.push_back(layerBiases);
    }
}
