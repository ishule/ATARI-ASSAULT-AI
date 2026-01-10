#include "MLP.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <numeric>

// ============================================================================
// CONSTRUCCIÓN E INICIALIZACIÓN
// ============================================================================

MLP::MLP(const MLPConfig& cfg) : config(cfg), bestEpoch(0), bestValLoss(1e9) {
    if (config.layerSizes.size() < 2) {
        throw std::invalid_argument("Se necesitan al menos 2 capas (entrada y salida)");
    }
    initializeWeights();
}

MLP::MLP(const std::vector<int>& layers, ActivationType act) : bestEpoch(0), bestValLoss(1e9) {
    config.layerSizes = layers;
    config.activation = act;
    if (layers.size() < 2) {
        throw std::invalid_argument("Se necesitan al menos 2 capas (entrada y salida)");
    }
    initializeWeights();
}

void MLP::initializeWeights() {
    weights.clear();
    biases.clear();
    dropoutMasks.clear();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (size_t i = 0; i < config.layerSizes.size() - 1; ++i) {
        int inputSize = config.layerSizes[i];
        int outputSize = config.layerSizes[i + 1];
        
        // Xavier/Glorot initialization
        double limit = std::sqrt(6.0 / (inputSize + outputSize));
        std::uniform_real_distribution<> dis(-limit, limit);
        
        // Inicializar pesos
        MatDouble_t layerWeights(outputSize, VecDouble_t(inputSize));
        for (auto& row : layerWeights) {
            for (auto& w : row) {
                w = dis(gen);
            }
        }
        weights.push_back(layerWeights);
        
        // Inicializar bias a 0
        biases.push_back(VecDouble_t(outputSize, 0.0));
        
        // Inicializar máscaras de dropout
        dropoutMasks.push_back(std::vector<bool>(outputSize, true));
    }
}

// ============================================================================
// FORWARD PROPAGATION
// ============================================================================

VecDouble_t MLP::forwardPass(const VecDouble_t& input, bool training) const {
    layerOutputs.clear();
    layerInputs.clear();
    
    VecDouble_t current = input;
    layerOutputs.push_back(current);
    
    for (size_t layer = 0; layer < weights.size(); ++layer) {
        VecDouble_t z(weights[layer].size());  // Pre-activación
        
        // Calcular z = W * x + b
        for (size_t j = 0; j < weights[layer].size(); ++j) {
            double sum = biases[layer][j];
            for (size_t k = 0; k < current.size(); ++k) {
                sum += weights[layer][j][k] * current[k];
            }
            z[j] = sum;
        }
        
        layerInputs.push_back(z);
        
        // Aplicar función de activación
        VecDouble_t activated(z.size());
        bool isOutputLayer = (layer == weights.size() - 1);
        
        if (isOutputLayer) {
            // Capa de salida: usar sigmoid para binario o softmax para multiclase
            if (z.size() == 1) {
                activated[0] = ActivationFunctions::apply(z[0], ActivationType::SIGMOID);
            } else {
                activated = ActivationFunctions::softmax(z);
            }
        } else {
            // Capas ocultas: usar la activación configurada
            for (size_t j = 0; j < z.size(); ++j) {
                activated[j] = ActivationFunctions::apply(z[j], config.activation);
            }
            
            // Aplicar dropout en entrenamiento
            if (training && config.useDropout && layer < weights.size() - 1) {
                applyDropout(activated, layer, training);
            }
        }
        
        layerOutputs.push_back(activated);
        current = activated;
    }
    
    return current;
}

void MLP::applyDropout(VecDouble_t& layer, size_t layerIdx, bool training) const {
    if (!training) {
        // En inferencia, escalar por (1 - dropoutRate)
        for (auto& val : layer) {
            val *= (1.0 - config.dropoutRate);
        }
        return;
    }
    
    // En entrenamiento, aplicar máscara aleatoria
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (size_t i = 0; i < layer.size(); ++i) {
        bool keep = dis(gen) > config.dropoutRate;
        dropoutMasks[layerIdx][i] = keep;
        if (!keep) {
            layer[i] = 0.0;
        } else {
            layer[i] /= (1.0 - config.dropoutRate);  // Inverted dropout
        }
    }
}

VecDouble_t MLP::predict(const VecDouble_t& input) const {
    return forwardPass(input, false);  // No training mode
}

// ============================================================================
// BACKPROPAGATION
// ============================================================================

void MLP::computeGradients(const VecDouble_t& x, const VecDouble_t& y,
                           std::vector<MatDouble_t>& weightGrads,
                           std::vector<VecDouble_t>& biasGrads) {
    // Forward pass para obtener activaciones
    forwardPass(x, true);  // En modo training
    
    // Calcular error de salida (delta de la última capa)
    size_t outputLayer = weights.size() - 1;
    VecDouble_t delta = layerOutputs.back();
    
    // Para MSE: delta = (y_pred - y_true)
    for (size_t i = 0; i < delta.size(); ++i) {
        delta[i] -= y[i];
    }
    
    // Backpropagation hacia atrás
    for (int layer = static_cast<int>(weights.size()) - 1; layer >= 0; --layer) {
        // Calcular gradientes para esta capa
        const VecDouble_t& prevOutput = layerOutputs[layer];
        
        // Gradiente de pesos: dW = delta * prevOutput^T
        for (size_t j = 0; j < weights[layer].size(); ++j) {
            for (size_t k = 0; k < weights[layer][j].size(); ++k) {
                weightGrads[layer][j][k] += delta[j] * prevOutput[k];
            }
            // Gradiente de bias: db = delta
            biasGrads[layer][j] += delta[j];
        }
        
        // Si no es la primera capa, propagar el error hacia atrás
        if (layer > 0) {
            VecDouble_t nextDelta(prevOutput.size(), 0.0);
            
            // delta_prev = W^T * delta
            for (size_t k = 0; k < weights[layer][0].size(); ++k) {
                double sum = 0.0;
                for (size_t j = 0; j < weights[layer].size(); ++j) {
                    sum += weights[layer][j][k] * delta[j];
                }
                nextDelta[k] = sum;
            }
            
            // Multiplicar por derivada de la activación
            const VecDouble_t& z = layerInputs[layer - 1];
            for (size_t k = 0; k < nextDelta.size(); ++k) {
                nextDelta[k] *= ActivationFunctions::derivative(z[k], config.activation);
                
                // Considerar dropout
                if (config.useDropout && dropoutMasks[layer - 1][k] == false) {
                    nextDelta[k] = 0.0;
                }
            }
            
            delta = nextDelta;
        }
    }
}

void MLP::applyL2Regularization(std::vector<MatDouble_t>& weightGrads) {
    if (!config.useL2) return;
    
    for (size_t layer = 0; layer < weights.size(); ++layer) {
        for (size_t j = 0; j < weights[layer].size(); ++j) {
            for (size_t k = 0; k < weights[layer][j].size(); ++k) {
                // Añadir término de regularización: lambda * w
                weightGrads[layer][j][k] += config.l2Lambda * weights[layer][j][k];
            }
        }
    }
}

void MLP::updateWeights(const std::vector<MatDouble_t>& weightGrads,
                       const std::vector<VecDouble_t>& biasGrads,
                       size_t batchSize) {
    double lr = config.learningRate / batchSize;
    
    for (size_t layer = 0; layer < weights.size(); ++layer) {
        // Actualizar pesos
        for (size_t j = 0; j < weights[layer].size(); ++j) {
            for (size_t k = 0; k < weights[layer][j].size(); ++k) {
                weights[layer][j][k] -= lr * weightGrads[layer][j][k];
            }
            // Actualizar bias
            biases[layer][j] -= lr * biasGrads[layer][j];
        }
    }
}

void MLP::backpropagate(const MatDouble_t& X, const MatDouble_t& Y) {
    // Inicializar gradientes acumulados
    std::vector<MatDouble_t> weightGrads;
    std::vector<VecDouble_t> biasGrads;
    
    for (size_t layer = 0; layer < weights.size(); ++layer) {
        MatDouble_t wg(weights[layer].size(), 
                       VecDouble_t(weights[layer][0].size(), 0.0));
        weightGrads.push_back(wg);
        biasGrads.push_back(VecDouble_t(biases[layer].size(), 0.0));
    }
    
    // Procesar mini-batches
    size_t numSamples = X.size();
    size_t batchSize = std::min(static_cast<size_t>(config.batchSize), numSamples);
    
    for (size_t i = 0; i < numSamples; i += batchSize) {
        size_t currentBatchSize = std::min(batchSize, numSamples - i);
        
        // Resetear gradientes del batch
        for (auto& wg : weightGrads) {
            for (auto& row : wg) {
                std::fill(row.begin(), row.end(), 0.0);
            }
        }
        for (auto& bg : biasGrads) {
            std::fill(bg.begin(), bg.end(), 0.0);
        }
        
        // Acumular gradientes del batch
        for (size_t j = 0; j < currentBatchSize; ++j) {
            computeGradients(X[i + j], Y[i + j], weightGrads, biasGrads);
        }
        
        // Aplicar regularización
        if (config.useL2) {
            applyL2Regularization(weightGrads);
        }
        
        // Actualizar pesos
        updateWeights(weightGrads, biasGrads, currentBatchSize);
    }
}

// ============================================================================
// ENTRENAMIENTO
// ============================================================================

double MLP::calculateLoss(const MatDouble_t& X, const MatDouble_t& Y) const {
    double totalLoss = 0.0;
    
    for (size_t i = 0; i < X.size(); ++i) {
        auto pred = predict(X[i]);
        
        // MSE
        for (size_t j = 0; j < pred.size(); ++j) {
            double error = pred[j] - Y[i][j];
            totalLoss += error * error;
        }
    }
    
    return totalLoss / X.size();
}

bool MLP::checkEarlyStopping(double valLoss, int epoch) {
    if (!config.useEarlyStopping) return false;
    
    if (valLoss < bestValLoss - config.minDelta) {
        bestValLoss = valLoss;
        bestEpoch = epoch;
        return false;
    }
    
    return (epoch - bestEpoch) >= config.patience;
}

void MLP::train(const MatDouble_t& X, const MatDouble_t& Y,
                const MatDouble_t& Xval, const MatDouble_t& Yval) {
    validateDimensions(X, Y);
    
    trainLossHistory.clear();
    valLossHistory.clear();
    bestValLoss = 1e9;
    bestEpoch = 0;
    
    if (config.verbose) {
        std::cout << "\n=== Entrenamiento MLP ===\n";
        std::cout << "Arquitectura: ";
        for (int size : config.layerSizes) {
            std::cout << size << " ";
        }
        std::cout << "\nActivación: " << ActivationFunctions::toString(config.activation) << "\n";
        std::cout << "Learning Rate: " << config.learningRate << "\n";
        std::cout << "Batch Size: " << config.batchSize << "\n";
        std::cout << "Dropout: " << (config.useDropout ? "Yes (" + std::to_string(config.dropoutRate) + ")" : "No") << "\n";
        std::cout << "L2 Regularization: " << (config.useL2 ? "Yes (lambda=" + std::to_string(config.l2Lambda) + ")" : "No") << "\n";
        std::cout << "Early Stopping: " << (config.useEarlyStopping ? "Yes (patience=" + std::to_string(config.patience) + ")" : "No") << "\n\n";
    }
    
    for (int epoch = 0; epoch < config.maxEpochs; ++epoch) {
        // Entrenar una época
        backpropagate(X, Y);
        
        // Calcular losses
        double trainLoss = calculateLoss(X, Y);
        double valLoss = calculateLoss(Xval, Yval);
        
        trainLossHistory.push_back(trainLoss);
        valLossHistory.push_back(valLoss);
        
        // Mostrar progreso
        if (config.verbose && (epoch % config.printEvery == 0 || epoch == config.maxEpochs - 1)) {
            double trainAcc = evaluate(X, Y);
            double valAcc = evaluate(Xval, Yval);
            
            std::cout << "Epoch " << (epoch + 1) << "/" << config.maxEpochs
                     << " - Loss: " << trainLoss 
                     << " - Val Loss: " << valLoss
                     << " - Acc: " << trainAcc << "%"
                     << " - Val Acc: " << valAcc << "%\n";
        }
        
        // Early stopping
        if (checkEarlyStopping(valLoss, epoch)) {
            if (config.verbose) {
                std::cout << "\nEarly stopping en epoch " << (epoch + 1) 
                         << " (mejor epoch: " << (bestEpoch + 1) << ")\n";
            }
            break;
        }
    }
    
    if (config.verbose) {
        std::cout << "\nEntrenamiento completado!\n";
    }
}

// ============================================================================
// GUARDADO Y CARGA
// ============================================================================

void MLP::save(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file) throw std::runtime_error("No se pudo crear: " + filepath);
    
    // Guardar configuración
    file << config.layerSizes.size() << "\n";
    for (int size : config.layerSizes) {
        file << size << " ";
    }
    file << "\n";
    
    file << static_cast<int>(config.activation) << "\n";
    file << config.learningRate << "\n";
    
    // Guardar pesos y bias
    for (size_t i = 0; i < weights.size(); ++i) {
        for (const auto& row : weights[i]) {
            for (double w : row) {
                file << w << " ";
            }
            file << "\n";
        }
        for (double b : biases[i]) {
            file << b << " ";
        }
        file << "\n";
    }
    
    file.close();
}

void MLP::load(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file) throw std::runtime_error("No se pudo abrir: " + filepath);
    
    // Leer configuración
    size_t numLayers;
    file >> numLayers;
    
    config.layerSizes.clear();
    for (size_t i = 0; i < numLayers; ++i) {
        int size;
        file >> size;
        config.layerSizes.push_back(size);
    }
    
    int actType;
    file >> actType;
    config.activation = static_cast<ActivationType>(actType);
    file >> config.learningRate;
    
    // Leer pesos y bias
    weights.clear();
    biases.clear();
    
    for (size_t i = 0; i < numLayers - 1; ++i) {
        MatDouble_t layerWeights(config.layerSizes[i + 1], 
                                 VecDouble_t(config.layerSizes[i]));
        for (auto& row : layerWeights) {
            for (auto& w : row) {
                file >> w;
            }
        }
        weights.push_back(layerWeights);
        
        VecDouble_t layerBias(config.layerSizes[i + 1]);
        for (auto& b : layerBias) {
            file >> b;
        }
        biases.push_back(layerBias);
    }
    
    file.close();
}


double MLP::evaluate(const MatDouble_t& Xtest, const MatDouble_t& Ytest) const {
    if (Xtest.empty() || Ytest.empty()) {
        return 0.0;
    }
    
    if (Xtest.size() != Ytest.size()) {
        throw std::invalid_argument("Xtest y Ytest deben tener el mismo tamaño");
    }
    
    int correct = 0;
    
    for (size_t i = 0; i < Xtest.size(); ++i) {
        // Obtener predicción (sin dropout, modo inferencia)
        VecDouble_t pred = predict(Xtest[i]);
        const VecDouble_t& target = Ytest[i];
        
        if (pred.size() != target.size()) {
            throw std::runtime_error("Dimensión de predicción no coincide con target");
        }
        
        if (target.size() == 1) {
            // ===============================================
            // CLASIFICACIÓN BINARIA
            // ===============================================
            // Si la salida es un único valor (0 o 1)
            double predicted = (pred[0] >= 0.5) ? 1.0 : 0.0;
            double actual = target[0];
            
            if (std::abs(predicted - actual) < 0.1) {
                correct++;
            }
            
        } else {
            // ===============================================
            // CLASIFICACIÓN MULTICLASE
            // ===============================================
            // Encontrar la clase con mayor probabilidad en la predicción
            size_t predClass = 0;
            double maxPred = pred[0];
            for (size_t j = 1; j < pred.size(); ++j) {
                if (pred[j] > maxPred) {
                    maxPred = pred[j];
                    predClass = j;
                }
            }
            
            // Encontrar la clase real (one-hot encoding)
            size_t trueClass = 0;
            double maxTrue = target[0];
            for (size_t j = 1; j < target.size(); ++j) {
                if (target[j] > maxTrue) {
                    maxTrue = target[j];
                    trueClass = j;
                }
            }
            
            // Comparar las clases
            if (predClass == trueClass) {
                correct++;
            }
        }
    }
    
    // Retornar el porcentaje de aciertos
    return (static_cast<double>(correct) / Xtest.size()) * 100.0;
}

