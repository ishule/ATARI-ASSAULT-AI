#include "Perceptron.hpp"
#include <fstream>
#include <iostream>
#include <random>

Perceptron::Perceptron(int numInputs, int numOutputs)
    : weights(numOutputs, VecDouble_t(numInputs + 1, 0.0)) {}  // Inicializar pesos a 0

void Perceptron::load(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file) throw std::runtime_error("No se pudo abrir el archivo: " + filepath);

    for (auto& row : weights) {
        for (auto& w : row) {
            file >> w;
        }
    }
}

void Perceptron::save(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file) throw std::runtime_error("No se pudo crear el archivo: " + filepath);

    for (const auto& row : weights) {
        for (const auto& w : row) {
            file << w << " ";
        }
        file << "\n";
    }
}

void Perceptron::train(const MatDouble_t& X, const MatDouble_t& Y,
                       const MatDouble_t& Xval, const MatDouble_t& Yval) {
    validateDimensions(X, Y);  // Valida datos

    for (int epoch = 0; epoch < 100; ++epoch) {  // Ejemplo: 100 épocas
        for (size_t i = 0; i < X.size(); ++i) {
            for (size_t j = 0; j < weights.size(); ++j) {
                double pred = std::inner_product(X[i].begin(), X[i].end(), weights[j].begin(), 0.0);
                double error = Y[i][j] - (pred > 0 ? 1 : -1);
                // Actualizar pesos
                for (size_t k = 0; k < weights[j].size(); ++k) {
                    weights[j][k] += X[i][k] * error * 0.1;  // 0.1 = learning rate
                }
            }
        }
    }
}

VecDouble_t Perceptron::predict(const VecDouble_t& input) const {
    VecDouble_t outputs(weights.size());
    for (size_t j = 0; j < weights.size(); ++j) {
        outputs[j] = std::inner_product(input.begin(), input.end(), weights[j].begin(), 0.0);
    }
    return outputs;
}