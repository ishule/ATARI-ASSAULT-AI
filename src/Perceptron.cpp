#include "Perceptron.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>

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

    auto accuracy = [this](const MatDouble_t& Xin, const MatDouble_t& Yin) {
        if (Xin.empty()) return 0.0;
        int ok = 0;
        for (size_t i = 0; i < Xin.size(); ++i) {
            auto scores = predict(Xin[i]);
            if (scores.size() == 1) {
                double pred = scores[0] >= 0 ? 1.0 : -1.0;
                if (pred == Yin[i][0]) ok++;
            } else {
                size_t pred = static_cast<size_t>(std::distance(
                    scores.begin(), std::max_element(scores.begin(), scores.end())));
                size_t truth = static_cast<size_t>(std::distance(
                    Yin[i].begin(), std::max_element(Yin[i].begin(), Yin[i].end())));
                if (pred == truth) ok++;
            }
        }
        return (static_cast<double>(ok) / Xin.size()) * 100.0;
    };

    MatDouble_t bestWeights = weights;
    double bestVal = -1.0;
    bool hasVal = !Xval.empty();

    for (int epoch = 0; epoch < 100; ++epoch) {  // Ejemplo: 100 épocas
        for (size_t i = 0; i < X.size(); ++i) {
            for (size_t j = 0; j < weights.size(); ++j) {
                double pred = std::inner_product(X[i].begin(), X[i].end(), weights[j].begin(), 0.0);
                double error = Y[i][j] - (pred > 0 ? 1 : -1);
                // Actualizar pesos (solo hasta el tamaño de X[i])
                for (size_t k = 0; k < X[i].size(); ++k) {
                    weights[j][k] += X[i][k] * error * 0.1;  // 0.1 = learning rate
                }
            }
        }

        // Evaluar por época
        double accTrain = accuracy(X, Y);
        double accVal = hasVal ? accuracy(Xval, Yval) : accTrain;
        if (hasVal && accVal > bestVal) {
            bestVal = accVal;
            bestWeights = weights;
        }
        std::cout << "Epoch " << (epoch + 1) << " - train: " << accTrain
                  << "% val: " << accVal << "%\n";
    }

    // Dejar los mejores pesos (según val) si hay validación
    if (hasVal && bestVal >= 0.0) {
        weights = bestWeights;
    }
}

VecDouble_t Perceptron::predict(const VecDouble_t& input) const {
    VecDouble_t outputs(weights.size());
    for (size_t j = 0; j < weights.size(); ++j) {
        outputs[j] = std::inner_product(input.begin(), input.end(), weights[j].begin(), 0.0);
    }
    return outputs;
}