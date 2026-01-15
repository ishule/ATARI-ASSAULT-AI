#include "Perceptron.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>
using namespace std;
Perceptron::Perceptron(int numInputs, int numOutputs)
    : weights(numOutputs, VecDouble_t(numInputs + 1, 0.0)) {}  // Inicializar pesos a 0

void Perceptron::load(const string& filepath) {
    ifstream file(filepath);
    if (!file) throw runtime_error("No se pudo abrir el archivo: " + filepath);

    for (auto& row : weights) {
        for (auto& w : row) {
            file >> w;
        }
    }
}

void Perceptron::save(const string& filepath) const {
    ofstream file(filepath);
    if (!file) throw runtime_error("No se pudo crear el archivo: " + filepath);

    for (const auto& row : weights) {
        for (const auto& w : row) {
            file << w << " ";
        }
        file << "\n";
    }
}

void Perceptron::train(const MatDouble_t& X, const MatDouble_t& Y,
                       const MatDouble_t& Xval, const MatDouble_t& Yval) {
    validateDimensions(X, Y); 

    // Lambda para calcular accuracy (usa el predict corregido internamente)
    auto accuracy = [this](const MatDouble_t& Xin, const MatDouble_t& Yin) {
        if (Xin.empty()) return 0.0;
        int ok = 0;
        for (size_t i = 0; i < Xin.size(); ++i) {
            auto scores = predict(Xin[i]);
            // Lógica para clasificación binaria o multiclase
            if (scores.size() == 1) {
                double pred = scores[0] >= 0 ? 1.0 : -1.0; 
                // Asumimos etiquetas 1.0 y -1.0 (o 0.0 si ajustas la lógica)
                if (abs(pred - Yin[i][0]) < 0.1) ok++;
            } else {
                size_t pred = static_cast<size_t>(distance(
                    scores.begin(), max_element(scores.begin(), scores.end())));
                size_t truth = static_cast<size_t>(distance(
                    Yin[i].begin(), max_element(Yin[i].begin(), Yin[i].end())));
                if (pred == truth) ok++;
            }
        }
        return (static_cast<double>(ok) / Xin.size()) * 100.0;
    };

    MatDouble_t bestWeights = weights;
    double bestVal = -1.0;
    bool hasVal = !Xval.empty();
    double learningRate = 0.1; 

    for (int epoch = 0; epoch < 100; ++epoch) {
        for (size_t i = 0; i < X.size(); ++i) {
            for (size_t j = 0; j < weights.size(); ++j) {
                // Calculamos la salida actual incluyendo el bias
                double dot = inner_product(X[i].begin(), X[i].end(), weights[j].begin(), 0.0);
                double currentOutput = dot + weights[j].back(); // Sumar Bias
                
                // Función de activación escalón (Step)
                double pred = (currentOutput >= 0) ? 1.0 : -1.0; 
                
                // Calculamos error (Target - Predicción)
                double error = Y[i][j] - pred;

                if (error != 0.0) {
                    // Actualizar pesos de las entradas
                    for (size_t k = 0; k < X[i].size(); ++k) {
                        weights[j][k] += learningRate * error * X[i][k];
                    }
                    
                    // Actualizar Bias (wn+1)
                    // El bias siempre tiene una entrada imaginaria de 1.0
                    weights[j].back() += learningRate * error * 1.0;
                }
            }
        }

        // Evaluar progreso
        double accTrain = accuracy(X, Y);
        double accVal = hasVal ? accuracy(Xval, Yval) : accTrain;

        if (hasVal && accVal > bestVal) {
            bestVal = accVal;
            bestWeights = weights;
        }
        
        
        cout << "Epoch " << (epoch + 1) << " - train: " << accTrain<< "% val: " << accVal << "%\n";
        
    }

    if (hasVal && bestVal >= 0.0) {
        weights = bestWeights;
    }
}
VecDouble_t Perceptron::predict(const VecDouble_t& input) const {
    VecDouble_t outputs(weights.size());
    for (size_t j = 0; j < weights.size(); ++j) {
        double dot = inner_product(input.begin(), input.end(), weights[j].begin(), 0.0);
        double val = dot + weights[j].back();
        
        outputs[j] = (val >= 0.0) ? 1.0 : -1.0; 
    }
    return outputs;
}