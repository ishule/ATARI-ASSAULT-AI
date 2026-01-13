#include "ActivationFunctions.hpp"
#include <algorithm>
#include <stdexcept>
#include <numeric>

double ActivationFunctions::apply(double x, ActivationType type) {
    switch (type) {
        case ActivationType::SIGMOID:
            // Evitar overflow
            if (x > 20) return 1.0;
            if (x < -20) return 0.0;
            return 1.0 / (1.0 + std::exp(-x));
        
        case ActivationType::TANH:
            return std::tanh(x);
        
        case ActivationType::RELU:
            return x > 0 ? x : 0.0;
     
        case ActivationType::SOFTMAX:
            // Softmax se aplica a vectores, no a escalares
            throw std::runtime_error("Softmax debe aplicarse a vectores, no escalares");
        
        default:
            throw std::runtime_error("Tipo de activación desconocido");
    }
}

double ActivationFunctions::derivative(double x, ActivationType type) {
    switch (type) {
        case ActivationType::SIGMOID: {
            double sig = apply(x, ActivationType::SIGMOID);
            return sig * (1.0 - sig);
        }
        
        case ActivationType::TANH: {
            double t = std::tanh(x);
            return 1.0 - t * t;
        }
        
        case ActivationType::RELU:
            return x > 0 ? 1.0 : 0.0;
        
        default:
            throw std::runtime_error("Tipo de activación desconocido");
    }
}

std::vector<double> ActivationFunctions::applyVector(const std::vector<double>& x, ActivationType type) {
    if (type == ActivationType::SOFTMAX) {
        return softmax(x);
    }
    
    std::vector<double> result(x.size());
    std::transform(x.begin(), x.end(), result.begin(),
                   [type](double val) { return apply(val, type); });
    return result;
}

std::vector<double> ActivationFunctions::softmax(const std::vector<double>& x) {
    if (x.empty()) return {};
    
    // Encontrar el máximo para estabilidad numérica
    double maxVal = *std::max_element(x.begin(), x.end());
    
    // Calcular exponenciales
    std::vector<double> expVals(x.size());
    double sumExp = 0.0;
    
    for (size_t i = 0; i < x.size(); ++i) {
        expVals[i] = std::exp(x[i] - maxVal);
        sumExp += expVals[i];
    }
    
    // Normalizar
    for (auto& val : expVals) {
        val /= sumExp;
    }
    
    return expVals;
}

std::string ActivationFunctions::toString(ActivationType type) {
    switch (type) {
        case ActivationType::SIGMOID: return "SIGMOID";
        case ActivationType::TANH: return "TANH";
        case ActivationType::RELU: return "RELU";
        case ActivationType::SOFTMAX: return "SOFTMAX";
        default: return "UNKNOWN";
    }
}

ActivationType ActivationFunctions::fromString(const std::string& str) {
    if (str == "SIGMOID") return ActivationType::SIGMOID;
    if (str == "TANH") return ActivationType::TANH;
    if (str == "RELU") return ActivationType::RELU;
    if (str == "SOFTMAX") return ActivationType::SOFTMAX;
    throw std::runtime_error("Tipo de activación desconocido: " + str);
}