#ifndef ACTIVATION_FUNCTIONS_HPP
#define ACTIVATION_FUNCTIONS_HPP

#include <string>
#include <vector>
#include <cmath>

/**
 * Tipos de funciones de activación disponibles
 */
enum class ActivationType {
    SIGMOID,
    TANH,
    RELU,
    LEAKY_RELU,
    LINEAR,
    SOFTMAX
};

/**
 * Esta clase proporciona todas las funciones de activación necesarias
 * para entrenar redes neuronales, junto con sus derivadas para backpropagation.
 */
class ActivationFunctions {
public:
    //Aplica la función de activación a un valor
    static double apply(double x, ActivationType type);
    
    //Calcula la derivada de la función de activación
    static double derivative(double x, ActivationType type);
    
    //Aplica la función de activación a un vector completo
    static std::vector<double> applyVector(const std::vector<double>& x, ActivationType type);
    
    //Aplica softmax a un vector (para multiclase)
    static std::vector<double> softmax(const std::vector<double>& x);
    
    //Convierte tipo de activación a string
    static std::string toString(ActivationType type);
    
    //Convierte string a tipo de activación
    static ActivationType fromString(const std::string& str);
};

#endif  