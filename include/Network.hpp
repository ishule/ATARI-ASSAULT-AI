#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <vector>
#include <string>
#include <memory>

using VecDouble_t = std::vector<double>;
using MatDouble_t = std::vector<std::vector<double>>;

class Network {
public:
    virtual ~Network() = default;

    virtual void load(const std::string& filepath) = 0;
    virtual void save(const std::string& filepath) const = 0;

    virtual void setHyperparameters(double learningRate, int epochs) {
        (void)learningRate;
        (void)epochs;
    }

    virtual void train(const MatDouble_t& X,
                       const MatDouble_t& Y,
                       const MatDouble_t& Xval,
                       const MatDouble_t& Yval) = 0;

    // Hacer predicciones sobre datos de entrada
    virtual VecDouble_t predict(const VecDouble_t& input) const = 0;

    // Evaluar la red con datos de test
    virtual double evaluate(const MatDouble_t& Xtest, const MatDouble_t& Ytest) const {
        return 0.0;  // Por defecto devuelve 0, reemplazar en clases derivadas
    }

protected:
    // Validar dimensiones (útil en clases derivadas)
    void validateDimensions(const MatDouble_t& X, const MatDouble_t& Y) const {
        if (X.empty() || Y.empty()) {
            throw std::invalid_argument("Los datos de entrada (X) o las etiquetas (Y) están vacíos");
        }
        if (X.size() != Y.size()) {
            throw std::invalid_argument("El número de ejemplos en X y Y no coincide");
        }
    }
};

#endif  // NETWORK_HPP