#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include "Network.hpp"

class Perceptron : public Network {
private:
    MatDouble_t weights;  // 5 vectores de pesos para cada acción

public:
    // Constructor
    Perceptron(int numInputs, int numOutputs);

    // Implementación de métodos de Network
    void load(const std::string& filepath) override;
    void save(const std::string& filepath) const override;
    void train(const MatDouble_t& X, const MatDouble_t& Y,
               const MatDouble_t& Xval, const MatDouble_t& Yval) override;
    VecDouble_t predict(const VecDouble_t& input) const override;
};

#endif  // PERCEPTRON_HPP