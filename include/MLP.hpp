#ifndef MLP_HPP
#define MLP_HPP

#include "Network.hpp"
#include "ActivationFunctions.hpp"
#include <vector>
#include <string>

//Configuración para el Multi-Layer Perceptron
struct MLPConfig {
    std::vector<int> layerSizes;              // Tamaño de cada capa [input, hidden1, hidden2, ..., output]
    ActivationType activation = ActivationType::SIGMOID;  // Función de activación
    double learningRate = 0.01;               // Tasa de aprendizaje
    int maxEpochs = 100;                      // Número máximo de épocas
    int batchSize = 32;                       // Tamaño del mini-batch
    
    // Regularización
    bool useDropout = false;                  // Usar dropout
    double dropoutRate = 0.0;                 // Tasa de dropout (0.0 - 1.0)
    bool useL2 = false;                       // Usar regularización L2
    double l2Lambda = 0.01;                   // Parámetro lambda para L2
    
    // Early Stopping
    bool useEarlyStopping = false;            // Usar early stopping
    int patience = 10;                        // Épocas sin mejora antes de parar
    double minDelta = 0.001;                  // Mejora mínima para considerar progreso
    
    // Otros
    bool verbose = true;                      // Mostrar progreso
    int printEvery = 10;                      // Mostrar cada N épocas
};

/**
 * Implementa una red neuronal con:
 * - Forward propagation
 * - Backpropagation
 * - Regularización (Dropout, L2)
 * - Early Stopping
 * - Diferentes funciones de activación
 */
class MLP : public Network {
private:
    // Arquitectura
    std::vector<MatDouble_t> weights;         // Pesos de cada capa
    std::vector<VecDouble_t> biases;          // Bias de cada capa
    MLPConfig config;                         // Configuración
    
    // Para backpropagation (se almacenan durante forward pass)
    mutable std::vector<VecDouble_t> layerOutputs;  // Salidas de cada capa (post-activación)
    mutable std::vector<VecDouble_t> layerInputs;   // Entradas de cada capa (pre-activación)
    
    // Para dropout
    mutable std::vector<std::vector<bool>> dropoutMasks;
    
    // Métricas de entrenamiento
    std::vector<double> trainLossHistory;
    std::vector<double> valLossHistory;
    int bestEpoch;
    double bestValLoss;

public:

    explicit MLP(const MLPConfig& cfg);
    
    MLP(const std::vector<int>& layers, ActivationType act = ActivationType::SIGMOID);
    
    ~MLP() override = default;
    
    // Implementación de la Red
    void load(const std::string& filepath) override;
    void save(const std::string& filepath) const override;
    void train(const MatDouble_t& X, const MatDouble_t& Y,
               const MatDouble_t& Xval, const MatDouble_t& Yval) override;
    VecDouble_t predict(const VecDouble_t& input) const override;

    double evaluate(const MatDouble_t& Xtest, const MatDouble_t& Ytest) const override;
    
    //Forward propagation (predicción con detalles)
    VecDouble_t forwardPass(const VecDouble_t& input, bool training = false) const;
    
    //Inicializa los pesos de la red
    void initializeWeights();
    
    //Aplica dropout a una capa
    void applyDropout(VecDouble_t& layer, size_t layerIdx, bool training) const;
    
    // Getters
    const MLPConfig& getConfig() const { return config; }
    const std::vector<double>& getTrainLossHistory() const { return trainLossHistory; }
    const std::vector<double>& getValLossHistory() const { return valLossHistory; }
    int getBestEpoch() const { return bestEpoch; }
    
    // Setters
    void setLearningRate(double lr) { config.learningRate = lr; }
    void setVerbose(bool v) { config.verbose = v; }

private:
    //Backpropagation - actualiza pesos y bias
    void backpropagate(const MatDouble_t& X, const MatDouble_t& Y);
    
    //Calcula el loss
    double calculateLoss(const MatDouble_t& X, const MatDouble_t& Y) const;
    
    //Calcula gradientes para una muestra
    void computeGradients(const VecDouble_t& x, const VecDouble_t& y,
                         std::vector<MatDouble_t>& weightGrads,
                         std::vector<VecDouble_t>& biasGrads);
    
    //Actualiza pesos con los gradientes acumulados
    void updateWeights(const std::vector<MatDouble_t>& weightGrads,
                      const std::vector<VecDouble_t>& biasGrads,
                      size_t batchSize);
    
    //Aplica regularización L2 a los gradientes
    void applyL2Regularization(std::vector<MatDouble_t>& weightGrads);
    
    //Verifica si debe hacer early stopping
    bool checkEarlyStopping(double valLoss, int epoch);
};

#endif  