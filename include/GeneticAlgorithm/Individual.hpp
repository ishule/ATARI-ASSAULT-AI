/*
 * INDIVIDUAL.HPP - Representa un individuo (una red neuronal) en el AG
 * 
 * Básicamente cada individuo es una red neuronal con sus pesos.
 * El AG va evolucionando estos individuos para encontrar la mejor red.
 * 
 * Dos modos:
 * - Solo pesos (nota 1.75): la arquitectura es fija, solo cambian los pesos
 * - Neuroevolución (nota 2.00): cambia todo, arquitectura y pesos
 */

#ifndef INDIVIDUAL_HPP
#define INDIVIDUAL_HPP

#include "MLP.hpp"
#include <vector>
#include <memory>
#include <random>

// Vector 3D para guardar los pesos: [capa][neurona][peso]
using VecWeights = std::vector<std::vector<std::vector<double>>>;


class Individual {
private:
    double fitness_;                          // Qué tan bueno es (mayor = mejor)
    std::vector<int> topology_;               // Arquitectura: ej {4, 10, 3}
    VecWeights weights_;                      // Los pesos de la red
    std::vector<std::vector<double>> biases_; // Los biases
    ActivationType activation_;               // Función de activación (RELU, etc)
    
    static std::mt19937& getRandomEngine();   // Generador de randoms
    
public:
    // === CONSTRUCTORES ===
    
    // Crea individuo con arquitectura fija y pesos random
    explicit Individual(const std::vector<int>& topology, 
                       ActivationType act = ActivationType::RELU);
    
    // Crea individuo con pesos específicos (para cuando hacemos crossover)
    Individual(const std::vector<int>& topology,
               const VecWeights& weights,
               const std::vector<std::vector<double>>& biases,
               ActivationType act = ActivationType::RELU);
    
    // Crea individuo con arquitectura random (para neuroevolución)
    Individual(int inputSize, int outputSize,
               int minHiddenLayers, int maxHiddenLayers,
               int minNeurons, int maxNeurons,
               ActivationType act = ActivationType::RELU);
    
    // Constructor copia
    Individual(const Individual& other);
    Individual& operator=(const Individual& other);
    
    // === OPERADORES GENÉTICOS ===
    
    // Crossover normal: mezcla pesos de dos padres (misma arquitectura)
    Individual crossover(const Individual& other, double mutationRate) const;
    
    // Crossover para neuroevolución: puede cambiar la arquitectura
    Individual crossoverNeuroevolution(const Individual& other, 
                                       double mutationRate,
                                       double archMutationRate) const;
    
    // Muta los pesos añadiendo ruido gaussiano
    void mutateWeights(double mutationRate, double mutationStrength = 0.5);
    
    // Muta la arquitectura: añade/quita neuronas o capas
    void mutateArchitecture(double mutationRate, int minNeurons = 4, int maxNeurons = 128);
    
    // === PREDICCIÓN ===
    
    // Pasa un input por la red y devuelve el output
    std::vector<double> predict(const std::vector<double>& input) const;
    
    // === FITNESS ===
    
    // Calcula el fitness con un dataset (accuracy básicamente)
    double calculateFitness(const std::vector<std::vector<double>>& X,
                           const std::vector<std::vector<double>>& Y);
    
    // Calcula fitness jugando (para Atari y eso)
    double calculateGameFitness(int maxSteps = 20000);
    
    // === GETTERS ===
    
    double getFitness() const { return fitness_; }
    void setFitness(double f) { fitness_ = f; }
    const std::vector<int>& getTopology() const { return topology_; }
    const VecWeights& getWeights() const { return weights_; }
    const std::vector<std::vector<double>>& getBiases() const { return biases_; }
    ActivationType getActivation() const { return activation_; }
    int getTotalParameters() const;           // Cuenta pesos + biases
    std::string getArchitectureString() const; // Devuelve "4-10-3" por ejemplo
    
    // Para ordenar por fitness (mayor primero)
    bool operator<(const Individual& other) const {
        return fitness_ > other.fitness_;
    }
    
    bool operator>(const Individual& other) const {
        return fitness_ < other.fitness_;
    }
    
    // Guardar/cargar de archivo
    void save(const std::string& filepath) const;
    void load(const std::string& filepath);
    
private:
    // Inicializa pesos random con Xavier
    void initializeRandomWeights();
    
    // Forward pass de la red
    std::vector<double> forwardPass(const std::vector<double>& input) const;
    
    // Genera topología random
    static std::vector<int> generateRandomTopology(int inputSize, int outputSize,
                                                    int minHiddenLayers, int maxHiddenLayers,
                                                    int minNeurons, int maxNeurons);
};

#endif
