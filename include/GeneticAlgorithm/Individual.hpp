//Representa individo (red neuronal)
#ifndef INDIVIDUAL_HPP
#define INDIVIDUAL_HPP

#include "MLP.hpp"
#include <vector>
#include <memory>
#include <random>
using namespace std;
// Vector 3D para guardar los pesos: [capa][neurona][peso]
using VecWeights = vector<vector<vector<double>>>;


class Individual {
private:
    double fitness_;                          // Qué tan bueno es (mayor = mejor)
    vector<int> topology_;               // Arquitectura de la red
    VecWeights weights_;                      // Los pesos de la red
    vector<vector<double>> biases_; // Los biases
    ActivationType activation_;               // Función de activación 
    
    static mt19937& getRandomEngine();   // Generador de randoms
    
public:
    
    // Crea individuo con arquitectura fija y pesos random, explicit dicho por chatgpt para evitar bugs
    explicit Individual(const vector<int>& topology, ActivationType act = ActivationType::RELU);
    
    // Crea individuo con pesos específicos (para cuando hacemos crossover)
    Individual(const vector<int>& topology,const VecWeights& weights,const vector<vector<double>>& biases,ActivationType act = ActivationType::RELU);
    
    // Crea individuo con arquitectura random (para neuroevolución)
    Individual(int inputSize, int outputSize,int minHiddenLayers, int maxHiddenLayers,int minNeurons, int maxNeurons,ActivationType act = ActivationType::RELU);
    
    // Constructor copia
    Individual(const Individual& other);
    Individual& operator=(const Individual& other);
    
    
    // Mezcla pesos de dos padres
    Individual crossover(const Individual& other, double mutationRate) const;
    
    // Crossover pero para el segundo caso
    Individual crossoverNeuroevolution(const Individual& other, double mutationRate,double archMutationRate) const;
    
    // Muta los pesos añadiendo ruido gaussiano
    void mutateWeights(double mutationRate, double mutationStrength = 0.5);
    
    // Muta la arquitectura: añade/quita neuronas o capas
    void mutateArchitecture(double mutationRate, int minNeurons = 4, int maxNeurons = 128);
    
    // Pasa un input por la red y devuelve el output
    vector<double> predict(const vector<double>& input) const;
    
    
    // Calcula el fitness con un dataset (accuracy básicamente)
    double calculateFitness(const vector<vector<double>>& X,const vector<vector<double>>& Y);
    
    // Calcula fitness jugando (para Atari y eso)
    double calculateGameFitness(int maxSteps = 20000);
    
    double getFitness() const { return fitness_; }
    void setFitness(double f) { fitness_ = f; }
    const vector<int>& getTopology() const { return topology_; }
    const VecWeights& getWeights() const { return weights_; }
    const vector<vector<double>>& getBiases() const { return biases_; }
    ActivationType getActivation() const { return activation_; }
    int getTotalParameters() const;           // Cuenta pesos + biases
    string getArchitectureString() const; 
    
    // Para ordenar por fitness (mayor primero)
    bool operator<(const Individual& other) const {
        return fitness_ > other.fitness_;
    }
    
    bool operator>(const Individual& other) const {
        return fitness_ < other.fitness_;
    }
    
    // Guardar/cargar de archivo
    void save(const string& filepath) const;
    void load(const string& filepath);
    
private:
    // Inicializa pesos random con Xavier
    void initializeRandomWeights();
    
    // Forward pass de la red
    vector<double> forwardPass(const vector<double>& input) const;
    
    // Genera topología random
    static vector<int> generateRandomTopology(int inputSize, int outputSize,int minHiddenLayers, int maxHiddenLayers,int minNeurons, int maxNeurons);
};

#endif
