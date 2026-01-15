/*
 * - Crea la población inicial
 * - Evalúa el fitness de cada individuo
 * - Selecciona los mejores para reproducirse
 * - Hace crossover y mutación
 * - Repite hasta encontrar una buena solución
 *  Con evolucin de pesos o neuroevolucion
 */

#ifndef GENETIC_ALGORITHM_HPP
#define GENETIC_ALGORITHM_HPP

#include "GeneticAlgorithm/Individual.hpp"
#include <vector>
#include <string>
#include <functional>

using namespace std;

// Tipos de selección disponibles
enum class SelectionType {
    TOURNAMENT,     // Torneo: elige k random y se queda el mejor
    ROULETTE,       // Ruleta: prob proporcional al fitness
    RANK,           // Ranking: prob proporcional a la posición
    ELITISM         // Solo los mejores pasan
};

// Configuración del AG (todos los parámetros)
struct GAConfig {
    // Población
    int populationSize = 50;          // Cuántos individuos
    double eliteRatio = 0.1;          // % que pasa directo (élite)
    
    // Selección
    SelectionType selectionType = SelectionType::TOURNAMENT;
    int tournamentSize = 3;           // Tamaño del torneo
    
    // Mutación de pesos
    double mutationRate = 0.1;        // Prob de mutar cada peso
    double mutationStrength = 0.5;    // Cuánto muta (desv std)
    
    // Neuroevolución (si está activada)
    bool evolveArchitecture = false;  // true = neuroevolución
    double archMutationRate = 0.05;   // Prob de mutar arquitectura
    int minHiddenLayers = 1;
    int maxHiddenLayers = 5;
    int minNeuronsPerLayer = 4;
    int maxNeuronsPerLayer = 128;
    
    // Evolución
    int maxGenerations = 100;         // Máximo de generaciones
    double targetFitness = 1e9;       // Fitness objetivo (para parar antes)
    
    // Output
    bool verbose = true;              // Mostrar progreso
    int printEvery = 1;               // Cada cuántas generaciones imprimir
    string saveDir = "models/ga/";
};

class GeneticAlgorithm {
private:
    GAConfig config;
    vector<Individual> population;       // La población
    
    // Estadísticas
    vector<double> bestFitnessHistory;
    vector<double> avgFitnessHistory;
    int currentGeneration;
    
    // Función de fitness custom (opcional)
    function<double(Individual&)> fitnessFunction;
    
public:
    // Solo pesos
    GeneticAlgorithm(const GAConfig& cfg,const vector<int>& topology,ActivationType act = ActivationType::RELU);
    
    // Para neuroevolución 
    GeneticAlgorithm(const GAConfig& cfg,int inputSize,int outputSize,ActivationType act = ActivationType::RELU);
    
    // DataSet
    Individual evolve(const vector<vector<double>>& X,const vector<vector<double>>& Y);
    
    // Juego para de Atari
    Individual evolveWithCustomFitness(function<double(Individual&)> fitnessFunc);
    
    // Ejecuta una generación. Devuelve true si llegó al objetivo
    bool runGeneration();
    
    vector<Individual> selection();          // Selecciona padres
    Individual tournamentSelection(int k);        // Selección por torneo
    Individual rouletteSelection();               // Selección por ruleta
    Individual rankSelection();                   // Selección por ranking
    
    
    // Crea la siguiente generación a partir de los padres
    void breed(const vector<Individual>& parents);
    
    void evaluatePopulation(const vector<vector<double>>& X,const vector<vector<double>>& Y);
    void evaluatePopulationCustom();
    // Getters
    const Individual& getBest() const;
    double getBestFitness() const;
    double getAverageFitness() const;
    int getCurrentGeneration() const { return currentGeneration; }
    const vector<double>& getBestFitnessHistory() const { return bestFitnessHistory; }
    const vector<double>& getAvgFitnessHistory() const { return avgFitnessHistory; }
    const vector<Individual>& getPopulation() const { return population; }
    
    void setFitnessFunction(function<double(Individual&)> func) {
        fitnessFunction = func;
    }
    
    void saveBest(const string& filepath) const;
    void loadPopulation(const string& directory);
    void printStats() const;
    
private:
    void sortPopulation();            // Ordena por fitness (mejor primero)
    vector<Individual> getElite(); // Devuelve los élite
};

#endif
