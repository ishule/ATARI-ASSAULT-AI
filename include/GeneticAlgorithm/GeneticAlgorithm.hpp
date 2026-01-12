/*
 * GENETIC ALGORITHM - El algoritmo genético que evoluciona las redes
 * 
 * Esto es lo que controla toda la evolución:
 * - Crea la población inicial
 * - Evalúa el fitness de cada individuo
 * - Selecciona los mejores para reproducirse
 * - Hace crossover y mutación
 * - Repite hasta encontrar una buena solución
 * 
 * Dos modos:
 * - Evolución de pesos (nota 1.75): arquitectura fija
 * - Neuroevolución (nota 2.00): arquitectura + pesos
 */

#ifndef GENETIC_ALGORITHM_HPP
#define GENETIC_ALGORITHM_HPP

#include "GeneticAlgorithm/Individual.hpp"
#include <vector>
#include <string>
#include <functional>

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
    std::string saveDir = "models/ga/";
};

class GeneticAlgorithm {
private:
    GAConfig config;
    std::vector<Individual> population;       // La población
    
    // Estadísticas
    std::vector<double> bestFitnessHistory;
    std::vector<double> avgFitnessHistory;
    int currentGeneration;
    
    // Función de fitness custom (opcional)
    std::function<double(Individual&)> fitnessFunction;
    
public:
    // === CONSTRUCTORES ===
    
    // Para evolución de pesos (arquitectura fija)
    GeneticAlgorithm(const GAConfig& cfg,
                     const std::vector<int>& topology,
                     ActivationType act = ActivationType::RELU);
    
    // Para neuroevolución (arquitectura variable)
    GeneticAlgorithm(const GAConfig& cfg,
                     int inputSize,
                     int outputSize,
                     ActivationType act = ActivationType::RELU);
    
    // === EVOLUCIÓN ===
    
    // Evoluciona con un dataset (lo típico)
    Individual evolve(const std::vector<std::vector<double>>& X,
                      const std::vector<std::vector<double>>& Y);
    
    // Evoluciona con función de fitness custom (para juegos etc)
    Individual evolveWithCustomFitness(
        std::function<double(Individual&)> fitnessFunc);
    
    // Ejecuta una generación. Devuelve true si llegó al objetivo
    bool runGeneration();
    
    // === SELECCIÓN ===
    
    std::vector<Individual> selection();          // Selecciona padres
    Individual tournamentSelection(int k);        // Selección por torneo
    Individual rouletteSelection();               // Selección por ruleta
    Individual rankSelection();                   // Selección por ranking
    
    // === REPRODUCCIÓN ===
    
    // Crea la siguiente generación a partir de los padres
    void breed(const std::vector<Individual>& parents);
    
    // === EVALUACIÓN ===
    
    void evaluatePopulation(const std::vector<std::vector<double>>& X,
                           const std::vector<std::vector<double>>& Y);
    void evaluatePopulationCustom();
    
    // === GETTERS ===
    
    const Individual& getBest() const;
    double getBestFitness() const;
    double getAverageFitness() const;
    int getCurrentGeneration() const { return currentGeneration; }
    const std::vector<double>& getBestFitnessHistory() const { return bestFitnessHistory; }
    const std::vector<double>& getAvgFitnessHistory() const { return avgFitnessHistory; }
    const std::vector<Individual>& getPopulation() const { return population; }
    
    // === UTILS ===
    
    void setFitnessFunction(std::function<double(Individual&)> func) {
        fitnessFunction = func;
    }
    
    void saveBest(const std::string& filepath) const;
    void loadPopulation(const std::string& directory);
    void printStats() const;
    
private:
    void sortPopulation();            // Ordena por fitness (mejor primero)
    std::vector<Individual> getElite(); // Devuelve los élite
};

#endif
