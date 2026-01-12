/*
 * GENETICALGORITHM.CPP - Implementación del algoritmo genético
 */

#include "GeneticAlgorithm/GeneticAlgorithm.hpp"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <random>
#include <filesystem>

// ============================================
// CONSTRUCTORES
// ============================================

// Constructor para evolución de pesos (arquitectura fija)
GeneticAlgorithm::GeneticAlgorithm(const GAConfig& cfg,
                                   const std::vector<int>& topology,
                                   ActivationType act)
    : config(cfg), currentGeneration(0) {
    
    // Crear población con la misma arquitectura
    population.reserve(config.populationSize);
    for (int i = 0; i < config.populationSize; ++i) {
        population.emplace_back(topology, act);
    }
    
    if (config.verbose) {
        std::cout << "\n=== Algoritmo Genético Inicializado ===\n";
        std::cout << "Población: " << config.populationSize << " individuos\n";
        std::cout << "Arquitectura fija: ";
        for (int t : topology) std::cout << t << " ";
        std::cout << "\nModo: Evolución de PESOS solamente (Nota 1.75)\n";
        std::cout << "Tasa de mutación: " << config.mutationRate << "\n";
        std::cout << "Élite: " << (config.eliteRatio * 100) << "%\n\n";
    }
}

// Constructor para neuroevolución (arquitectura variable)
GeneticAlgorithm::GeneticAlgorithm(const GAConfig& cfg,
                                   int inputSize,
                                   int outputSize,
                                   ActivationType act)
    : config(cfg), currentGeneration(0) {
    
    config.evolveArchitecture = true;
    
    // Crear población con arquitecturas random
    population.reserve(config.populationSize);
    for (int i = 0; i < config.populationSize; ++i) {
        population.emplace_back(inputSize, outputSize,
                               config.minHiddenLayers, config.maxHiddenLayers,
                               config.minNeuronsPerLayer, config.maxNeuronsPerLayer,
                               act);
    }
    
    if (config.verbose) {
        std::cout << "\n=== Algoritmo Genético (Neuroevolución) Inicializado ===\n";
        std::cout << "Población: " << config.populationSize << " individuos\n";
        std::cout << "Entradas: " << inputSize << ", Salidas: " << outputSize << "\n";
        std::cout << "Modo: NEUROEVOLUCIÓN - Arquitectura + Pesos (Nota 2.00)\n";
        std::cout << "Capas ocultas: " << config.minHiddenLayers << "-" << config.maxHiddenLayers << "\n";
        std::cout << "Neuronas/capa: " << config.minNeuronsPerLayer << "-" << config.maxNeuronsPerLayer << "\n";
        std::cout << "Tasa mutación pesos: " << config.mutationRate << "\n";
        std::cout << "Tasa mutación arquitectura: " << config.archMutationRate << "\n\n";
    }
}

// ============================================
// EVOLUCIÓN
// ============================================

// Evoluciona con dataset
Individual GeneticAlgorithm::evolve(const std::vector<std::vector<double>>& X,
                                    const std::vector<std::vector<double>>& Y) {
    // Crear función de fitness con el dataset
    fitnessFunction = [&X, &Y](Individual& ind) {
        return ind.calculateFitness(X, Y);
    };
    
    return evolveWithCustomFitness(fitnessFunction);
}

// Evoluciona con función de fitness custom
Individual GeneticAlgorithm::evolveWithCustomFitness(
    std::function<double(Individual&)> fitnessFunc) {
    
    fitnessFunction = fitnessFunc;
    currentGeneration = 0;
    bestFitnessHistory.clear();
    avgFitnessHistory.clear();
    
    if (config.verbose) {
        std::cout << "Iniciando evolución...\n";
        std::cout << "Generaciones máximas: " << config.maxGenerations << "\n";
        std::cout << "Fitness objetivo: " << config.targetFitness << "\n\n";
    }
    
    // Loop principal de evolución
    for (int gen = 0; gen < config.maxGenerations; ++gen) {
        currentGeneration = gen;
        
        if (runGeneration()) {
            if (config.verbose) {
                std::cout << "\n¡Objetivo alcanzado en generación " << gen << "!\n";
            }
            break;
        }
    }
    
    // Guardar el mejor
    if (!config.saveDir.empty()) {
        try {
            std::filesystem::create_directories(config.saveDir);
            saveBest(config.saveDir + "best_individual.txt");
        } catch (...) {}
    }
    
    return getBest();
}

// Ejecuta una generación
bool GeneticAlgorithm::runGeneration() {
    // 1. Evaluar fitness
    evaluatePopulationCustom();
    
    // 2. Ordenar (mejor primero)
    sortPopulation();
    
    // 3. Guardar stats
    double bestFit = getBestFitness();
    double avgFit = getAverageFitness();
    bestFitnessHistory.push_back(bestFit);
    avgFitnessHistory.push_back(avgFit);
    
    // 4. Mostrar progreso
    if (config.verbose && (currentGeneration % config.printEvery == 0)) {
        printStats();
    }
    
    // 5. ¿Llegamos al objetivo?
    if (bestFit >= config.targetFitness) {
        return true;
    }
    
    // 6. Selección y reproducción
    auto parents = selection();
    breed(parents);
    
    return false;
}

// ============================================
// SELECCIÓN
// ============================================

// Selecciona padres para la siguiente generación
std::vector<Individual> GeneticAlgorithm::selection() {
    int numElite = static_cast<int>(config.eliteRatio * config.populationSize);
    numElite = std::max(1, numElite);
    
    std::vector<Individual> selected;
    
    // Los élite pasan directo
    for (int i = 0; i < numElite; ++i) {
        selected.push_back(population[i]);
    }
    
    // Completar con el método de selección elegido
    int remaining = config.populationSize / 2 - numElite;
    remaining = std::max(0, remaining);
    
    for (int i = 0; i < remaining; ++i) {
        switch (config.selectionType) {
            case SelectionType::TOURNAMENT:
                selected.push_back(tournamentSelection(config.tournamentSize));
                break;
            case SelectionType::ROULETTE:
                selected.push_back(rouletteSelection());
                break;
            case SelectionType::RANK:
                selected.push_back(rankSelection());
                break;
            case SelectionType::ELITISM:
                break;  // Solo élite
        }
    }
    
    return selected;
}

// Selección por torneo: elige k random, gana el mejor
Individual GeneticAlgorithm::tournamentSelection(int k) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, population.size() - 1);
    
    Individual best = population[dis(gen)];
    
    for (int i = 1; i < k; ++i) {
        Individual contender = population[dis(gen)];
        if (contender.getFitness() > best.getFitness()) {
            best = contender;
        }
    }
    
    return best;
}

// Selección por ruleta: prob proporcional al fitness
Individual GeneticAlgorithm::rouletteSelection() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    // Sumar fitness (hacer todos positivos)
    double totalFitness = 0.0;
    double minFitness = population[0].getFitness();
    for (const auto& ind : population) {
        if (ind.getFitness() < minFitness) minFitness = ind.getFitness();
    }
    
    double offset = (minFitness < 0) ? -minFitness + 1 : 0;
    
    for (const auto& ind : population) {
        totalFitness += ind.getFitness() + offset;
    }
    
    std::uniform_real_distribution<> dis(0.0, totalFitness);
    double pick = dis(gen);
    
    double current = 0.0;
    for (const auto& ind : population) {
        current += ind.getFitness() + offset;
        if (current >= pick) {
            return ind;
        }
    }
    
    return population.back();
}

// Selección por ranking: prob proporcional a la posición
Individual GeneticAlgorithm::rankSelection() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    int n = population.size();
    int totalRank = n * (n + 1) / 2;
    
    std::uniform_int_distribution<> dis(1, totalRank);
    int pick = dis(gen);
    
    int current = 0;
    for (size_t i = 0; i < population.size(); ++i) {
        current += n - i;
        if (current >= pick) {
            return population[i];
        }
    }
    
    return population.back();
}

// ============================================
// REPRODUCCIÓN
// ============================================

// Crea la siguiente generación
void GeneticAlgorithm::breed(const std::vector<Individual>& parents) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> parentDis(0, parents.size() - 1);
    
    std::vector<Individual> newPopulation;
    
    // Mantener élite
    int numElite = static_cast<int>(config.eliteRatio * config.populationSize);
    for (int i = 0; i < numElite && i < static_cast<int>(parents.size()); ++i) {
        newPopulation.push_back(parents[i]);
    }
    
    // Generar hijos
    while (static_cast<int>(newPopulation.size()) < config.populationSize) {
        int p1 = parentDis(gen);
        int p2;
        do {
            p2 = parentDis(gen);
        } while (p2 == p1 && parents.size() > 1);
        
        Individual child(parents[0].getTopology());
        
        if (config.evolveArchitecture) {
            // Neuroevolución
            child = parents[p1].crossoverNeuroevolution(
                parents[p2], 
                config.mutationRate,
                config.archMutationRate);
        } else {
            // Solo pesos
            child = parents[p1].crossover(parents[p2], config.mutationRate);
        }
        
        newPopulation.push_back(child);
    }
    
    population = newPopulation;
}

// ============================================
// EVALUACIÓN
// ============================================

void GeneticAlgorithm::evaluatePopulation(const std::vector<std::vector<double>>& X,
                                          const std::vector<std::vector<double>>& Y) {
    for (auto& ind : population) {
        ind.calculateFitness(X, Y);
    }
}

void GeneticAlgorithm::evaluatePopulationCustom() {
    if (!fitnessFunction) {
        throw std::runtime_error("Función de fitness no establecida");
    }
    
    for (auto& ind : population) {
        fitnessFunction(ind);
    }
}

// ============================================
// UTILIDADES
// ============================================

void GeneticAlgorithm::sortPopulation() {
    std::sort(population.begin(), population.end());
}

std::vector<Individual> GeneticAlgorithm::getElite() {
    int numElite = static_cast<int>(config.eliteRatio * config.populationSize);
    return std::vector<Individual>(population.begin(), 
                                   population.begin() + numElite);
}

const Individual& GeneticAlgorithm::getBest() const {
    return population.front();
}

double GeneticAlgorithm::getBestFitness() const {
    if (population.empty()) return 0.0;
    return population.front().getFitness();
}

double GeneticAlgorithm::getAverageFitness() const {
    if (population.empty()) return 0.0;
    
    double sum = 0.0;
    for (const auto& ind : population) {
        sum += ind.getFitness();
    }
    return sum / population.size();
}

void GeneticAlgorithm::saveBest(const std::string& filepath) const {
    if (!population.empty()) {
        population.front().save(filepath);
    }
}

void GeneticAlgorithm::printStats() const {
    std::cout << "Gen " << currentGeneration 
             << " | Best: " << getBestFitness()
             << " | Avg: " << getAverageFitness();
    
    if (config.evolveArchitecture) {
        std::cout << " | Arch: " << getBest().getArchitectureString();
    }
    
    std::cout << " | Params: " << getBest().getTotalParameters() << "\n";
}

void GeneticAlgorithm::loadPopulation(const std::string& directory) {
    // TODO: cargar población desde archivos
}
