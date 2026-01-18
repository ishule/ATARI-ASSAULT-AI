#include "GeneticAlgorithm/GeneticAlgorithm.hpp"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <random>
#include <filesystem>
using namespace std;

// Constructor para evolución de pesos 
GeneticAlgorithm::GeneticAlgorithm(const GAConfig& cfg, const vector<int>& topology, ActivationType act) : config(cfg), currentGeneration(0) {
    population.reserve(config.populationSize);
    for (int i = 0; i < config.populationSize; ++i) {
        population.emplace_back(topology, act);
    }
    if (config.verbose) {
        cout << "\n=== Algoritmo Genético Inicializado ===\n";
        cout << "Población: " << config.populationSize << " individuos\n";
        cout << "Arquitectura fija: ";
        for (int t : topology) cout << t << " ";
        cout << "\nModo: Evolución de PESOS\n";
        cout << "Tasa de mutación: " << config.mutationRate << "\n";
        cout << "Élite: " << (config.eliteRatio * 100) << "%\n\n";
    }
}

// Constructor para neuroevolución 
GeneticAlgorithm::GeneticAlgorithm(const GAConfig& cfg, int inputSize, int outputSize, ActivationType act) : config(cfg), currentGeneration(0) {
    config.evolveArchitecture = true;
    population.reserve(config.populationSize);
    for (int i = 0; i < config.populationSize; ++i) {
        population.emplace_back(inputSize, outputSize, config.minHiddenLayers, config.maxHiddenLayers, config.minNeuronsPerLayer, config.maxNeuronsPerLayer, act);
    }
    if (config.verbose) {
        cout << "\n=== Algoritmo Genético (Neuroevolución) Inicializado ===\n";
        cout << "Población: " << config.populationSize << " individuos\n";
        cout << "Entradas: " << inputSize << ", Salidas: " << outputSize << "\n";
        cout << "Modo: Neuroevolucion\n";
        cout << "Capas ocultas: " << config.minHiddenLayers << "-" << config.maxHiddenLayers << "\n";
        cout << "Neuronas/capa: " << config.minNeuronsPerLayer << "-" << config.maxNeuronsPerLayer << "\n";
        cout << "Tasa mutación pesos: " << config.mutationRate << "\n";
        cout << "Tasa mutación arquitectura: " << config.archMutationRate << "\n\n";
    }
}

// Evoluciona con dataset
Individual GeneticAlgorithm::evolve(const vector<vector<double>>& X, const vector<vector<double>>& Y) {
    fitnessFunction = [&X, &Y](Individual& ind) {
        return ind.calculateFitness(X, Y);
    };
    return evolveWithCustomFitness(fitnessFunction);
}

// Evoluciona con función de fitness custom
Individual GeneticAlgorithm::evolveWithCustomFitness(function<double(Individual&)> fitnessFunc) {
    fitnessFunction = fitnessFunc;
    currentGeneration = 0;
    bestFitnessHistory.clear();
    avgFitnessHistory.clear();
    if (config.verbose) {
        cout << "Iniciando evolución...\n";
        cout << "Generaciones máximas: " << config.maxGenerations << "\n";
        cout << "Fitness objetivo: " << config.targetFitness << "\n\n";
    }
    for (int gen = 0; gen < config.maxGenerations; ++gen) {
        currentGeneration = gen;
        if (runGeneration()) {
            if (config.verbose) {
                cout << "\n¡Objetivo alcanzado en generación " << gen << "!\n";
            }
            break;
        }
    }
    if (!config.saveDir.empty()) {
        try {
            filesystem::create_directories(config.saveDir); // Asegura que el directorio existe
            saveBest(config.saveDir + "best_individual.txt");
        } catch (...) {}
    }
    return getBest();
}

// Ejecuta una generación
bool GeneticAlgorithm::runGeneration() {
    evaluatePopulationCustom();
    sortPopulation();
    double bestFit = getBestFitness();
    double avgFit = getAverageFitness();
    bestFitnessHistory.push_back(bestFit);
    avgFitnessHistory.push_back(avgFit);
    if (config.verbose && (currentGeneration % config.printEvery == 0)) { 
        printStats();
    }
    if (bestFit >= config.targetFitness) {
        return true;
    }
    auto parents = selection(); // Selección de padres
    breed(parents);
    return false;
}
// Seleccion de padres 
vector<Individual> GeneticAlgorithm::selection() {
    int numElite = static_cast<int>(config.eliteRatio * config.populationSize);
    numElite = max(1, numElite);
    vector<Individual> selected;
    for (int i = 0; i < numElite; ++i) {
        selected.push_back(population[i]);
    }
    int remaining = config.populationSize / 2 - numElite;
    remaining = max(0, remaining);
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
                break;
        }
    }
    return selected;
}
// Selección por torneo
Individual GeneticAlgorithm::tournamentSelection(int k) {
    static random_device rd;
    static mt19937 gen(rd());
    uniform_int_distribution<> dis(0, population.size() - 1);
    Individual best = population[dis(gen)];
    for (int i = 1; i < k; ++i) {
        Individual contender = population[dis(gen)];
        if (contender.getFitness() > best.getFitness()) {
            best = contender;
        }
    }
    return best;
}
// Selección por ruleta
Individual GeneticAlgorithm::rouletteSelection() {
    static random_device rd;
    static mt19937 gen(rd());
    double totalFitness = 0.0;
    double minFitness = population[0].getFitness();
    for (const auto& ind : population) {
        if (ind.getFitness() < minFitness) minFitness = ind.getFitness();
    }
    double offset = (minFitness < 0) ? -minFitness + 1 : 0;
    for (const auto& ind : population) {
        totalFitness += ind.getFitness() + offset;
    }
    uniform_real_distribution<> dis(0.0, totalFitness);
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
// Selección por ranking
Individual GeneticAlgorithm::rankSelection() {
    static random_device rd;
    static mt19937 gen(rd());
    int n = population.size();
    int totalRank = n * (n + 1) / 2;
    uniform_int_distribution<> dis(1, totalRank);
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

// Crea la siguiente generación
void GeneticAlgorithm::breed(const vector<Individual>& parents) {
    static random_device rd;
    static mt19937 gen(rd());
    uniform_int_distribution<> parentDis(0, parents.size() - 1);
    vector<Individual> newPopulation;
    int numElite = static_cast<int>(config.eliteRatio * config.populationSize); // Mantener élite
    for (int i = 0; i < numElite && i < static_cast<int>(parents.size()); ++i) { // Asegura no exceder el tamaño de parents
        newPopulation.push_back(parents[i]);
    }
    // Rellenar el resto de la población con hijos
    while (static_cast<int>(newPopulation.size()) < config.populationSize) {
        int p1 = parentDis(gen);
        int p2;
        do {
            p2 = parentDis(gen);
        } while (p2 == p1 && parents.size() > 1);
        Individual child(parents[0].getTopology());
        if (config.evolveArchitecture) {
            child = parents[p1].crossoverNeuroevolution(parents[p2], config.mutationRate, config.archMutationRate);
        } else {
            child = parents[p1].crossover(parents[p2], config.mutationRate);
        }
        newPopulation.push_back(child);
    }
    population = move(newPopulation); // con move va mucho más rápido porque se copian punteros
}

// Evaluar la población usando datos de entrada y salida
void GeneticAlgorithm::evaluatePopulation(const vector<vector<double>>& X, const vector<vector<double>>& Y) {
    for (auto& ind : population) {
        ind.calculateFitness(X, Y);
    }
}
// En GeneticAlgorithm.cpp

void GeneticAlgorithm::evaluatePopulationCustom() {
    if (!fitnessFunction) {
        throw runtime_error("Función de fitness no establecida");
    }
    
    int counter = 0;
    int total = population.size();

    for (auto& ind : population) {
        // 🔥🔥 CORRECCIÓN CRÍTICA 🔥🔥
        // Antes tenías: fitnessFunction(ind);  --> Calculaba y tiraba el valor
        // Ahora hacemos:
        double result = fitnessFunction(ind);
        ind.setFitness(result); // <-- ¡GUARDAMOS LA NOTA!
        
    }
}


void GeneticAlgorithm::sortPopulation() {
    sort(population.begin(), population.end());
}

vector<Individual> GeneticAlgorithm::getElite() {
    int numElite = static_cast<int>(config.eliteRatio * config.populationSize);
    return vector<Individual>(population.begin(), population.begin() + numElite);
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

void GeneticAlgorithm::saveBest(const string& filepath) const {
    if (!population.empty()) {
        population.front().save(filepath);
    }
}

void GeneticAlgorithm::printStats() const {
    cout << "Gen " << currentGeneration 
         << " | Best: " << getBestFitness()
         << " | Avg: " << getAverageFitness();
    if (config.evolveArchitecture) {
        cout << " | Arch: " << getBest().getArchitectureString();
    }
    cout << " | Params: " << getBest().getTotalParameters() << "\n";
}

void GeneticAlgorithm::loadPopulation(const string& directory) {
  
}
