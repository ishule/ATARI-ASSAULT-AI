#include "GeneticAlgorithm/GeneticAlgorithm.hpp"
#include "GeneticAlgorithm/Individual.hpp"
#include "ale_interface.hpp"
#include <iostream>
#include <vector>
#include <algorithm> // Para max_element

// Índices de RAM importantes para Assault
const std::vector<int> ramImportant = {
    15,47,48,49,50,51,52,20,21,23,24,25,39,71,109,113,
    16,18,32,33,34,35,36,37,44,46,42,60,101,102,106,121,67,68,79,80,
    53,54,55,56,61,62,65,69,70,72,74,85,87,91,92,104,105,114,119,120,123,125,126
};

int main() {
    // 1. Configuración inicial de ALE para obtener dimensiones
    ALEInterface ale_setup;
    ale_setup.loadROM("supported/assault.bin");
    int inputSize = ramImportant.size();
    int outputSize = ale_setup.getLegalActionSet().size();

    // 2. Configuración del Algoritmo Genético
    GAConfig config;
    config.populationSize = 50;       // Población decente
    config.mutationRate = 0.1;        // Diversidad estándar
    config.eliteRatio = 0.1;          // Estabilidad (mejores pasan siempre)
    config.targetFitness = 10000;     // Objetivo optimista
    config.verbose = true;
    config.saveDir = "models/ga/";    // Carpeta para checkpoints
    config.maxGenerations = 100;      // Límite para que termine en unas horas

    // Topología: Input -> 32 neuronas (Oculta) -> Output
    std::vector<int> topology = {inputSize, 32, outputSize};

    // Instancia del GA
    GeneticAlgorithm ga(config, topology);

    std::cout << "Iniciando evolución para Atari Assault..." << std::endl;
    std::cout << "Los checkpoints se guardarán en: " << config.saveDir << std::endl;

    // 3. Ejecutar Evolución
    Individual best = ga.evolveWithCustomFitness([&](Individual& ind) {
        // Instancia ALE por hilo/individuo
        ALEInterface ale;
        ale.setBool("display_screen", false); 
        ale.setBool("sound", false);
        ale.setInt("frame_skip", 4);    // VELOCIDAD X4 (Crítico)
        ale.setInt("random_seed", 123);
        ale.loadROM("supported/assault.bin");
        
        double totalReward = 0.0;
        int steps = 0;
        int maxSteps = 4500; // PROTECCIÓN: Límite de pasos por partida
        
        while (!ale.game_over() && steps < maxSteps) {
            steps++;

            // A. Extraer estado de la RAM
            std::vector<double> state;
            const auto& RAM = ale.getRAM();
            state.reserve(ramImportant.size());
            
            for (int idx : ramImportant) {
                state.push_back(static_cast<double>(RAM.get(idx)) / 255.0);
            }

            // B. Predecir acción
            std::vector<double> output = ind.predict(state);
            
            // C. Elegir la acción
            int actionIdx = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
            Action action = ale.getLegalActionSet()[actionIdx];
            
            // D. Ejecutar acción
            totalReward += ale.act(action);
        }
        return totalReward;
    });

    // 4. Guardar resultado final
    std::cout << "Entrenamiento finalizado." << std::endl;
    std::cout << "Mejor puntuación final: " << best.getFitness() << std::endl;
    best.save("models/assault_agent_final.txt");

    return 0;
}