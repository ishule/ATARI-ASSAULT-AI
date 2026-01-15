#include "GeneticAlgorithm/GeneticAlgorithm.hpp"
#include "GeneticAlgorithm/Individual.hpp"
#include "ale/ale_interface.hpp"
#include <iostream>
#include <vector>
#include <algorithm> // Necesario para max_element

// RAM importante para Atari Assault
const std::vector<int> ramImportant = {
    15,47,48,49,50,51,52,20,21,23,24,25,39,71,109,113,
    16,18,32,33,34,35,36,37,44,46,42,60,101,102,106,121,67,68,79,80,
    53,54,55,56,61,62,65,69,70,72,74,85,87,91,92,104,105,114,119,120,123,125,126
};

int main() {
    // Configuración para obtener el tamaño de inputs/outputs solamente
    ALEInterface ale_setup;
    ale_setup.loadROM("supported/assault.bin");
    int inputSize = ramImportant.size();
    int outputSize = ale_setup.getLegalActionSet().size();

    // 1. CORRECCIÓN: Usar GAConfig en lugar de GeneticAlgorithmConfig
    GAConfig config;
    config.populationSize = 50;
    config.mutationRate = 0.1;
    config.eliteRatio = 0.1;
    config.targetFitness = 10000;
    // Opciones adicionales para evitar bugs
    config.verbose = true; 
    config.saveDir = "models/ga/";

    // Topología: Input -> 32 neuronas -> Output
    std::vector<int> topology = {inputSize, 32, outputSize};

    // 2. CORRECCIÓN: Orden de argumentos (config, topology)
    GeneticAlgorithm ga(config, topology);

    // 3. Función de fitness
    // NOTA: Instanciamos ALE dentro para evitar problemas si el GA usa hilos (OpenMP)
    // Si tu GA es estrictamente secuencial, puedes sacar 'ale' fuera para ganar velocidad.
    ga.setFitnessFunction([&](Individual& ind) {
        ALEInterface ale;
        // Configuración mínima para velocidad (sin pantalla, sin sonido)
        ale.setBool("display_screen", false); 
        ale.setBool("sound", false);
        ale.setInt("frame_skip", 0); // O frame_skip 4 para entrenar más rápido
        ale.setInt("random_seed", 123);
        ale.loadROM("supported/assault.bin");
        
        double totalReward = 0.0;
        
        while (!ale.game_over()) {
            // Extraer estado
            std::vector<double> state;
            const auto& RAM = ale.getRAM();
            state.reserve(ramImportant.size()); // Optimización pequeña
            for (int idx : ramImportant) {
                state.push_back(static_cast<double>(RAM[idx]) / 255.0);
            }

            // Predecir
            std::vector<double> output = ind.predict(state);
            
            // Decidir acción
            int actionIdx = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
            Action action = ale.getLegalActionSet()[actionIdx];
            
            totalReward += ale.act(action);
        }
        return totalReward;
    });

    std::cout << "Iniciando evolución para Atari Assault..." << std::endl;

    // Ejecutar evolución
    // Nota: evolveWithCustomFitness ya hace el bucle de generaciones internamente
    Individual best = ga.evolveWithCustomFitness(nullptr); // nullptr porque ya hicimos setFitnessFunction arriba
    
    // O si tu implementación de evolveWithCustomFitness requiere pasar la función:
    // Individual best = ga.evolveWithCustomFitness([&](Individual& ind) { ... misma lógica ... });

    std::cout << "Mejor puntuación obtenida: " << best.getFitness() << std::endl;
    best.save("models/assault_agent_genetic.txt");

    return 0;
}