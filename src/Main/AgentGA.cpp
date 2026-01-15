#include "GeneticAlgorithm/GeneticAlgorithm.hpp"
#include "GeneticAlgorithm/Individual.hpp"
#include "ale/ale_interface.hpp"
#include <iostream>
#include <vector>
#include <functional>

// RAM importante para Atari Assault (igual que AgentManual)
const std::vector<int> ramImportant = {
    15,47,48,49,50,51,52,20,21,23,24,25,39,71,109,113,
    16,18,32,33,34,35,36,37,44,46,42,60,101,102,106,121,67,68,79,80,
    53,54,55,56,61,62,65,69,70,72,74,85,87,91,92,104,105,114,119,120,123,125,126
};

int main() {
    ALEInterface ale;
    ale.setInt("random_seed", 123);
    ale.setBool("display_screen", false);
    ale.setBool("sound", false);
    ale.setInt("frame_skip", 0);
    ale.loadROM("supported/assault.bin");

    // Configuración GA
    GeneticAlgorithmConfig config;
    config.populationSize = 50;
    config.mutationRate = 0.1;
    config.eliteRatio = 0.1;
    config.targetFitness = 10000;

    // Topología: entradas = ramImportant.size(), capa oculta, salidas = acciones posibles
    int inputSize = ramImportant.size();
    int outputSize = ale.getLegalActionSet().size();
    std::vector<int> topology = {inputSize, 32, outputSize};

    GeneticAlgorithm ga(topology, config);

    // Función de fitness: juega una partida y devuelve la puntuación
    ga.setFitnessFunction([&ale](Individual& ind) {
        ale.reset_game();
        double totalReward = 0.0;
        while (!ale.game_over()) {
            // Extrae RAM importante
            std::vector<double> state;
            const auto& RAM = ale.getRAM();
            for (int idx : ramImportant) {
                state.push_back(static_cast<double>(RAM[idx]) / 255.0); // Normaliza
            }
            // El individuo decide la acción
            std::vector<double> output = ind.predict(state);
            int actionIdx = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
            Action action = ale.getLegalActionSet()[actionIdx];
            totalReward += ale.act(action);
        }
        return totalReward;
    });

    // Ejecuta la evolución
    Individual best = ga.evolveWithCustomFitness(ga.getFitnessFunction());
    std::cout << "Mejor puntuación obtenida: " << best.getFitness() << std::endl;
    best.save("models/assault_agent_genetic.txt");

    return 0;
}