// AgentPerceptron.cpp
// Juega Atari Assault usando un perceptrón entrenado

#include "Perceptron.hpp"
#include "ale/ale_interface.hpp"
#include <iostream>
#include <vector>
#include <fstream>

// RAM importante para Atari Assault (igual que AgentManual)
const std::vector<int> ramImportant = {
    15,47,48,49,50,51,52,20,21,23,24,25,39,71,109,113,
    16,18,32,33,34,35,36,37,44,46,42,60,101,102,106,121,67,68,79,80,
    53,54,55,56,61,62,65,69,70,72,74,85,87,91,92,104,105,114,119,120,123,125,126
};

int main() {
    ALEInterface ale;
    ale.setInt("random_seed", 123);
    ale.setBool("display_screen", true);
    ale.setBool("sound", true);
    ale.setInt("frame_skip", 0);
    ale.loadROM("supported/assault.bin");

    // Carga el modelo entrenado
    Perceptron agent(ramImportant.size(), ale.getLegalActionSet().size());
    agent.load("models/assault_agent_perceptron.txt");

    double totalReward = 0.0;
    ale.reset_game();
    while (!ale.game_over()) {
        // Extrae RAM importante
        std::vector<double> state;
        const auto& RAM = ale.getRAM();
        for (int idx : ramImportant) {
            state.push_back(static_cast<double>(RAM[idx]) / 255.0); // Normaliza
        }
        // El perceptrón decide la acción
        std::vector<double> output = agent.predict(state);
        int actionIdx = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
        Action action = ale.getLegalActionSet()[actionIdx];
        totalReward += ale.act(action);
    }
    std::cout << "Puntuación final: " << totalReward << std::endl;
    return 0;
}
