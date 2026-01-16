#include "Perceptron.hpp"
#include "ale_interface.hpp"
#include <iostream>
#include <vector>

// Indices de RAM (63 elementos)
const std::vector<int> ramImportant = {
    15,47,48,49,50,51,52,20,21,23,24,25,39,71,109,113,
    16,18,32,33,34,35,36,37,44,46,42,60,101,102,106,121,67,68,79,80,
    53,54,55,56,61,62,65,69,70,72,74,85,87,91,92,104,105,114,119,120,123,125,126
};

int main() {
    ALEInterface ale;
    ale.setBool("display_screen", true); 
    ale.setBool("sound", true);
    ale.loadROM("supported/assault.bin");

    // Cargamos Perceptrón (63 inputs, 3 outputs)
    Perceptron agent(ramImportant.size(), 3); 
    try {
        agent.load("models/assault_perceptron.txt");
        std::cout << "Modelo cargado. Neuronas de salida: 3 (Der, Izq, Fuego)\n";
    } catch (...) {
        std::cerr << "Error: Modelo no encontrado o incorrecto.\n";
        return 1;
    }

    while (!ale.game_over()) {
        // 1. Inputs
        const auto& RAM = ale.getRAM();
        std::vector<double> state;
        for (int idx : ramImportant) state.push_back(RAM.get(idx) / 255.0);

        // 2. Predicción (Vector de 3 valores)
        // [0]: Derecha?, [1]: Izquierda?, [2]: Fuego?
        auto scores = agent.predict(state);
        
        bool reqRight = scores[0] >= 0;
        bool reqLeft  = scores[1] >= 0;
        bool reqFire  = scores[2] >= 0;

        // 3. Decodificador de Acciones (Mapping a ALE)
        // Assault Actions: 0=NOOP, 1=FIRE, 2=UP, 3=RIGHT, 4=LEFT, ...
        // 35=FIRERIGHT, 36=FIRELEFT
        
        Action action = (Action)0; // Default NOOP

        if (reqFire) {
            if (reqRight) action = (Action)35;      // FIRE + RIGHT
            else if (reqLeft) action = (Action)36;  // FIRE + LEFT
            else action = (Action)1;                // FIRE SOLO
        } else {
            if (reqRight) action = (Action)3;       // RIGHT SOLO
            else if (reqLeft) action = (Action)4;   // LEFT SOLO
            else action = (Action)0;                // NOOP
        }

        ale.act(action);
    }
    
    std::cout << "Final: " << ale.getEpisodeFrameNumber() << std::endl;
    return 0;
}