#include "Perceptron.hpp"
#include "ale_interface.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

// Indices de RAM
const std::vector<int> ramImportant = {
    15,47,48,49,50,51,52,20,21,23,24,25,39,71,109,113,
    16,18,32,33,34,35,36,37,44,46,42,60,101,102,106,121,67,68,79,80,
    53,54,55,56,61,62,65,69,70,72,74,85,87,91,92,104,105,114,119,120,123,125,126
};

std::string actionToString(Action a) {
    if (a == PLAYER_A_FIRE) return "FUEGO";
    if (a == PLAYER_A_RIGHT) return "DERECHA";
    if (a == PLAYER_A_LEFT) return "IZQUIERDA";
    if (a == PLAYER_A_RIGHTFIRE) return "FUEGO+DER";
    if (a == PLAYER_A_LEFTFIRE) return "FUEGO+IZQ";
    return "QUIETO";
}

int main() {
    ALEInterface ale;
    ale.setBool("display_screen", true); 
    ale.setBool("sound", true);
    ale.loadROM("supported/assault.bin");

    Perceptron agent(ramImportant.size(), 3); 
    try {
        agent.load("models/assault_perceptron.txt");
        std::cout << ">>> CEREBRO CARGADO OK.\n";
    } catch (...) {
        std::cerr << "ERROR: No encuentro 'models/assault_perceptron.txt'.\n";
        return 1;
    }

    std::cout << ">>> KICKSTART...\n";
    for(int i=0; i<40; ++i) ale.act(PLAYER_A_FIRE);

    std::cout << "--------------------------------------------------------\n";
    std::cout << " VALORES CRUDOS (SCORES)      | DECISION   | PUNTOS \n";
    std::cout << "--------------------------------------------------------\n";

    // --- CORRECCIÓN: Variable para llevar la cuenta de los puntos ---
    double totalScore = 0.0;

    while (!ale.game_over()) {
        const auto& RAM = ale.getRAM();
        std::vector<double> state;
        for (int idx : ramImportant) state.push_back((double)RAM.get(idx) / 255.0);

        // 1. Obtener valores
        auto scores = agent.predict(state);
        double valRight = scores[0];
        double valLeft  = scores[1];
        double valFire  = scores[2];

        // 2. Lógica de Competición
        bool doFire = valFire >= 0; 
        bool doRight = false;
        bool doLeft  = false;

        if (valRight > 0 || valLeft > 0) {
            if (valRight > valLeft) doRight = true;
            else doLeft = true;
        }
        
        // 3. Elegir Acción
        Action action = PLAYER_A_NOOP;

        if (doFire) {
            if (doRight) action = PLAYER_A_RIGHTFIRE;
            else if (doLeft) action = PLAYER_A_LEFTFIRE;
            else action = PLAYER_A_FIRE;
        } else {
            if (doRight) action = PLAYER_A_RIGHT;
            else if (doLeft) action = PLAYER_A_LEFT;
            else action = PLAYER_A_NOOP;
        }

        // --- CORRECCIÓN: Sumar lo que devuelve act() ---
        totalScore += ale.act(action);

        if (ale.getEpisodeFrameNumber() % 20 == 0) {
            std::cout << " R:" << std::fixed << std::setprecision(2) << valRight 
                      << " L:" << valLeft 
                      << " F:" << valFire << " | "
                      << std::setw(10) << actionToString(action) << " | "
                      << (int)totalScore // Imprimimos la variable acumulada
                      << "\r" << std::flush;
        }
    }
    
    std::cout << "\nFIN. Score Real: " << totalScore << std::endl;
    return 0;
}