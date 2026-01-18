#include "GeneticAlgorithm/Individual.hpp"
#include "ale_interface.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace std;
const vector<int> ramImportant = { 
    0x00, 0x01, 0x02, 0x09, 0x0A, 0x0B, 0x10, 0x11, 0x12, 0x13, 
    0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 
    0x1F, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2B, 
    0x2C, 0x2D, 0x2E, 0x2F, 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 
    0x36, 0x37, 0x38, 0x39, 0x3C, 0x3E, 0x3F, 0x42, 0x43, 0x44, 
    0x45, 0x46, 0x47, 0x48, 0x49, 0x4B, 0x50, 0x51, 0x56, 0x58, 
    0x5A, 0x5C, 0x5D, 0x65, 0x68, 0x69, 0x6A, 0x6C, 0x6E, 0x6F, 
    0x72, 0x73, 0x78, 0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E, 0x7F
};

int main() {
    ALEInterface ale;
    

    ale.setBool("display_screen", true); 
    ale.setBool("sound", true);           
    ale.setInt("random_seed", 123);
    

    ale.setInt("frame_skip", 1);          
    
    ale.loadROM("supported/assault.bin");

    ActionVect move_actions = {
        PLAYER_A_NOOP, 
        PLAYER_A_LEFT, 
        PLAYER_A_RIGHT
    };


    Individual agent({1}); 
    
    // Asegúrate de apuntar al archivo correcto que generó el entrenamiento nuevo
    string modelPath = "models/neuro_ga/assault_sniper_relu_neuro.txt"; 

    try {
        cout << "Cargando modelo SNIPER (4 Neuronas) desde: " << modelPath << "...\n";
        agent.load(modelPath);
        cout << "Modelo cargado con exito.\n";
    } catch (const exception& e) {
        cerr << "Error cargando el modelo: " << e.what() << "\n";
        return 1;
    }

    ale.reset_game();
    double totalReward = 0.0;
    int frames = 0;

    while (!ale.game_over()) {
        frames++;

        vector<double> state;
        const auto& RAM = ale.getRAM();
        state.reserve(ramImportant.size());
        for (int idx : ramImportant) {
            state.push_back((static_cast<double>(RAM.get(idx)) / 255.0) - 0.5);
        }

        vector<double> output = agent.predict(state);

        // Se decide movimiento 
        auto maxIt = std::max_element(output.begin(), output.begin() + 3);
        int moveIdx = std::distance(output.begin(), maxIt);
        Action intendedMove = move_actions[moveIdx];

        // Se decide disparo
        bool wantToShoot = output[3] > 0.0;

        // Se determina la acción final a ejecutar
        Action actionToExecute;

        if (wantToShoot) {
            // Quiere disparar + Moverse
            if (intendedMove == PLAYER_A_LEFT) {
                actionToExecute = PLAYER_A_LEFTFIRE;
            } else if (intendedMove == PLAYER_A_RIGHT) {
                actionToExecute = PLAYER_A_RIGHTFIRE;
            } else {
                actionToExecute = PLAYER_A_UPFIRE; // (Quieto + Disparo)
            }
        } else {
            // Solo quiere moverse (Enfriar arma o esquivar)
            actionToExecute = intendedMove;
        }

        // Actuar
        totalReward += ale.act(actionToExecute);
    }

    cout << "GAME OVER. Puntuación final: " << totalReward << endl;

    return 0;
}