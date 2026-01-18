#include "GeneticAlgorithm/Individual.hpp"
#include "ale_interface.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

// 1. IMPORTANTE: LA MISMA RAM QUE EN ENTRENAMIENTO
// Si cambias un solo byte aquí, el cerebro de la IA no funcionará.
const std::vector<int> ramImportant = { 
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
    
    // === CONFIGURACIÓN VISUAL ===
    ale.setBool("display_screen", true); 
    ale.setBool("sound", true);           
    ale.setInt("random_seed", 123);
    
    // 2. IMPORTANTE: EL MISMO FRAME SKIP (3)
    ale.setInt("frame_skip", 3);          
    
    ale.loadROM("supported/assault.bin");

    // 3. IMPORTANTE: LAS MISMAS ACCIONES QUE EN ENTRENAMIENTO
    // El modelo espera decidir entre 3 salidas, no 6.
    ActionVect legal_actions = {
        PLAYER_A_NOOP, 
        PLAYER_A_LEFT, 
        PLAYER_A_RIGHT
    };

    // --- CARGAR AGENTE ---
    Individual agent({1}); // Placeholder
    std::string modelPath = "models/assault_auto_fire.txt"; 

    try {
        std::cout << "Cargando modelo desde: " << modelPath << "...\n";
        agent.load(modelPath);
        std::cout << "Modelo cargado con exito.\n";
    } catch (const std::exception& e) {
        std::cerr << "Error cargando el modelo: " << e.what() << "\n";
        std::cerr << "Asegurate de que el archivo existe y la ruta es correcta.\n";
        return 1;
    }

    // --- BUCLE DE JUEGO ---
    ale.reset_game();
    double totalReward = 0.0;
    int frames = 0;
    Action lastMove = PLAYER_A_NOOP; // Memoria para el auto-fire

    std::cout << "Iniciando partida DEMO..." << std::endl;

    while (!ale.game_over()) {
        frames++;

        // A. Obtener estado NORMALIZADO (-0.5 a 0.5)
        std::vector<double> state;
        const auto& RAM = ale.getRAM();
        state.reserve(ramImportant.size());
        for (int idx : ramImportant) {
            state.push_back((static_cast<double>(RAM.get(idx)) / 255.0) - 0.5);
        }

        // B. El agente piensa
        std::vector<double> output = agent.predict(state);
        int actionIdx = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
        Action desiredAction = legal_actions[actionIdx];

        // C. LOGICA HIBRIDA (DIRECTIONAL AUTO-FIRE)
        // Replicamos exactamente lo que hicimos en el entrenamiento
        Action actionToExecute;

        if (frames % 2 == 0) {
            // Frame PAR: Turno de la IA (Moverse)
            actionToExecute = desiredAction;
            lastMove = desiredAction; // Recordamos intencion
        } else {
            // Frame IMPAR: Turno del Codigo (Disparar manteniendo direccion)
            if (lastMove == PLAYER_A_LEFT) {
                actionToExecute = PLAYER_A_LEFTFIRE;
            } else if (lastMove == PLAYER_A_RIGHT) {
                actionToExecute = PLAYER_A_RIGHTFIRE;
            } else {
                actionToExecute = PLAYER_A_FIRE;
            }
        }

        // D. Actuar
        totalReward += ale.act(actionToExecute);
    }

    std::cout << "GAME OVER. Puntuación final: " << totalReward << std::endl;

    return 0;
}