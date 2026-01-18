#include "GeneticAlgorithm/Individual.hpp"
#include "ale_interface.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

// 1. IMPORTANTE: LA MISMA RAM QUE EN ENTRENAMIENTO
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
    
    // 2. IMPORTANTE: DEBE COINCIDIR CON EL ENTRENAMIENTO
    // En tu código "Sniper" usaste frame_skip = 1 para máxima precisión.
    // Si lo entrenaste con 3, cámbialo a 3. Pero por defecto Sniper es 1.
    ale.setInt("frame_skip", 1);          
    
    ale.loadROM("supported/assault.bin");

    // 3. ACCIONES BASE (Solo Movimiento)
    // Estas corresponden a las neuronas 0, 1 y 2
    ActionVect move_actions = {
        PLAYER_A_NOOP, 
        PLAYER_A_LEFT, 
        PLAYER_A_RIGHT
    };

    // --- CARGAR AGENTE ---
    // Nota: El inputSize es ramImportant.size() y outputSize debe ser 4
    Individual agent({1}); 
    
    // Asegúrate de apuntar al archivo correcto que generó el entrenamiento nuevo
    std::string modelPath = "models/neuro_ga/assault_sniper_relu_neuro.txt"; 

    try {
        std::cout << "Cargando modelo SNIPER (4 Neuronas) desde: " << modelPath << "...\n";
        agent.load(modelPath);
        std::cout << "Modelo cargado con exito.\n";
    } catch (const std::exception& e) {
        std::cerr << "Error cargando el modelo: " << e.what() << "\n";
        return 1;
    }

    // --- BUCLE DE JUEGO ---
    ale.reset_game();
    double totalReward = 0.0;
    int frames = 0;

    std::cout << "Iniciando partida DEMO con DOBLE CEREBRO..." << std::endl;

    while (!ale.game_over()) {
        frames++;

        // A. Obtener estado NORMALIZADO (-0.5 a 0.5)
        // (Debe coincidir con la normalización del entrenamiento)
        std::vector<double> state;
        const auto& RAM = ale.getRAM();
        state.reserve(ramImportant.size());
        for (int idx : ramImportant) {
            state.push_back((static_cast<double>(RAM.get(idx)) / 255.0) - 0.5);
        }

        // B. EL AGENTE PIENSA (Forward Pass)
        std::vector<double> output = agent.predict(state);

        // =========================================================
        // C. LÓGICA DE "DOBLE CEREBRO" (Split Brain Decoder)
        // =========================================================

        // 1. CEREBRO MOTOR (Neuronas 0, 1, 2) -> Decide Movimiento
        // Buscamos cuál de las 3 primeras neuronas tiene el valor más alto
        auto maxIt = std::max_element(output.begin(), output.begin() + 3);
        int moveIdx = std::distance(output.begin(), maxIt);
        Action intendedMove = move_actions[moveIdx];

        // 2. CEREBRO GATILLO (Neurona 3) -> Decide Disparo
        // Si el valor es positivo (> 0.0), quiere disparar.
        // (Esto funciona bien con Linear Output y ReLU/Tanh)
        bool wantToShoot = output[3] > 0.0;

        // 3. FUSIÓN DE INTENCIONES
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

        // D. Actuar
        totalReward += ale.act(actionToExecute);
    }

    std::cout << "GAME OVER. Puntuación final: " << totalReward << std::endl;

    return 0;
}