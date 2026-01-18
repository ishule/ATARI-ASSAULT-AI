#include "GeneticAlgorithm/Individual.hpp"
#include "ale_interface.hpp"
#include <SDL/SDL.h>
#include <iostream>
#include <vector>
#include <algorithm>

// La misma RAM que usamos para entrenar
const std::vector<int> ramImportant = {
    15,47,48,49,50,51,52,20,21,23,24,25,39,71,109,113,
    16,18,32,33,34,35,36,37,44,46,42,60,101,102,106,121,67,68,79,80,
    53,54,55,56,61,62,65,69,70,72,74,85,87,91,92,104,105,114,119,120,123,125,126
};

int main() {
    ALEInterface ale;
    
    // === CONFIGURACIÓN VISUAL ===
    ale.setBool("display_screen", true); 
    ale.setBool("sound", true);           
    ale.setInt("random_seed", 123);
    ale.setInt("frame_skip", 4);          
    
    ale.loadROM("supported/assault.bin");

    // ⚠️ CORRECCIÓN 1: DEFINIR LAS MISMAS ACCIONES QUE EN EL TRAINER
    ActionVect legal_actions = {
        PLAYER_A_NOOP, 
        PLAYER_A_UPFIRE,    // Recuerda que cambiamos UPFIRE por FIRE
        PLAYER_A_RIGHT,     
        PLAYER_A_LEFT, 
        PLAYER_A_RIGHTFIRE, 
        PLAYER_A_LEFTFIRE
    };

    // 1. Creamos placeholder
    Individual agent({1}); 

    // 2. Cargamos el cerebro
    // Asegúrate de que el nombre del archivo coincide con el que guardaste
    std::string modelPath = "models/assault_neuro_final_02.txt"; 
    try {
        std::cout << "Cargando modelo desde: " << modelPath << "...\n";
        agent.load(modelPath);
        std::cout << "Modelo cargado. Arquitectura: " << agent.getArchitectureString() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    // 3. Bucle de juego
    ale.reset_game();
    double totalReward = 0.0;

    std::cout << "Iniciando partida..." << std::endl;

    while (!ale.game_over()) {
        // A. Obtener estado
        std::vector<double> state;
        const auto& RAM = ale.getRAM();
        state.reserve(ramImportant.size());
        
        // ⚠️ CORRECCIÓN 2: NORMALIZACIÓN IDENTICA AL ENTRENAMIENTO (-0.5)
        for (int idx : ramImportant) {
            state.push_back((static_cast<double>(RAM.get(idx)) / 255.0) - 0.5);
        }

        // B. El agente piensa
        std::vector<double> output = agent.predict(state);

        // C. Decidir acción usando NUESTRO vector reducido
        int actionIdx = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
        
        // Protección de rango (aunque la red debe tener el tamaño correcto)
        if(actionIdx >= 0 && actionIdx < legal_actions.size()) {
            Action action = legal_actions[actionIdx];
            totalReward += ale.act(action);
        } else {
            // Fallback por si acaso cargaste un modelo antiguo con más salidas
            ale.act(PLAYER_A_NOOP);
        }
    }

    std::cout << "GAME OVER. Puntuación final: " << totalReward << std::endl;

    return 0;
}