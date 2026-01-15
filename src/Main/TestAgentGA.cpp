#include "GeneticAlgorithm/Individual.hpp"
#include "ale/ale_interface.hpp"
#include <iostream>
#include <vector>
#include <algorithm> // Para max_element
//ARCHIVO PARA PROBAR LA IA ENTRENADA CON GENETIC ALGORITHM
// La misma RAM que usamos para entrenar
const std::vector<int> ramImportant = {
    15,47,48,49,50,51,52,20,21,23,24,25,39,71,109,113,
    16,18,32,33,34,35,36,37,44,46,42,60,101,102,106,121,67,68,79,80,
    53,54,55,56,61,62,65,69,70,72,74,85,87,91,92,104,105,114,119,120,123,125,126
};

int main() {
    ALEInterface ale;
    
    // === CONFIGURACIÓN VISUAL ===
    ale.setBool("display_screen", true);  // ¡AQUÍ SÍ VEMOS EL JUEGO!
    ale.setBool("sound", true);           // Con sonido
    ale.setInt("random_seed", 123);
    ale.setInt("frame_skip", 0);          // Ver cada frame
    
    ale.loadROM("supported/assault.bin");

    // 1. Creamos un individuo vacío (placeholder)
    // Le ponemos una topología dummy {1} porque 'load' la va a sobreescribir
    Individual agent({1}); 

    // 2. Cargamos el cerebro entrenado por el GA
    std::string modelPath = "models/assault_agent_genetic.txt"; // MO
    try {
        std::cout << "Cargando modelo desde: " << modelPath << "...\n";
        agent.load(modelPath);
        std::cout << "Modelo cargado correctamente.\n";
    } catch (const std::exception& e) {
        std::cerr << "Error cargando el modelo: " << e.what() << "\n";
        std::cerr << "Asegúrate de ejecutar primero AgentGA para generar el archivo.\n";
        return 1;
    }

    // 3. Bucle de juego
    ale.reset_game();
    double totalReward = 0.0;

    std::cout << "Iniciando partida..." << std::endl;

    while (!ale.game_over()) {
        // A. Obtener estado (RAM)
        std::vector<double> state;
        const auto& RAM = ale.getRAM();
        state.reserve(ramImportant.size());
        for (int idx : ramImportant) {
            state.push_back(static_cast<double>(RAM[idx]) / 255.0);
        }

        // B. El agente piensa
        std::vector<double> output = agent.predict(state);

        // C. Decidir acción (el valor más alto gana)
        int actionIdx = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
        Action action = ale.getLegalActionSet()[actionIdx];

        // D. Actuar
        totalReward += ale.act(action);
    }

    std::cout << "GAME OVER. Puntuación final: " << totalReward << std::endl;

    return 0;
}