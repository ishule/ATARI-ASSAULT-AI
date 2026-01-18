#include "GeneticAlgorithm/GeneticAlgorithm.hpp"
#include "GeneticAlgorithm/Individual.hpp"
#include "ale_interface.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

// RAM crítica para Assault
vector<int> ramImportant = { 
    0x00,
    0x01,
    0x02,
    0x09, 
    0x0A, 
    0x0B, 
    0x10,
    0x11, 
    0x12, 
    0x13, 
    0x15,
    0x16, 
    0x17, 
    0x18, 
    0x19, 
    0x1A, 
    0x1B, 
    0x1C, 
    0x1D, 
    0x1E, 
    0x1F, 
    0x21, 
    0x22, 
    0x23, 
    0x24, 
    0x25, 
    0x26, 
    0x27, 
    0x28, 
    0x2B, 
    0x2C, 
    0x2D, 
    0x2E, 
    0x2F, 
    0x30, 
    0x31, 
    0x32, 
    0x33, 
    0x34, 
    0x35, 
    0x36, 
    0x37, 
    0x38, 
    0x39, 
    0x3C, 
    0x3E,
    0x3F, 
    0x42, 
    0x43, 
    0x44, 
    0x45, 
    0x46, 
    0x47, 
    0x48, 
    0x49, 
    0x4B, 
    0x50, 
    0x51, 
    0x56, 
    0x58, 
    0x5A, 
    0x5C, 
    0x5D, 
    0x65,
    0x68, 
    0x69, 
    0x6A, 
    0x6C, 
    0x6E, 
    0x6F, 
    0x72, 
    0x73, 
    0x78, 
    0x79, 
    0x7A, 
    0x7B, 
    0x7C, 
    0x7D, 
    0x7E, 
    0x7F,
};

int main() {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // ========================================
    // 1. SETUP
    // ========================================
    int inputSize = ramImportant.size();
    
    ActionVect legal_actions = {
        PLAYER_A_NOOP,
        PLAYER_A_UPFIRE,
        PLAYER_A_RIGHT,
        PLAYER_A_LEFT,
        PLAYER_A_RIGHTFIRE,
        PLAYER_A_LEFTFIRE
    };
    int outputSize = legal_actions.size();
    
    std::cout << "\n════════════════════════════════════════\n";
    std::cout << "  ASSAULT - NEUROEVOLUTION TRAINER\n";
    std::cout << "════════════════════════════════════════\n\n";
    std::cout << "Inputs:  " << inputSize << "\n";
    std::cout << "Outputs: " << outputSize << "\n\n";

    // ========================================
    // 2. CONFIGURACIÓN GA (CON NEUROEVOLUCIÓN)
    // ========================================
    GAConfig config;
    config.populationSize = 50;
    config.maxGenerations = 200;
    config.mutationRate = 0.2;
    config.eliteRatio = 0.10;
    config.targetFitness = 200000;
    config.selectionType = SelectionType::TOURNAMENT;
    config.tournamentSize = 3; // Un poco de presión para eliminar arquitecturas malas
    config.verbose = true;
    config.saveDir = "models/ga/";
    config.printEvery = 1;

    // ★ CAMBIO 1: Parámetros de Arquitectura Dinámica
    config.minHiddenLayers = 1;       // Mínimo 1 capa oculta
    config.maxHiddenLayers = 3;       // Puede crecer hasta 3 capas si lo necesita
    config.minNeuronsPerLayer = 40;   // Mínimo 10 neuronas (evita cerebros inútiles)
    config.maxNeuronsPerLayer = 80;   // Tope para no explotar la CPU
    config.archMutationRate = 0.1;   // ★ 5% de probabilidad de cambiar la estructura (añadir/quitar neurona/capa)

    // ★ CAMBIO 2: Constructor SIN Topología Fija
    // Al pasar solo input y output, la librería inicia poblaciones con tamaños aleatorios
    GeneticAlgorithm ga(config, inputSize, outputSize);

    std::cout << "Population:   " << config.populationSize << "\n";
    std::cout << "Generations:  " << config.maxGenerations << "\n";
    std::cout << "Arch Mutation:" << (config.archMutationRate * 100) << "% (Neuroevolucion Activa)\n";
    std::cout << "\n";

    // ========================================
    // 3. CREAR UNA SOLA INSTANCIA DE ALE
    // ========================================
    ALEInterface ale;
    ale.setBool("display_screen", false);
    ale.setBool("sound", false);
    ale.setInt("frame_skip", 3);
    ale.loadROM("supported/assault.bin");
    
    std::cout << "✓ ALE initialized\n";
    std::cout << "✓ Starting evolution...\n\n";

    // ========================================
    // 4. ENTRENAMIENTO
    // ========================================
    Individual best = ga.evolveWithCustomFitness([&](Individual& ind) {
        
        // SOLO 1 SEMILLA (Velocidad x3)
        std::vector<int> seeds = {123}; 
        double totalFitness = 0.0;
        
        for (int seed : seeds) {
            ale.setInt("random_seed", seed);
            ale.reset_game();

            int framesAlive = 0;
            int movementFrames = 0;
            int shotsFired = 0;
            int maxFrames = 2500; // 2500 es suficiente
            double gameScore = 0.0;
            
            while (!ale.game_over() && framesAlive < maxFrames) {
                framesAlive++;
                
                std::vector<double> state;
                const auto& RAM = ale.getRAM();
                state.reserve(ramImportant.size());
                for (int idx : ramImportant) {
                    state.push_back(static_cast<double>(RAM.get(idx)) / 255.0);
                }

                std::vector<double> output = ind.predict(state);
                
                // --- LÓGICA DE ACCIÓN SEGURA ---
                int actionIdx = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
                if (actionIdx < 0 || actionIdx >= static_cast<int>(legal_actions.size())) actionIdx = 0;
                Action action = legal_actions[actionIdx];
                
                if (action == PLAYER_A_RIGHT || action == PLAYER_A_LEFT || action == PLAYER_A_RIGHTFIRE || action == PLAYER_A_LEFTFIRE) movementFrames++;
                if (action == PLAYER_A_UPFIRE || action == PLAYER_A_RIGHTFIRE || action == PLAYER_A_LEFTFIRE) shotsFired++;
                
                gameScore += ale.act(action);
            }
            
            // =========================================================
            // FITNESS "HAMBRE CANINA" (O matas o mueres)
            // =========================================================
            
            // 1. PUNTOS (Lo único que importa)
            // 1 Kill = 1000 Puntos
            double scorePoints = gameScore * 100.0; 
            
            // 2. SUPERVIVENCIA (CASI CERO)
            // Antes dabas 100. Ahora damos un máximo de 5 puntos.
            // Solo sirve para desempatar entre dos que tienen 0 kills.
            double survivalBonus = framesAlive / 500.0; 
            
            // 3. MOVIMIENTO Y DISPARO (SIMBÓLICO)
            // Antes dabas 50. Ahora damos 5.
            // Solo para que no se queden totalmente estáticos.
            double movementRatio = static_cast<double>(movementFrames) / (framesAlive + 1);
            double movementBonus = (movementRatio > 0.3) ? 5.0 : 0.0;

            double shotRatio = static_cast<double>(shotsFired) / (framesAlive + 1);
            double shotBonus = (shotRatio > 0.2) ? 5.0 : 0.0;

            // RESULTADO:
            // Si baila y sobrevive sin matar: Fitness ~15 (Antes era 200)
            // Si mata a UNO y muere al instante: Fitness ~1000
            // LA IA ELEGIRÁ ARRIESGARSE.
            
            totalFitness += (scorePoints + survivalBonus + movementBonus + shotBonus);
        }
        
        return totalFitness; // Sin dividir porque es 1 seed
    });
    // ========================================
    // 5. RESULTADOS
    // ========================================
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
    
    std::cout << "\n════════════════════════════════════════\n";
    std::cout << "       TRAINING COMPLETED\n";
    std::cout << "════════════════════════════════════════\n\n";
    std::cout << "Best fitness:  " << best.getFitness() << "\n";
    std::cout << "Architecture:  " << best.getArchitectureString() << " (Evolved)\n";
    std::cout << "Parameters:    " << best.getTotalParameters() << "\n";
    std::cout << "Time:          " << duration.count() << " seconds\n\n";
    
    best.save("models/assault_neuro_final.txt");
    std::cout << "✓ Model saved: models/assault_neuro_final.txt\n\n";
    
    // ========================================
    // 6. DEMO VISUAL (3 partidas)
    // ========================================
    ALEInterface ale_visual;
    ale_visual.setBool("display_screen", true);
    ale_visual.setBool("sound", true);
    ale_visual.setInt("frame_skip", 0);
    ale_visual.loadROM("supported/assault.bin");
    
    for (int test = 0; test < 3; ++test) {
        ale_visual.setInt("random_seed", 1000 + test);
        ale_visual.reset_game();
        
        int frames = 0;
        double finalScore = 0.0;
        
        std::cout << "Game " << (test + 1) << ": ";
        while (!ale_visual.game_over() && frames < 5000) {
            frames++;
            std::vector<double> state;
            const auto& RAM = ale_visual.getRAM();
            for (int idx : ramImportant) state.push_back(static_cast<double>(RAM.get(idx)) / 255.0);
            
            std::vector<double> output = best.predict(state);
            int actionIdx = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
            if (actionIdx >= 0 && actionIdx < static_cast<int>(legal_actions.size())) {
                finalScore += ale_visual.act(legal_actions[actionIdx]);
            }
        }
        std::cout << "Score=" << finalScore << "\n";
    }
    
    return 0;
}