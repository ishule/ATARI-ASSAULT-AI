#include "GeneticAlgorithm/GeneticAlgorithm.hpp"
#include "GeneticAlgorithm/Individual.hpp"
#include "ale_interface.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath> 

vector<int> ramImportant = { 
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
    auto startTime = std::chrono::high_resolution_clock::now();
    int inputSize = ramImportant.size();
    
    // Las acciones "base" siguen siendo de movimiento
    ActionVect move_actions = {
        PLAYER_A_NOOP,   
        PLAYER_A_LEFT,   
        PLAYER_A_RIGHT   
    };
    
    // [0, 1, 2] -> Deciden Movimiento (Gana el más alto)
    // [3]       -> Decide Gatillo (Si > 0.0 dispara)
    int outputSize = 4; 
    
    GAConfig config;
    config.populationSize = 100;     
    config.maxGenerations = 150;     
    config.mutationRate = 0.2;
    config.eliteRatio = 0.10;
    config.targetFitness = 500000;   
    config.selectionType = SelectionType::TOURNAMENT;
    config.tournamentSize = 5;       
    config.verbose = true;
    config.saveDir = "models/ga/";
    config.printEvery = 1;

    config.minHiddenLayers = 1;
    config.maxHiddenLayers = 2;
    config.minNeuronsPerLayer = 40;
    config.maxNeuronsPerLayer = 80;   
    config.archMutationRate = 0.05;

    GeneticAlgorithm ga(config, inputSize, outputSize, ActivationType::TANH);

    ALEInterface ale;
    ale.setBool("display_screen", false); 
    ale.setBool("sound", false);
    
    ale.setInt("frame_skip", 1); 
    
    ale.loadROM("supported/assault.bin");
    
    cout << "✓ Starting evolution...\n\n";

    Individual best = ga.evolveWithCustomFitness([&](Individual& ind) {
        
        vector<int> seeds = {123}; 
        double totalFitness = 0.0;
        
        for (int seed : seeds) {
            ale.setInt("random_seed", seed);
            ale.reset_game();

            int framesAlive = 0;
            int leftMoves = 0;
            int rightMoves = 0;
            double gameScore = 0.0;
            int maxFrames = 6000; 
            
            while (!ale.game_over() && framesAlive < maxFrames) {
                framesAlive++;
                
                vector<double> state;
                const auto& RAM = ale.getRAM();
                state.reserve(ramImportant.size());
                for (int idx : ramImportant) {
                    state.push_back((static_cast<double>(RAM.get(idx)) / 255.0) - 0.5);
                }

                vector<double> output = ind.predict(state);

                auto maxIt = max_element(output.begin(), output.begin() + 3);
                int moveIdx = distance(output.begin(), maxIt);
                Action intendedMove = move_actions[moveIdx];

                bool wantToShoot = output[3] > 0.0;

                Action actionToExecute;

                if (wantToShoot) {
                    // La IA quiere disparar. Combinamos con el movimiento.
                    if (intendedMove == PLAYER_A_LEFT) actionToExecute = PLAYER_A_LEFTFIRE;
                    else if (intendedMove == PLAYER_A_RIGHT) actionToExecute = PLAYER_A_RIGHTFIRE;
                    else actionToExecute = PLAYER_A_UPFIRE;
                } else {
                    // La IA NO quiere disparar (está enfriando o no hay enemigos).
                    // Solo nos movemos.
                    actionToExecute = intendedMove;
                }
                
                // Estadísticas Anti-Camping
                if (intendedMove == PLAYER_A_LEFT) leftMoves++;
                if (intendedMove == PLAYER_A_RIGHT) rightMoves++;
                
                gameScore += ale.act(actionToExecute);
            }
            
            // FITNESS
            double scorePoints = gameScore; 
            double minSide = (double)min(leftMoves, rightMoves);
            double maxSide = (double)max(leftMoves, rightMoves);
            if (maxSide == 0) maxSide = 1.0; 
            double balanceRatio = minSide / maxSide;
            
            // Castigo por campear
            double penaltyFactor = (balanceRatio < 0.15) ? 0.2 : 1.0; 
            
            // Bono por explorar
            double explorationBonus = min(minSide , 100.0)*5;

            totalFitness += (scorePoints * penaltyFactor) + explorationBonus;
        }
        
        return totalFitness; 
    });

    // RESULTADOS
    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
    
    cout << "\nTRAINING COMPLETED\n";
    cout << "Best fitness: " << best.getFitness() << "\n";
    best.save("models/assault_sniper_tanh_neuro.txt");

    // DEMO VISUAL
    ALEInterface ale_visual;
    ale_visual.setBool("display_screen", true); 
    ale_visual.setBool("sound", true);
    ale_visual.setInt("frame_skip", 1); 
    ale_visual.loadROM("supported/assault.bin");
    
    for (int test = 0; test < 3; ++test) {
        ale_visual.setInt("random_seed", 1000 + test);
        ale_visual.reset_game();
        int f = 0; double fs = 0.0;
        
        cout << "Demo Game " << (test + 1) << " running...\n";
        while (!ale_visual.game_over() && f < 10000) {
            f++;
            vector<double> state;
            const auto& RAM = ale_visual.getRAM();
            for (int idx : ramImportant) state.push_back((static_cast<double>(RAM.get(idx)) / 255.0) - 0.5);
            
            vector<double> output = best.predict(state);

            // DECODIFICACIÓN EN DEMO (IGUAL QUE TRAINING)
            auto maxIt = std::max_element(output.begin(), output.begin() + 3);
            int moveIdx = std::distance(output.begin(), maxIt);
            Action intendedMove = move_actions[moveIdx];
            bool wantToShoot = output[3] > 0.0;
            
            Action actionToExecute;
            if (wantToShoot) {
                if (intendedMove == PLAYER_A_LEFT) actionToExecute = PLAYER_A_LEFTFIRE;
                else if (intendedMove == PLAYER_A_RIGHT) actionToExecute = PLAYER_A_RIGHTFIRE;
                else actionToExecute = PLAYER_A_UPFIRE;
            } else {
                actionToExecute = intendedMove;
            }
            
            fs += ale_visual.act(actionToExecute);
        }
        cout << "Final Score: " << fs << "\n";
    }
    return 0;
}