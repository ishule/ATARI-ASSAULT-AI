#include "MLP.hpp"
#include <ale_interface.hpp>
#include <SDL/SDL.h>
#include <iostream>
#include <vector>

using namespace std;
using namespace ale;

// Mismas posiciones RAM que AgentManual.cpp
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



// Extraer features de RAM
vector<double> extractFeatures(ALEInterface& ale) {
    const auto& RAM = ale.getRAM();
    vector<double> features;
    
    for (int idx : ramImportant) {
        features.push_back(static_cast<double>(RAM.get(idx)) / 255.0);
    }
    
    return features;
}

enum ActionIndex {
    ACTION_NOOP = 0,
    ACTION_FIRE = 1,
    ACTION_LEFT = 2,
    ACTION_RIGHT = 3,
    ACTION_LEFTFIRE = 4,
    ACTION_RIGHTFIRE = 5
};

Action mlpOutputToAction(const vector<double>& output) {
    double fire = output[0];
    double left = output[1];
    double right = output[2];
    
    // ✅ Umbral adaptativo
    const double FIRE_THRESHOLD = 0.4;
    const double MOVE_THRESHOLD = 0.5;
    
    bool should_fire = (fire > FIRE_THRESHOLD);
    
    // ✅ Decidir movimiento (LEFT vs RIGHT vs NOOP)
    double move_diff = abs(left - right);
    
    Action movement = PLAYER_A_NOOP;
    
    if (move_diff > 0.1) {  // Diferencia significativa
        if (left > right && left > MOVE_THRESHOLD) {
            movement = PLAYER_A_LEFT;
        } else if (right > left && right > MOVE_THRESHOLD) {
            movement = PLAYER_A_RIGHT;
        }
    }
    
    // ✅ Combinar FIRE + movimiento
    if (should_fire) {
        if (movement == PLAYER_A_LEFT) return PLAYER_A_LEFTFIRE;
        if (movement == PLAYER_A_RIGHT) return PLAYER_A_RIGHTFIRE;
        return PLAYER_A_UPFIRE;
    }
    
    return movement;  // LEFT, RIGHT o NOOP
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Uso: " << argv[0] << " <ROM> [modelo.txt]\n";
        return -1;
    }
    
    string romPath = argv[1];
    string modelPath = (argc > 2) ? argv[2] : "models/atari_mlp_manual.txt";
    
    // Cargar modelo MLP
    cout << "Cargando modelo: " << modelPath << "\n";
    MLP mlp({80, 128, 64, 3}); // Arquitectura debe coincidir
    mlp.load(modelPath);
    cout << "✓ Modelo cargado\n\n";
    
    // Configurar ALE
    ALEInterface ale;
    ale.setBool("display_screen", true);
    ale.setBool("sound", true);
    ale.setInt("frame_skip", 4);
    ale.loadROM(romPath);
    
    cout << "=== JUGANDO CON MLP ENTRENADO ===\n\n";
    
    int episode = 0;
    const int MAX_EPISODES = 10;
    int sumReward = 0;

    
    while (episode < MAX_EPISODES) {
        ale.reset_game();
        int steps = 0;
        int totalReward = 0;

        cout << "Episodio " << (episode + 1) << "/" << MAX_EPISODES << " - ";
        
        while (!ale.game_over() && steps < 10000) {
            // 1. Extraer features de RAM
            vector<double> features = extractFeatures(ale);
            
            // 2. MLP predice acciones
            vector<double> output = mlp.predict(features);
            
            // 3. Convertir a acción de Atari
            Action action = mlpOutputToAction(output);
            
            // 4. Ejecutar acción
            int reward = ale.act(action);
            totalReward += reward;
            steps++;
            
            SDL_Delay(10); // Ajustar velocidad
        }
        
        cout << "Score: " << totalReward << " (steps: " << steps << ")\n";
        sumReward += totalReward;
        episode++;
    }
    
    cout << "Media: " << (sumReward / MAX_EPISODES) << "\n";
    cout << "\n✓ Finalizado\n";
    return 0;
}