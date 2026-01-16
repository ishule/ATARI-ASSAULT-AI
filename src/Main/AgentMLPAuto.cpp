#include "MLP.hpp"
#include <ale_interface.hpp>
#include <SDL/SDL.h>
#include <iostream>
#include <vector>

using namespace std;
using namespace ale;

// Mismas posiciones RAM que AgentManual.cpp
vector<int> ramImportant = {
    15, 47,48,49, 50,51,52, 20, 21, 23,24,25, 39, 71, 109, 113,
    16,18, 32,33,34,35,36,37, 44,46, 42, 60, 101,102, 106, 121,
    67,68, 79,80, 53,54,55,56,61,62, 65,69,70,72,74,85,87,91,92,
    104,105, 114,119,120,123,125,126
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

// Convertir salida MLP a acción (versión mejorada)
Action mlpOutputToAction(const vector<double>& output) {
    // Usar valores continuos en lugar de umbral
    double fire_prob = output[0];
    double left_prob = output[1];
    double right_prob = output[2];
    
    // Encontrar la acción dominante
    vector<pair<double, Action>> actions = {
        {fire_prob, PLAYER_A_UPFIRE},
        {left_prob, PLAYER_A_LEFT},
        {right_prob, PLAYER_A_RIGHT},
        {fire_prob * left_prob, PLAYER_A_LEFTFIRE},   // Combinación
        {fire_prob * right_prob, PLAYER_A_RIGHTFIRE}, // Combinación
        {1.0 - fire_prob - left_prob - right_prob, PLAYER_A_NOOP}
    };
    
    // Elegir acción con mayor probabilidad
    auto best = max_element(actions.begin(), actions.end());
    
    return best->second;
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
    MLP mlp({61, 128, 64, 3}); // Arquitectura debe coincidir
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
    
    while (episode < MAX_EPISODES) {
        ale.reset_game();
        int totalReward = 0;
        int steps = 0;
        
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
        episode++;
    }
    
    cout << "\n✓ Finalizado\n";
    return 0;
}