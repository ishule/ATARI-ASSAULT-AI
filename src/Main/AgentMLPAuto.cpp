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

Action mlpOutputToAction_v0(const vector<double>& output) {
    double fire = output[0];
    double left = output[1];
    double right = output[2];
    
    // Umbral adaptativo de prueba y error
    const double FIRE_THRESHOLD = 0.4;
    const double MOVE_THRESHOLD = 0.5;
    
    bool should_fire = (fire > FIRE_THRESHOLD);
    
    // Decidir movimiento (LEFT vs RIGHT vs NOOP)
    double move_diff = abs(left - right);
    
    Action movement = PLAYER_A_NOOP;
    
    if (move_diff > 0.1) { 
        if (left > right && left > MOVE_THRESHOLD) {
            movement = PLAYER_A_LEFT;
        } else if (right > left && right > MOVE_THRESHOLD) {
            movement = PLAYER_A_RIGHT;
        }
    }
    
    // Combinar FIRE + movimiento
    if (should_fire) {
        if (movement == PLAYER_A_LEFT) return PLAYER_A_LEFTFIRE;
        if (movement == PLAYER_A_RIGHT) return PLAYER_A_RIGHTFIRE;
        return PLAYER_A_UPFIRE;
    }
    
    return movement;  // LEFT, RIGHT o NOOP
}


Action mlpOutputToAction_v1(const vector<double>& output) {
    double fire = output[0];
    double left = output[1];
    double right = output[2];
    
    // 1. Decisión de disparo con umbral dinámico
    bool should_fire = (fire > 0.3); // Umbral más bajo para disparar más
    
    // 2. Movimiento basado en máximo directo (sin diferencia)
    Action movement = PLAYER_A_NOOP;
    
    double max_move = max(left, right);
    if (max_move > 0.35) { // Umbral más bajo para moverse más
        movement = (left > right) ? PLAYER_A_LEFT : PLAYER_A_RIGHT;
    }
    
    // 3. Combinar
    if (should_fire) {
        if (movement == PLAYER_A_LEFT) return PLAYER_A_LEFTFIRE;
        if (movement == PLAYER_A_RIGHT) return PLAYER_A_RIGHTFIRE;
        return PLAYER_A_UPFIRE;
    }
    
    return movement;
}


Action mlpOutputToAction_v2(const vector<double>& output) {
    // Mapeo directo: la acción con mayor activación gana
    vector<pair<double, Action>> acciones = {
        {output[0] * 1.2, PLAYER_A_UPFIRE},      // Bonus al disparo
        {output[1], PLAYER_A_LEFT},
        {output[2], PLAYER_A_RIGHT},
        {output[1] * output[0], PLAYER_A_LEFTFIRE},   // Combinación
        {output[2] * output[0], PLAYER_A_RIGHTFIRE},  // Combinación
        {0.1, PLAYER_A_NOOP}  // Baseline para NOOP
    };
    
    // Seleccionar la acción con mayor valor
    auto max_action = max_element(acciones.begin(), acciones.end());
    return max_action->second;
}

Action mlpOutputToAction_v3(const vector<double>& output, int gameScore) {
    double fire = output[0];
    double left = output[1];
    double right = output[2];
    
    // Ajustar agresividad según el score (inicio más agresivo)
    double fire_threshold = (gameScore < 500) ? 0.25 : 0.4;
    double move_threshold = (gameScore < 500) ? 0.3 : 0.5;
    
    bool should_fire = (fire > fire_threshold);
    
    Action movement = PLAYER_A_NOOP;
    if (left > move_threshold || right > move_threshold) {
        movement = (left > right) ? PLAYER_A_LEFT : PLAYER_A_RIGHT;
    }
    
    if (should_fire) {
        if (movement == PLAYER_A_LEFT) return PLAYER_A_LEFTFIRE;
        if (movement == PLAYER_A_RIGHT) return PLAYER_A_RIGHTFIRE;
        return PLAYER_A_UPFIRE;
    }
    
    return movement;
}

// v4: FIRE=0.35, MOVE=0.45, diff=0.08
// v5: FIRE=0.38, MOVE=0.48, diff=0.12
// v6: FIRE=0.42, MOVE=0.52, diff=0.09

Action mlpOutputToAction_v4_v5_v6(const vector<double>& output) {
    double fire = output[0];
    double left = output[1];
    double right = output[2];
    
    // VARIANTE 1: Umbrales ligeramente más bajos
    const double FIRE_THRESHOLD = 0.42;    // antes 0.4
    const double MOVE_THRESHOLD = 0.52;    // antes 0.5
    
    bool should_fire = (fire > FIRE_THRESHOLD);
    
    double move_diff = abs(left - right);
    Action movement = PLAYER_A_NOOP;
    
    if (move_diff > 0.09) {  // antes 0.1, más sensible
        if (left > right && left > MOVE_THRESHOLD) {
            movement = PLAYER_A_LEFT;
        } else if (right > left && right > MOVE_THRESHOLD) {
            movement = PLAYER_A_RIGHT;
        }
    }
    
    if (should_fire) {
        if (movement == PLAYER_A_LEFT) return PLAYER_A_LEFTFIRE;
        if (movement == PLAYER_A_RIGHT) return PLAYER_A_RIGHTFIRE;
        return PLAYER_A_UPFIRE;
    }
    
    return movement;
}

Action mlpOutputToAction_v7(const vector<double>& output) {
    double fire = output[0];
    double left = output[1];
    double right = output[2];
    
    // Calcular "confianza" de cada acción
    double fire_confidence = fire;
    double left_confidence = left * (1.0 - right);  // Penaliza si right también es alto
    double right_confidence = right * (1.0 - left);
    
    // Decisión de disparo más conservadora
    bool should_fire = (fire_confidence > 0.42);
    
    // Movimiento por confianza relativa
    Action movement = PLAYER_A_NOOP;
    double max_move_conf = max(left_confidence, right_confidence);
    
    if (max_move_conf > 0.4) {
        movement = (left_confidence > right_confidence) ? PLAYER_A_LEFT : PLAYER_A_RIGHT;
    }
    
    if (should_fire) {
        if (movement == PLAYER_A_LEFT) return PLAYER_A_LEFTFIRE;
        if (movement == PLAYER_A_RIGHT) return PLAYER_A_RIGHTFIRE;
        return PLAYER_A_UPFIRE;
    }
    
    return movement;
}

// Variable global o miembro de clase
static Action last_action = PLAYER_A_NOOP;
static int action_persistence = 0;

Action mlpOutputToAction(const vector<double>& output) {
    double fire = output[0];
    double left = output[1];
    double right = output[2];
    
    const double FIRE_THRESHOLD = 0.4;
    const double MOVE_THRESHOLD = 0.5;
    
    bool should_fire = (fire > FIRE_THRESHOLD);
    
    Action movement = PLAYER_A_NOOP;
    double move_diff = abs(left - right);
    
    if (move_diff > 0.1) {
        if (left > right && left > MOVE_THRESHOLD) {
            movement = PLAYER_A_LEFT;
        } else if (right > left && right > MOVE_THRESHOLD) {
            movement = PLAYER_A_RIGHT;
        }
    }
    
    Action new_action = movement;
    if (should_fire) {
        if (movement == PLAYER_A_LEFT) new_action = PLAYER_A_LEFTFIRE;
        else if (movement == PLAYER_A_RIGHT) new_action = PLAYER_A_RIGHTFIRE;
        else new_action = PLAYER_A_UPFIRE;
    }
    
    // Mantener acción anterior si es similar (reduce "nerviosismo")
    if (new_action == last_action) {
        action_persistence++;
    } else {
        if (action_persistence < 3) {  // Mínimo 3 frames antes de cambiar
            new_action = last_action;
            action_persistence++;
        } else {
            action_persistence = 0;
        }
    }
    
    last_action = new_action;
    return new_action;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Uso: " << argv[0] << " <ROM> [modelo.txt]\n";
        return -1;
    }
    
    string romPath = argv[1];
    string modelPath = (argc > 2) ? argv[2] : "models/atari_mlp_manual.txt";
    
    // Cargar modelo MLP
    std::ifstream file(modelPath);
    
    int numCapas;
    file >> numCapas;
    
    std::vector<int> arquitectura(numCapas);
    for (int i = 0; i < numCapas; ++i) {
        file >> arquitectura[i];
    }
    
    MLP mlp(arquitectura);


    mlp.load(modelPath);
    cout << "Modelo cargado\n\n";
    
    // Configurar ALE
    ALEInterface ale;
    ale.setBool("display_screen", true);
    ale.setBool("sound", true);
    ale.setInt("frame_skip", 4);
    ale.loadROM(romPath);
    
    cout << "=== JUGANDO CON MLP ENTRENADO ===\n\n";
    
    int episode = 0;
    const int MAX_EPISODES = 50;
    int sumReward = 0;

    
    while (episode < MAX_EPISODES) {
        ale.reset_game();
        int steps = 0;
        int totalReward = 0;

        cout << "Test " << (episode + 1) << "/" << MAX_EPISODES << " - ";
        
        while (!ale.game_over() && steps < 5000) {
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
    cout << "\nFinalizado\n";
    return 0;
}