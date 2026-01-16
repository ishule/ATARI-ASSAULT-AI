#include <iostream>
#include <ale_interface.hpp>
#include <SDL/SDL.h>
#include <fstream>
#include <vector>
#include <map>      // Para asociar etiquetas a posiciones RAM
#include <cstdio>   // Para printf y manipulaciones de consola
#include <iomanip>  // Para mejorar el formato de las salidas

using namespace std;
using namespace ale;

const int maxSteps = 15000;  // Número máximo de pasos por partida
const char splitSymbol = ';';  // Delimitador para los datos

// Mapeo de etiquetas con posiciones importantes de RAM.
map<u_int8_t, string> ramImportant = { 
    {0x00, "Score (byte alto)"},
    {0x01, "Score (byte medio)"},
    {0x02, "Score (byte bajo)"},
    {0x09, "Undefined"},
    {0x0A, "Undefined"},
    {0x0B, "Undefined"},
    {0x10, "Posición X del jugador"},
    {0x11, "Undefined"},
    {0x12, "Undefined"},
    {0x13, "Undefined"},
    {0x15, "Peligro barra disparo"},
    {0x16, "Undefined"},
    {0x17, "Undefined"},
    {0x18, "Undefined"},
    {0x19, "Undefined"},
    {0x1A, "Undefined"},
    {0x1B, "Undefined"},
    {0x1C, "Undefined"},
    {0x1D, "Undefined"},
    {0x1E, "Undefined"},
    {0x1F, "Undefined"},
    {0x1D, "Undefined"},
    {0x21, "Undefined"},
    {0x22, "Undefined"},
    {0x23, "Undefined"},
    {0x24, "Undefined"},
    {0x25, "Undefined"},
    {0x26, "Undefined"},
    {0x27, "Undefined"},
    {0x28, "Undefined"},
    {0x2B, "Undefined"},
    {0x2C, "Undefined"},
    {0x2D, "Undefined"},
    {0x2E, "Undefined"},
    {0x2F, "Undefined"},
    {0x30, "Undefined"},
    {0x31, "Undefined"},
    {0x32, "Undefined"},
    {0x33, "Undefined"},
    {0x34, "Undefined"},
    {0x35, "Undefined"},
    {0x36, "Undefined"},
    {0x37, "Undefined"},
    {0x38, "Undefined"},
    {0x39, "Undefined"},
    {0x3C, "Undefined"},
    {0x3E, "cooldown de disparo"},
    {0x3F, "undefined"},
    {0x42, "undefined"},
    {0x43, "undefined"},
    {0x44, "undefined"},
    {0x45, "undefined"},
    {0x46, "undefined"},
    {0x47, "undefined"},
    {0x48, "undefined"},
    {0x49, "undefined"},
    {0x4B, "undefined"},
    {0x50, "undefined"},
    {0x51, "undefined"},
    {0x56, "undefined"},
    {0x58, "undefined"},
    {0x5A, "undefined"},
    {0x5C, "undefined"},
    {0x5D, "undefined"},
    {0x65, "Vidas restantes"},
    {0x68, "undefined"},
    {0x69, "undefined"},
    {0x6A, "undefined"},
    {0x6C, "undefined"},
    {0x6E, "undefined"},
    {0x6F, "undefined"},
    {0x72, "undefined"},
    {0x73, "undefined"},
    {0x78, "undefined"},
    {0x79, "undefined"},
    {0x7A, "undefined"},
    {0x7B, "undefined"},
    {0x7C, "undefined"},
    {0x7D, "undefined"},
    {0x7E, "undefined"},
    {0x7F, "undefined"}
};

// Variables globales para el análisis RAM
vector<int> currentState(128, 0);  // Estado previo de RAM
vector<int> changes(128, 0);       // Conteo de cambios para cada posición
ALEInterface alei;

// Función: Asignación de color según mapa de calor
string getHeatmapColor(int value, int maxValue) {
    if (value == 0) return "\033[34m";    // Azul para valor 0
    float ratio = static_cast<float>(value) / maxValue;
    if (ratio < 0.33) return "\033[32m";  // Verde para valores bajos
    if (ratio < 0.66) return "\033[33m";  // Amarillo para valores medios
    return "\033[31m";                    // Rojo para valores altos
}

// Guardar estado en archivo CSV
void saveState(ofstream &file, ALEInterface &alei, const Uint8 *keystates) {
    const auto &RAM = alei.getRAM();

    for (const auto &[idx, label] : ramImportant) {
        int value = static_cast<int>(RAM.get(idx));
        file << value << splitSymbol;
    }

    file << static_cast<int>(keystates[SDLK_SPACE]) << splitSymbol;
    file << static_cast<int>(keystates[SDLK_LEFT]) << splitSymbol;
    file << static_cast<int>(keystates[SDLK_RIGHT]) << "\n";
}

// Mostrar estado actual de la RAM con colores del mapa de calor
void showRAM() {
    const auto &RAM = alei.getRAM();

    uint8_t add = 0;
    int maxChanges = *max_element(changes.begin(), changes.end()); // Max valor de cambios

    std::printf("\033[H"); // Mueve el cursor al inicio de la consola
    std::printf("\nEstado actual de la RAM:\n");
    std::printf("AD |  00   01   02   03   04   05   06   07   08   09   0A   0B   0C   0D   0E   0F");
    std::printf("\n====================================================================================");
    for (std::size_t i = 0; i < 8; i++) {
        std::printf("\n%02X | ", add);
        for (std::size_t j = 0; j < 16; j++) {
            int value = RAM.get(add);
            string color = getHeatmapColor(changes[add], maxChanges); // Color según cambios
            std::printf("%s %02X \033[0m", color.c_str(), value);    // Restablecer color
            cout << " ";
            add++;
        }
    }
    std::printf("\n");
}

// Comparar cambios en la RAM
void compareRAM() {
    const auto &RAM = alei.getRAM();

    uint8_t add = 0;
    for (std::size_t i = 0; i < 128; i++) {
        int currentValue = static_cast<int>(RAM.get(add));
        if (currentValue != currentState[add]) {
            changes[add]++;
        }
        currentState[add] = currentValue;
        add++;
    }
}

// Mostrar cambios acumulados en la RAM con colores tipo mapa de calor
void showChanges() {
    uint8_t add = 0;
    int maxChanges = *max_element(changes.begin(), changes.end()); // Max valor de cambios

    std::printf("\nCambios acumulados en la RAM:\n");
    std::printf("AD |  00   01   02   03   04   05   06   07   08   09   0A   0B   0C   0D   0E   0F");
    std::printf("\n=====================================================================================");
    for (std::size_t i = 0; i < 8; i++) {
        std::printf("\n%02X | ", add);
        for (std::size_t j = 0; j < 16; j++) {
            int val = changes[add];
            string color = getHeatmapColor(val, maxChanges); // Color según cambios acumulados
            std::printf("%s%4d\033[0m", color.c_str(), val); // Centrar con setw y reiniciar color
            cout << " ";
            add++;
        }
    }
    std::printf("\n");
}

// Mostrar las posiciones importantes de la RAM con etiquetas
void showImportantPositions() {
    const auto &RAM = alei.getRAM();

    std::cout << "Posiciones importantes en la RAM:\n";
    std::cout << "=================================\n";
    for (const auto &[idx, label] : ramImportant) {
        int value = RAM.get(idx);
        std::printf("RAM[%02X] (%s) = %02X\n", idx, label.c_str(), value); // Imprime etiqueta y valor
    }
    std::cout << "=================================\n";
}



int main(int argc, char **argv) {
    if (argc != 2) {
        cerr << "Uso: " << argv[0] << " <ROM del juego>\n";
        return -1;
    }

    // Configurar ALE
    alei.setBool("display_screen", true);
    alei.setBool("sound", true);
    alei.setInt("frame_skip", 0);  // Reducir la velocidad del juego para precisión
    alei.loadROM(argv[1]);

    // Limpiar terminal
    std::printf("\033[2J\033[H");

    ofstream file("data_manual.csv");

    // Loop Principal del Juego
    for (int step = 0; step < maxSteps && !alei.game_over(); ++step) {
        SDL_PumpEvents();
        Uint8 *keystates = SDL_GetKeyState(NULL);

        saveState(file, alei, keystates);
        compareRAM();  // Analizar y contar cambios en la RAM

        // Mostrar estado actual y cambios acumulados con etiquetas
        //showRAM();
        //showChanges();
        //showImportantPositions();

        // Ejecutar acción según las teclas presionadas
        Action action = PLAYER_A_NOOP;
        if (keystates[SDLK_SPACE]) {
            if (keystates[SDLK_LEFT]) {
                action = PLAYER_A_LEFTFIRE;
            } else if (keystates[SDLK_RIGHT]) {
                action = PLAYER_A_RIGHTFIRE;
            } else {
                action = PLAYER_A_UPFIRE;
            }
        } else {
            if (keystates[SDLK_LEFT]) {
                action = PLAYER_A_LEFT;
            } else if (keystates[SDLK_RIGHT]) {
                action = PLAYER_A_RIGHT;
            }
        }

        alei.act(action);  // Ejecuta la acción seleccionada
        SDL_Delay(10); // Pausar brevemente para el siguiente ciclo
    }

    file.close();
    cout << "Partida terminada. Datos guardados en data_manual.csv\n";

    return 0;
}