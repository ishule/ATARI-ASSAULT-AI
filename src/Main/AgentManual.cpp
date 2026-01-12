#include <iostream>
#include <ale_interface.hpp>
#include <SDL/SDL.h>
#include <fstream>
#include <vector>

using namespace std;
using namespace ale;

const int maxSteps = 15000;  // Número máximo de pasos por partida
const char splitSymbol = ';';  // Delimitador para los datos
vector<int> ramImportant = {//Muy importantes
   15, /*Posicion principal*/
   47,48,49, /*Velocidades enemigos*/
   50/*0x32*/,51,52, /*¿Velocidades enemigos?*/ 
   20/*Cambia cuando barra disparo entra en peligro*/,  

   //En duda
   21/*Cambia si hay enemigo lateral*/,
   23,24,25, /*Cambia si hay bala lateral*/     
   39, /*Tipo enemigo graficamente*/
   71,/*Cambia al pasar de ronda*/ 
   109, /*Posicion bala enemiga*/ 
   113/*0x71*/, /*Cambia periódicamente 40-80*/ 

   //No sabemos
   16,18, /*Cambia demasiado rapido*/
   32,33,34,35,36,37, /*Cambia demasiado rapido*/
   44,46, /*Cambian continuamente entre FC-FE, si mato a alguien se pone a FF*/
   42, /*¿No cambia?*/
   60, /*Cambia demasiado rapido*/
   101,102, /*Cambia demasiado rapido*/
   106, /*Cambia demasiado rapido*/
   121, /*Muy aleatorio*/
   67,68, /*Cambia demasiado rapido*/
   79,80, /*Cambia demasiado rapido*/

   //Por observar
   53,54,55,56/*0x38*/,61,62,
   65,69,70,72,74,85,87,91,92,
   104,105,
   114,119,120,123,125,126
};

ALEInterface alei;

// Guardar estado en archivo y depuración de RAM
void saveState(std::ofstream &file, ALEInterface &alei, Uint8 *keystates) {
    const auto &RAM = alei.getRAM();

    for (const auto &idx : ramImportant) {
        int value = static_cast<int>(RAM.get(idx));
        file << value << splitSymbol;
    }

    file << static_cast<int>(keystates[SDLK_SPACE]) << splitSymbol;
    file << static_cast<int>(keystates[SDLK_LEFT]) << splitSymbol;
    file << static_cast<int>(keystates[SDLK_RIGHT]) << "\n";
}

int main(int argc, char **argv) {
    if (argc != 2) {
        cerr << "Uso: " << argv[0] << " <ROM del juego>\n";
        return -1;
    }

    // Configurar ALE
    alei.setBool("display_screen", true);
    alei.setBool("sound", true);
    alei.setInt("frame_skip", 0);  // Reducir la velocidad del juego
    alei.loadROM(argv[1]);

    ofstream file("data_manual.csv");

    // Loop Principal del Juego
    for (int step = 0; step < maxSteps && !alei.game_over(); ++step) {
        SDL_PumpEvents();
        Uint8 *keystates = SDL_GetKeyState(NULL);

        saveState(file, alei, keystates);

        // Combinar acciones según las teclas presionadas
        Action action = PLAYER_A_NOOP;  // Acción predeterminada: ninguna

        if (keystates[SDLK_SPACE]) {
            if (keystates[SDLK_LEFT]) {
                action = PLAYER_A_LEFTFIRE;  // Disparar y moverse a la izquierda
            } else if (keystates[SDLK_RIGHT]) {
                action = PLAYER_A_RIGHTFIRE;  // Disparar y moverse a la derecha
            } else {
                action = PLAYER_A_UPFIRE;  // Disparar hacia arriba
            }
        } else {
            if (keystates[SDLK_LEFT]) {
                action = PLAYER_A_LEFT;  // Solo moverse a la izquierda
            } else if (keystates[SDLK_RIGHT]) {
                action = PLAYER_A_RIGHT;  // Solo moverse a la derecha
            }
        }

        alei.act(action);  // Ejecuta la acción
        SDL_Delay(10);     // Ajusta el retraso manual
    }

    file.close();
    cout << "Partida terminada. Datos guardados en data_manual.csv\n";

    return 0;
}