#include <iostream>
#include <cmath>
#include <cstdint>
#include "src/ale_interface.hpp"

// Constants
constexpr uint32_t maxSteps = 7500;

///////////////////////////////////////////////////////////////////////////////
/// Get info from RAM
///////////////////////////////////////////////////////////////////////////////
int32_t getPlayerX(ALEInterface& alei) {
   return alei.getRAM().get(72) + ((rand() % 3) - 1);
}

int32_t getBallX(ALEInterface& alei) {
   return alei.getRAM().get(99) + ((rand() % 3) - 1);
}

///////////////////////////////////////////////////////////////////////////////
/// Do Next Agent Step
///////////////////////////////////////////////////////////////////////////////
reward_t agentStep(ALEInterface& alei) {
   static constexpr int32_t wide { 9 };
   static int32_t lives { alei.lives() };
   reward_t reward{0};

   // When we loose a live, we need to press FIRE to start again
   if (alei.lives() < lives) {
      lives = alei.lives();
      alei.act(PLAYER_A_FIRE);
   }

   // Apply rules.
   auto playerX { getPlayerX(alei) };
   auto ballX   { getBallX(alei)   };
   
   if       (ballX < playerX + wide) { reward = alei.act(PLAYER_A_LEFT);   }
   else if  (ballX > playerX + wide) { reward = alei.act(PLAYER_A_RIGHT);  }
   
   return reward + alei.act(PLAYER_A_NOOP);
}

///////////////////////////////////////////////////////////////////////////////
/// Print usage and exit
///////////////////////////////////////////////////////////////////////////////
void usage(char const* pname) {
   std::cerr
      << "\nUSAGE:\n" 
      << "   " << pname << " <romfile>\n";
   exit(-1);
}

///////////////////////////////////////////////////////////////////////////////
/// MAIN PROGRAM
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
   reward_t totalReward{};
   ALEInterface alei{};

   // Check input parameter
   if (argc != 2)
      usage(argv[0]);

   // Configure alei object.
   alei.setInt  ("random_seed", 0);
   alei.setFloat("repeat_action_probability", 0);
   alei.setBool ("display_screen", true);
   alei.setBool ("sound", false);
   alei.loadROM (argv[1]);

   // Init
   std::srand(static_cast<uint32_t>(std::time(0)));

   // Main loop
   {
      alei.act(PLAYER_A_FIRE);
      uint32_t step{};
      while ( !alei.game_over() && step < maxSteps ) { 
         totalReward += agentStep(alei);
         ++step;
      }

      std::cout << "Steps: " << step << std::endl;
      std::cout << "Reward: " << totalReward << std::endl;
   }

   return 0;
}
