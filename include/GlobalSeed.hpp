// Global seed declaration
#ifndef GLOBAL_SEED_HPP
#define GLOBAL_SEED_HPP

#include <cstdint>

extern std::uint32_t GLOBAL_SEED;

// Set a new global seed at runtime
inline void setGlobalSeed(std::uint32_t s) { GLOBAL_SEED = s; }

#endif // GLOBAL_SEED_HPP
