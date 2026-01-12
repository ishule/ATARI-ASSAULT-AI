#include "Utils/Balance.hpp"
#include <algorithm>
#include <random>
#include <stdexcept>
#include <iostream>
#include <map>

// Método para sobre-muestrear clases con menor frecuencia
void Balance::oversample(std::vector<std::vector<double>> &X, std::vector<std::vector<double>> &Y) {
    // Crear un mapa para contar cuántas instancias tiene cada clase
    std::map<std::vector<double>, int> classCounts;
    for (const auto &label : Y) {
        classCounts[label]++;
    }

    // Encontrar la clase mayoritaria
    int maxCount = 0;
    for (const auto &[label, count] : classCounts) {
        maxCount = std::max(maxCount, count);
    }

    // Sobremuestrear las clases minoritarias
    std::vector<std::vector<double>> augmentedX;
    std::vector<std::vector<double>> augmentedY;

    for (const auto &[label, count] : classCounts) {
        std::vector<std::vector<double>> currentClassX;
        auto it = std::find_if(Y.begin(), Y.end(), [&](const std::vector<double> &y) { return y == label; });

        while (it != Y.end()) {
            currentClassX.push_back(X[std::distance(Y.begin(), it)]);
            it = std::find_if(it + 1, Y.end(), [&](const std::vector<double> &y) { return y == label; });
        }

        // Replicar hasta que todas las clases tengan el mismo número de instancias
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, currentClassX.size() - 1);

        while (currentClassX.size() < maxCount) {
            currentClassX.push_back(currentClassX[dist(rng)]);
        }

        for (const auto &row : currentClassX) {
            augmentedX.push_back(row);
            augmentedY.push_back(label);
        }
    }

    X = augmentedX;
    Y = augmentedY;
}