#include "Normalize.hpp"
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <iostream>

// Método para calcular las medias y desviaciones estándar
void Normalize::fit(const std::vector<std::vector<double>> &data) {
    int numCols = data[0].size();
    means.resize(numCols, 0.0);
    stdDevs.resize(numCols, 0.0);

    // Calcular medias
    for (const auto &row : data) {
        for (int j = 0; j < numCols; ++j) {
            means[j] += row[j];
        }
    }
    for (double &mean : means) {
        mean /= data.size();
    }

    // Calcular desviaciones estándar
    for (const auto &row : data) {
        for (int j = 0; j < numCols; ++j) {
            stdDevs[j] += std::pow(row[j] - means[j], 2);
        }
    }
    for (double &stdDev : stdDevs) {
        stdDev = std::sqrt(stdDev / data.size());
        if (stdDev == 0) stdDev = 1;  // Evitar divisiones por 0
    }
}

// Método para transformar los datos con las medias y desviaciones calculadas
std::vector<std::vector<double>> Normalize::transform(const std::vector<std::vector<double>> &data) {
    std::vector<std::vector<double>> transformed = data;
    for (auto &row : transformed) {
        for (int j = 0; j < row.size(); ++j) {
            row[j] = (row[j] - means[j]) / stdDevs[j];
        }
    }
    return transformed;
}

// Guardar el escalador a un archivo
void Normalize::saveScaler(const std::string &filename) {
    std::ofstream file(filename);
    if (!file.is_open()) throw std::runtime_error("No se pudo abrir el archivo: " + filename);

    for (const auto &mean : means) file << mean << " ";
    file << "\n";
    for (const auto &stdDev : stdDevs) file << stdDev << " ";
    file.close();
}

// Cargar el escalador desde un archivo
void Normalize::loadScaler(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("No se pudo abrir el archivo: " + filename);

    means.clear();
    stdDevs.clear();
    double val;

    while (file >> val) means.push_back(val);
    for (auto &mean : means) {
        file >> val;
        stdDevs.push_back(val);
    }
    file.close();
}