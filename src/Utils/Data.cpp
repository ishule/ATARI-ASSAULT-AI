#include "Data.hpp"
#include <fstream>
#include <sstream>

// Método para cargar datos desde un archivo CSV
void Data::loadCSV(const std::string &filename, int xCols, int yCols) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("No se pudo abrir el archivo: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> xRow, yRow;
        std::istringstream ss(line);
        std::string value;
        int col = 0;

        while (std::getline(ss, value, ';')) {
            double val = std::stod(value);
            if (col < xCols) {
                xRow.push_back(val);  // Valores para X
            } else {
                yRow.push_back(val);  // Valores para Y
            }
            col++;
        }

        X.push_back(xRow);
        Y.push_back(yRow);
    }
    file.close();
}

// Método para dividir los datos en train y test
void Data::splitData(double trainRatio, Data &train, Data &test) {
    int numTrain = static_cast<int>(X.size() * trainRatio);

    for (int i = 0; i < numTrain; ++i) {
        train.X.push_back(X[i]);
        train.Y.push_back(Y[i]);
    }
    for (int i = numTrain; i < X.size(); ++i) {
        test.X.push_back(X[i]);
        test.Y.push_back(Y[i]);
    }
}