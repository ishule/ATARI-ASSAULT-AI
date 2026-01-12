#ifndef DATA_H
#define DATA_H

#include <string>
#include <vector>
#include <iostream>

class Data {
public:
    std::vector<std::vector<double>> X;
    std::vector<std::vector<double>> Y;

    void loadCSV(const std::string &filename, int xCols, int yCols);
    void splitData(double trainRatio, Data &train, Data &test);
};

#endif