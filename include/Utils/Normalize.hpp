#ifndef NORMALIZE_H
#define NORMALIZE_H

#include <vector>
#include <string>

class Normalize {
private:
    std::vector<double> means, stdDevs;

public:
    void fit(const std::vector<std::vector<double>> &data);
    std::vector<std::vector<double>> transform(const std::vector<std::vector<double>> &data);
    void saveScaler(const std::string &filename);
    void loadScaler(const std::string &filename);
};

#endif