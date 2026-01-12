#ifndef NSAVE_H
#define NSAVE_H

#include <vector>
#include <string>
using namespace std;

class NSave {
public:
    static void save(const vector<vector<double>>& weights, const string& filename);
    static vector<vector<double>> load(const string& filename);
};

#endif