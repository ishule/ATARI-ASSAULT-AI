#include "GeneticAlgorithm/GeneticAlgorithm.hpp"
#include "ActivationFunctions.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cstring>
#include <cmath>
using namespace std;

struct DatasetGA {
    vector<vector<double>> Xtrain, Ytrain, Xval, Yval, Xtest, Ytest;
    size_t inputSize = 0;
    size_t outputSize = 0;
    string name;
};


static void shuffleSplit3(const vector<vector<double>>& Xall,
                          const vector<vector<double>>& Yall,
                          double trainRatio, double valRatio,
                          vector<vector<double>>& Xtrain,
                          vector<vector<double>>& Ytrain,
                          vector<vector<double>>& Xval,
                          vector<vector<double>>& Yval,
                          vector<vector<double>>& Xtest,
                          vector<vector<double>>& Ytest) {
    vector<size_t> idx(Xall.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::mt19937 g(std::random_device{}());
    std::shuffle(idx.begin(), idx.end(), g);

    size_t trainSize = static_cast<size_t>(idx.size() * trainRatio);
    size_t valSize   = static_cast<size_t>(idx.size() * valRatio);

    for (size_t i = 0; i < idx.size(); ++i) {
        size_t j = idx[i];
        if (i < trainSize) {
            Xtrain.push_back(Xall[j]);
            Ytrain.push_back(Yall[j]);
        } else if (i < trainSize + valSize) {
            Xval.push_back(Xall[j]);
            Yval.push_back(Yall[j]);
        } else {
            Xtest.push_back(Xall[j]);
            Ytest.push_back(Yall[j]);
        }
    }
}

static DatasetGA loadIrisGA(const string& path, double trainRatio = 0.7, double valRatio = 0.15) {
    ifstream file(path);
    if (!file) throw runtime_error("No se pudo abrir " + path);

    string line;
    getline(file, line); // header

    vector<vector<double>> Xall, Yall;
    const vector<string> classes = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};

    while (getline(file, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string v;
        vector<double> row;

        getline(ss, v, ','); 
        for (int i = 0; i < 4; ++i) {
            if (!getline(ss, v, ',')) break;
            row.push_back(stod(v));
        }
        if (!getline(ss, v, ',')) continue;

        vector<double> oh(3, 0.0);
        auto it = find(classes.begin(), classes.end(), v);
        if (it != classes.end()) oh[distance(classes.begin(), it)] = 1.0;

        if (row.size() == 4) {
            Xall.push_back(row);
            Yall.push_back(oh);
        }
    }

    DatasetGA d;
    d.name = "iris";
    d.inputSize = 4;
    d.outputSize = 3;
    shuffleSplit3(Xall, Yall, trainRatio, valRatio, d.Xtrain, d.Ytrain, d.Xval, d.Yval, d.Xtest, d.Ytest);
    return d;
}

static DatasetGA loadCancerGA(const string& path, double trainRatio = 0.7, double valRatio = 0.15) {
    ifstream file(path);
    if (!file) throw runtime_error("No se pudo abrir " + path);

    string line;
    getline(file, line);
    vector<vector<double>> Xall, Yall;

    while (getline(file, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string v;
        vector<double> row;

        getline(ss, v, ','); 
        getline(ss, v, ','); 
        double label = (v == "M") ? 1.0 : 0.0;

        for (int i = 0; i < 30; ++i) {
            if (!getline(ss, v, ',')) break;
            if (!v.empty()) row.push_back(stod(v));
        }

        if (row.size() == 30) {
            Xall.push_back(row);
            Yall.push_back({label});
        }
    }

    // z-score normalization
    if (!Xall.empty()) {
        size_t numFeatures = 30;
        vector<double> means(numFeatures, 0.0), stds(numFeatures, 0.0);
        for (const auto& s : Xall) for (size_t j = 0; j < numFeatures; ++j) means[j] += s[j];
        for (auto& m : means) m /= Xall.size();
        for (const auto& s : Xall) for (size_t j = 0; j < numFeatures; ++j) {
            double diff = s[j] - means[j]; stds[j] += diff * diff;
        }
        for (auto& sd : stds) sd = sqrt(sd / Xall.size());
        for (auto& s : Xall) for (size_t j = 0; j < numFeatures; ++j) if (stds[j] > 1e-8) s[j] = (s[j] - means[j]) / stds[j];
    }

    DatasetGA d;
    d.name = "cancer";
    d.inputSize = 30;
    d.outputSize = 1;
    shuffleSplit3(Xall, Yall, trainRatio, valRatio, d.Xtrain, d.Ytrain, d.Xval, d.Yval, d.Xtest, d.Ytest);
    return d;
}

static DatasetGA loadWineGA(const string& path, double trainRatio = 0.7, double valRatio = 0.15) {
    ifstream file(path);
    if (!file) throw runtime_error("No se pudo abrir " + path);

    string line;
    getline(file, line);

    vector<vector<double>> Xall, Yall;

    while (getline(file, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string v;
        vector<double> row;
        for (int i = 0; i < 11; ++i) {
            if (!getline(ss, v, ',')) break;
            row.push_back(stod(v));
        }
        if (!getline(ss, v, ',')) continue;
        int quality = stoi(v);
        double label = (quality >= 6) ? 1.0 : 0.0;

        if (row.size() == 11) {
            Xall.push_back(row);
            Yall.push_back({label});
        }
    }

    // z-score normalization
    if (!Xall.empty()) {
        size_t numFeatures = Xall[0].size();
        vector<double> means(numFeatures, 0.0), stds(numFeatures, 0.0);
        for (const auto& s : Xall) for (size_t j = 0; j < numFeatures; ++j) means[j] += s[j];
        for (auto& m : means) m /= Xall.size();
        for (const auto& s : Xall) for (size_t j = 0; j < numFeatures; ++j) {
            double diff = s[j] - means[j]; stds[j] += diff * diff;
        }
        for (auto& sd : stds) sd = sqrt(sd / Xall.size());
        for (auto& s : Xall) for (size_t j = 0; j < numFeatures; ++j) if (stds[j] > 1e-8) s[j] = (s[j] - means[j]) / stds[j];
    }

    DatasetGA d;
    d.name = "wine";
    d.inputSize = 11;
    d.outputSize = 1;
    shuffleSplit3(Xall, Yall, trainRatio, valRatio, d.Xtrain, d.Ytrain, d.Xval, d.Yval, d.Xtest, d.Ytest);
    return d;
}

static DatasetGA loadMNISTGA(const string& trainPath, const string& testPath, 
                             double trainRatio, double valRatio) {
    cout << "Cargando MNIST para GA...\n";
    
    DatasetGA d;
    d.name = "mnist";
    d.inputSize = 784;  // 28x28 pixeles
    d.outputSize = 10;  // 10 dígitos (0-9)
    
    // Cargar datos de TRAIN
    ifstream trainFile(trainPath);
    if (!trainFile) throw runtime_error("No se pudo abrir " + trainPath);
    
    string line;
    getline(trainFile, line);  // Skip header
    
    vector<vector<double>> XtrainAll, YtrainAll;
    int trainCount = 0;
    int maxTrainSamples = 10000;  // Limitado para velocidad del GA
    
    cout << "  Leyendo train (max " << maxTrainSamples << ")..." << flush;
    
    while (getline(trainFile, line) && trainCount < maxTrainSamples) {
        if (line.empty()) continue;
        stringstream ss(line);
        string v;
        
        getline(ss, v, ',');
        int label = stoi(v);
        
        vector<double> oh(10, 0.0);
        if (label >= 0 && label < 10) oh[label] = 1.0;
        
        vector<double> pixels;
        pixels.reserve(784);
        
        for (int i = 0; i < 784; ++i) {
            if (!getline(ss, v, ',')) break;
            pixels.push_back(stod(v) / 255.0);
        }
        
        if (pixels.size() == 784) {
            XtrainAll.push_back(pixels);
            YtrainAll.push_back(oh);
            trainCount++;
        }
        if (trainCount % 2000 == 0) cout << "." << flush;
    }
    trainFile.close();
    cout << " " << trainCount << " muestras.\n";
    
    // Cargar datos de TEST
    ifstream testFile(testPath);
    if (!testFile) throw runtime_error("No se pudo abrir " + testPath);
    
    getline(testFile, line);  // Skip header
    int testCount = 0;
    int maxTestSamples = 2000;  
    
    cout << "  Leyendo test (max " << maxTestSamples << ")..." << flush;
    
    while (getline(testFile, line) && testCount < maxTestSamples) {
        if (line.empty()) continue;
        stringstream ss(line);
        string v;
        
        getline(ss, v, ',');
        int label = stoi(v);
        
        vector<double> oh(10, 0.0);
        if (label >= 0 && label < 10) oh[label] = 1.0;
        
        vector<double> pixels;
        pixels.reserve(784);
        
        for (int i = 0; i < 784; ++i) {
            if (!getline(ss, v, ',')) break;
            pixels.push_back(stod(v) / 255.0);
        }
        
        if (pixels.size() == 784) {
            d.Xtest.push_back(pixels);
            d.Ytest.push_back(oh);
            testCount++;
        }
        if (testCount % 500 == 0) cout << "." << flush;
    }
    testFile.close();
    cout << " " << testCount << " muestras.\n";
    
    // Split Train / Val (Para GA usaremos todo Train+Val en la evolución)
    vector<size_t> idx(XtrainAll.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::mt19937 g(std::random_device{}());
    std::shuffle(idx.begin(), idx.end(), g);
    
    size_t actualTrainSize = static_cast<size_t>(idx.size() * trainRatio / (trainRatio + valRatio));
    
    for (size_t i = 0; i < idx.size(); ++i) {
        size_t j = idx[i];
        if (i < actualTrainSize) {
            d.Xtrain.push_back(XtrainAll[j]);
            d.Ytrain.push_back(YtrainAll[j]);
        } else {
            d.Xval.push_back(XtrainAll[j]);
            d.Yval.push_back(YtrainAll[j]);
        }
    }
    
    return d;
}

void loadCSV(const string& filepath,
             vector<vector<double>>& X,
             vector<vector<double>>& Y,
             bool hasHeader = true,
             int labelCol = -1) {
    
    ifstream file(filepath);
    if (!file) throw runtime_error("No se pudo abrir: " + filepath);
    
    string line;
    bool isFirst = true;
    int numCols = 0;
    
    while (getline(file, line)) {
        if (line.empty()) continue;
        if (hasHeader && isFirst) { isFirst = false; continue; }
        
        stringstream ss(line);
        string cell;
        vector<double> row;
        
        while (getline(ss, cell, ',')) {
            try { row.push_back(stod(cell)); } catch (...) { row.push_back(0.0); }
        }
        if (row.empty()) continue;
        if (numCols == 0) numCols = row.size();
        
        int lc = (labelCol < 0) ? numCols - 1 : labelCol;
        vector<double> features;
        for (int i = 0; i < static_cast<int>(row.size()); ++i) if (i != lc) features.push_back(row[i]);
        
        X.push_back(features);
        Y.push_back({row[lc]});
    }
}

void normalize(vector<vector<double>>& X) {
    if (X.empty()) return;
    size_t numFeatures = X[0].size();
    for (size_t f = 0; f < numFeatures; ++f) {
        double minVal = X[0][f], maxVal = X[0][f];
        for (const auto& row : X) {
            if (row[f] < minVal) minVal = row[f];
            if (row[f] > maxVal) maxVal = row[f];
        }
        double range = maxVal - minVal;
        if (range < 1e-8) range = 1.0;
        for (auto& row : X) row[f] = (row[f] - minVal) / range;
    }
}

void toOneHot(vector<vector<double>>& Y, int numClasses) {
    vector<vector<double>> oneHot;
    for (const auto& y : Y) {
        int label = static_cast<int>(y[0]);
        vector<double> encoded(numClasses, 0.0);
        if (label >= 0 && label < numClasses) encoded[label] = 1.0;
        oneHot.push_back(encoded);
    }
    Y = oneHot;
}

void trainTestSplit(const vector<vector<double>>& X,
                    const vector<vector<double>>& Y,
                    vector<vector<double>>& Xtrain,
                    vector<vector<double>>& Ytrain,
                    vector<vector<double>>& Xtest,
                    vector<vector<double>>& Ytest,
                    double testRatio = 0.2) {
    vector<size_t> indices(X.size());
    iota(indices.begin(), indices.end(), 0);
    random_device rd;
    mt19937 gen(rd());
    shuffle(indices.begin(), indices.end(), gen);
    
    size_t testSize = static_cast<size_t>(X.size() * testRatio);
    for (size_t i = 0; i < indices.size(); ++i) {
        if (i < testSize) {
            Xtest.push_back(X[indices[i]]);
            Ytest.push_back(Y[indices[i]]);
        } else {
            Xtrain.push_back(X[indices[i]]);
            Ytrain.push_back(Y[indices[i]]);
        }
    }
}

vector<int> parseArchitecture(const string& arch) {
    vector<int> topology;
    stringstream ss(arch);
    string token;
    while (getline(ss, token, '-')) topology.push_back(stoi(token));
    return topology;
}

double evaluateAccuracy(const Individual& ind,
                       const vector<vector<double>>& X,
                       const vector<vector<double>>& Y) {
    int correct = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        auto pred = ind.predict(X[i]);
        if (Y[i].size() == 1) {
            double predicted = (pred[0] >= 0.5) ? 1.0 : 0.0;
            if (abs(predicted - Y[i][0]) < 0.1) correct++;
        } else {
            auto predMax = max_element(pred.begin(), pred.end());
            auto trueMax = max_element(Y[i].begin(), Y[i].end());
            if (distance(pred.begin(), predMax) == distance(Y[i].begin(), trueMax)) correct++;
        }
    }
    if (X.empty()) return 0.0;
    return 100.0 * correct / X.size();
}

void printUsage(const char* programName) {
    cout << "\nUso: " << programName << " [opciones]\n\n";
    cout << "MODOS:\n";
    cout << "  --mode weights    Evolución de pesos\n";
    cout << "  --mode neuro      Neuroevolución (topología + pesos)\n\n";
    cout << "DATOS:\n";
    cout << "  --dataset <name>  mnist, iris, cancer, wine (o path CSV)\n";
    cout << "  --arch <topo>     Ej: 784-20-10 (para mode weights)\n";
}
int main(int argc, char* argv[]) {
    // Valores por defecto
    string mode = "weights";
    string archStr = "4-10-3";
    string datasetPath = "";
    string savePath = "";
    string activationStr = "RELU";
    
    int inputSize = 0, outputSize = 0, numClasses = 0;
    int labelCol = -1;
    bool hasHeader = true;
    
    GAConfig config;
    config.populationSize = 50;
    config.maxGenerations = 100;
    config.mutationRate = 0.1;
    config.archMutationRate = 0.05;
    config.eliteRatio = 0.1;
    config.minHiddenLayers = 1;
    config.maxHiddenLayers = 4;
    config.minNeuronsPerLayer = 4;
    config.maxNeuronsPerLayer = 64;
    config.verbose = true;
    
    // Parsear argumentos
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { printUsage(argv[0]); return 0; }
        else if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (arg == "--arch" && i + 1 < argc) archStr = argv[++i];
        else if (arg == "--dataset" && i + 1 < argc) datasetPath = argv[++i];
        else if (arg == "--save" && i + 1 < argc) savePath = argv[++i];
        else if (arg == "--input" && i + 1 < argc) inputSize = std::stoi(argv[++i]);
        else if (arg == "--output" && i + 1 < argc) outputSize = std::stoi(argv[++i]);
        else if (arg == "--num-classes" && i + 1 < argc) numClasses = std::stoi(argv[++i]);
        else if (arg == "--label-col" && i + 1 < argc) labelCol = std::stoi(argv[++i]);
        else if (arg == "--pop" && i + 1 < argc) config.populationSize = std::stoi(argv[++i]);
        else if (arg == "--gen" && i + 1 < argc) config.maxGenerations = std::stoi(argv[++i]);
        else if (arg == "--mutation" && i + 1 < argc) config.mutationRate = std::stod(argv[++i]);
        else if (arg == "--arch-mutation" && i + 1 < argc) config.archMutationRate = std::stod(argv[++i]);
        else if (arg == "--elite" && i + 1 < argc) config.eliteRatio = std::stod(argv[++i]);
        else if (arg == "--min-layers" && i + 1 < argc) config.minHiddenLayers = std::stoi(argv[++i]);
        else if (arg == "--max-layers" && i + 1 < argc) config.maxHiddenLayers = std::stoi(argv[++i]);
        else if (arg == "--min-neurons" && i + 1 < argc) config.minNeuronsPerLayer = std::stoi(argv[++i]);
        else if (arg == "--max-neurons" && i + 1 < argc) config.maxNeuronsPerLayer = std::stoi(argv[++i]);
        else if (arg == "--activation" && i + 1 < argc) activationStr = argv[++i];
        else if (arg == "--no-header") hasHeader = false;
        else if (arg == "--quiet") config.verbose = false;
    }
    
    if (datasetPath.empty()) { cerr << "Error: --dataset requerido\n"; return 1; }
    ActivationType activation = ActivationFunctions::fromString(activationStr);
    
    // Carga de datos
    vector<vector<double>> Xtrain, Ytrain, Xtest, Ytest;
    bool loadedWithSpecial = false;

    cout << "Dataset: " << datasetPath << "\n";

    if (datasetPath == "iris" || datasetPath == "cancer" || datasetPath == "wine" || datasetPath == "mnist") {
        DatasetGA d;
        if (datasetPath == "iris") d = loadIrisGA("data/Iris.csv");
        else if (datasetPath == "cancer") d = loadCancerGA("data/cancermama.csv");
        else if (datasetPath == "wine") d = loadWineGA("data/winequality-red.csv");
        else if (datasetPath == "mnist") {
             d = loadMNISTGA("data/MNIST/train.csv", "data/MNIST/test.csv", 0.7, 0.15);
        }

        if (!d.Xtrain.empty()) {
            Xtrain = d.Xtrain;
            Ytrain = d.Ytrain;
            // Unir Train+Val para GA
            Xtrain.insert(Xtrain.end(), d.Xval.begin(), d.Xval.end());
            Ytrain.insert(Ytrain.end(), d.Yval.begin(), d.Yval.end());
            Xtest = d.Xtest;
            Ytest = d.Ytest;
            if (numClasses <= 0) numClasses = static_cast<int>(d.outputSize);
            loadedWithSpecial = true;
        }
    }

    if (!loadedWithSpecial) {
        vector<vector<double>> X, Y;
        loadCSV(datasetPath, X, Y, hasHeader, labelCol);
        if (X.empty()) return 1;
        normalize(X);
        if (numClasses > 0) toOneHot(Y, numClasses);
        trainTestSplit(X, Y, Xtrain, Ytrain, Xtest, Ytest, 0.2);
    }
    
    cout << "Train Samples: " << Xtrain.size() << "\n";
    cout << "Test Samples:  " << Xtest.size() << "\n\n";
    
    Individual best({1});
    
    if (mode == "weights") {
        cout << "=== MODO: WEIGHTS EVOLUTION ===\n";
        vector<int> topology = parseArchitecture(archStr);
        
        // Ajuste automático de entradas/salidas si es necesario
        if (!Xtrain.empty() && !Ytrain.empty()) {
            if (topology.front() != static_cast<int>(Xtrain[0].size())) 
                topology[0] = static_cast<int>(Xtrain[0].size());
            if (topology.back() != static_cast<int>(Ytrain[0].size())) 
                topology.back() = static_cast<int>(Ytrain[0].size());
        }
        
        // MOSTRAR ARQUITECTURA AL INICIO
        cout << "Arquitectura: ";
        for (size_t i = 0; i < topology.size(); ++i) {
            cout << topology[i] << (i < topology.size()-1 ? "-" : "");
        }
        cout << "\n\n";
        
        config.evolveArchitecture = false;
        GeneticAlgorithm ga(config, topology, activation);
        best = ga.evolve(Xtrain, Ytrain);
        
    } else if (mode == "neuro") {
        cout << "=== MODO: NEUROEVOLUTION ===\n";
        if (inputSize == 0 && !Xtrain.empty()) inputSize = static_cast<int>(Xtrain[0].size());
        if (outputSize == 0 && !Ytrain.empty()) outputSize = static_cast<int>(Ytrain[0].size());
        
        cout << "Inputs: " << inputSize << " | Outputs: " << outputSize << "\n";
        cout << "Hidden Layers: " << config.minHiddenLayers << "-" << config.maxHiddenLayers << "\n";
        
        config.evolveArchitecture = true;
        GeneticAlgorithm ga(config, inputSize, outputSize, activation);
        best = ga.evolve(Xtrain, Ytrain);
    }
    
    // Resultados finales
    cout << "\n========================================\n";
    cout << "RESULTADOS FINALES\n";
    cout << "========================================\n";
    
    double trainAcc = evaluateAccuracy(best, Xtrain, Ytrain);
    double testAcc = evaluateAccuracy(best, Xtest, Ytest);
    
    // Arquitectura final
    cout << "Mejor arquitectura: " << best.getArchitectureString() << "\n";
    cout << "Fitness: " << best.getFitness() << "\n";
    cout << "Accuracy Train: " << trainAcc << "%\n";
    cout << "Accuracy Test:  " << testAcc << "%\n";
    
    if (!savePath.empty()) {
        best.save(savePath);
        cout << "Modelo guardado en: " << savePath << "\n";
    }
    
    return 0;
}