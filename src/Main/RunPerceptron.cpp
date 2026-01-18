#include "Perceptron.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <map>
#include <iomanip>
#include <numeric>

// Definiciones de tipos para facilitar lectura
using VecDouble_t = std::vector<double>;
using MatDouble_t = std::vector<VecDouble_t>;

struct Dataset {
    MatDouble_t Xtrain;
    MatDouble_t Ytrain;
    MatDouble_t Xval;
    MatDouble_t Yval;
    MatDouble_t Xtest;
    MatDouble_t Ytest;
    size_t inputSize{};   
    size_t outputSize{};  
    std::vector<std::string> classNames;
    bool isMultiLabel = false; 
    std::string name;
};

// Baraja y separa en train/val/test
static void shuffleSplit3(const MatDouble_t& X, const MatDouble_t& Y,
                          double trainRatio, double valRatio,
                          MatDouble_t& Xtrain, MatDouble_t& Ytrain,
                          MatDouble_t& Xval, MatDouble_t& Yval,
                          MatDouble_t& Xtest, MatDouble_t& Ytest) {
    std::vector<size_t> idx(X.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::mt19937 g(std::random_device{}());
    std::shuffle(idx.begin(), idx.end(), g);

    size_t trainSize = static_cast<size_t>(idx.size() * trainRatio);
    size_t valSize   = static_cast<size_t>(idx.size() * valRatio);
    for (size_t i = 0; i < idx.size(); ++i) {
        size_t j = idx[i];
        if (i < trainSize) {
            Xtrain.push_back(X[j]);
            Ytrain.push_back(Y[j]);
        } else if (i < trainSize + valSize) {
            Xval.push_back(X[j]);
            Yval.push_back(Y[j]);
        } else {
            Xtest.push_back(X[j]);
            Ytest.push_back(Y[j]);
        }
    }
}

// ============================================================================
// BALANCEO INTELIGENTE (Adaptado de RunMLP para Perceptrón)
// ============================================================================
static void balanceAtariData(MatDouble_t& X, MatDouble_t& Y, size_t minSamplesPerClass = 5) {
    std::cout << "\n=== BALANCEANDO DATASET DE ATARI (Adaptado) ===\n";
    
    // PASO 1: Filtrar acciones inválidas (Izquierda+Derecha a la vez)
    // Asumimos orden Y: [Derecha, Izquierda, Fuego] (segun tu parseo anterior)
    // Si Y[i][0]==1 (Derecha) y Y[i][1]==1 (Izquierda) -> CONFLICTO
    MatDouble_t X_filtered, Y_filtered;
    int invalidCount = 0;
    
    for (size_t i = 0; i < Y.size(); ++i) {
        // En tu perceptrón usas -1/1, así que "activado" es > 0.5 (es decir, 1.0)
        if (Y[i].size() >= 2 && Y[i][0] > 0.5 && Y[i][1] > 0.5) {
            invalidCount++;
            continue;
        }
        X_filtered.push_back(X[i]);
        Y_filtered.push_back(Y[i]);
    }
    
    if (invalidCount > 0) {
        std::cout << "  ⚠️  Eliminadas " << invalidCount << " muestras con conflicto L+R\n";
    }
    
    // PASO 2: Agrupar por clase
    std::map<VecDouble_t, std::vector<size_t>> classIndices;
    for (size_t i = 0; i < Y_filtered.size(); ++i) {
        classIndices[Y_filtered[i]].push_back(i);
    }
    
    // Helper para imprimir nombres (Orden: Right, Left, Fire)
    auto getActionName = [](const VecDouble_t& action) -> std::string {
        if (action.size() < 3) return "UNKNOWN";
        bool r = action[0] > 0;
        bool l = action[1] > 0;
        bool f = action[2] > 0;
        
        if (r && f) return "RIGHT+FIRE";
        if (l && f) return "LEFT+FIRE";
        if (f) return "FIRE";
        if (r) return "RIGHT";
        if (l) return "LEFT";
        return "NOOP";
    };
    
    // PASO 3: Filtrar clases válidas y calcular objetivo
    std::vector<VecDouble_t> validClasses;
    std::cout << "\n  Clases válidas (>= " << minSamplesPerClass << " muestras):\n";
    
    std::vector<size_t> sizes;
    for (const auto& pair : classIndices) {
        if (pair.second.size() >= minSamplesPerClass) {
            validClasses.push_back(pair.first);
            sizes.push_back(pair.second.size());
            std::cout << "    ✓ " << std::left << std::setw(12) << getActionName(pair.first) 
                      << ": " << pair.second.size() << "\n";
        }
    }
    
    if (validClasses.empty()) throw std::runtime_error("No hay clases válidas.");

    // Ordenar tamaños para elegir el target (balanceo suave)
    std::sort(sizes.begin(), sizes.end(), std::greater<size_t>());
    size_t targetSize = sizes[0]; 
    if (sizes.size() >= 3) targetSize = sizes[2]; // Usar la 3ª clase más común como límite
    else if (sizes.size() == 2) targetSize = sizes[1]; // O la minoritaria si solo hay 2
    
    std::cout << "\n  → Recortando clases mayoritarias a: " << targetSize << " muestras\n";
    
    // PASO 4: Construir dataset balanceado
    MatDouble_t X_balanced, Y_balanced;
    std::mt19937 g(std::random_device{}());
    
    for (const auto& action : validClasses) {
        auto indices = classIndices[action];
        std::shuffle(indices.begin(), indices.end(), g); // Barajar para coger aleatorios
        
        size_t numToTake = std::min(targetSize, indices.size());
        for (size_t i = 0; i < numToTake; ++i) {
            size_t idx = indices[i];
            X_balanced.push_back(X_filtered[idx]);
            Y_balanced.push_back(Y_filtered[idx]);
        }
    }
    
    // PASO 5: Mezclar todo
    std::vector<size_t> allIndices(X_balanced.size());
    std::iota(allIndices.begin(), allIndices.end(), 0);
    std::shuffle(allIndices.begin(), allIndices.end(), g);
    
    X.clear(); Y.clear();
    for (size_t idx : allIndices) {
        X.push_back(X_balanced[idx]);
        Y.push_back(Y_balanced[idx]);
    }
    
    std::cout << "  → Total Final: " << Y.size() << " muestras (" 
              << 100.0 * Y.size() / Y_filtered.size() << "% del original)\n";
}

// ============================================================================
// CARGA DE ATARI (Adaptada para 83 columnas y salida -1/1)
// ============================================================================
static Dataset loadAtari(const std::string& path, double trainRatio, double valRatio) {
    std::cout << "Cargando Atari Assault desde: " << path << "\n";
    std::ifstream file(path);
    if (!file) throw std::runtime_error("No se pudo abrir " + path);
    
    std::string line;
    MatDouble_t Xall, Yall;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string v;
        std::vector<double> values;
        
        while (std::getline(ss, v, ';')) {
            try { values.push_back(std::stod(v)); } catch(...) {}
        }
        
        if (values.size() < 83) continue; // Ignorar filas rotas

        // 1. INPUTS (Columnas 0-62)
        std::vector<double> features;
        for (int i = 0; i < 63; ++i) {
            features.push_back(values[i] / 255.0);
        }
        
        // 2. OUTPUTS (Columnas 80, 81, 82) -> Right, Left, Fire
        // Convertimos 0/1 a -1/1 para tu Perceptrón
        std::vector<double> actions;
        actions.push_back(values[80] > 0.5 ? 1.0 : -1.0); // Right
        actions.push_back(values[81] > 0.5 ? 1.0 : -1.0); // Left
        actions.push_back(values[82] > 0.5 ? 1.0 : -1.0); // Fire
        
        Xall.push_back(features);
        Yall.push_back(actions);
    }
    
    std::cout << "  Total leido: " << Xall.size() << " muestras\n";
    
    // Aplicar Balanceo
    balanceAtariData(Xall, Yall, 5);
    
    Dataset d;
    d.name = "atari";
    // Lista completa de clases de acción (incluye combinadas y NOOP), similar a RunMLP
    d.classNames = {"NOOP", "RIGHT", "LEFT", "FIRE", "RIGHT+FIRE", "LEFT+FIRE"};
    d.isMultiLabel = true;
    d.inputSize = 63;
    d.outputSize = 3;
    
    shuffleSplit3(Xall, Yall, trainRatio, valRatio, 
                  d.Xtrain, d.Ytrain, d.Xval, d.Yval, d.Xtest, d.Ytest);
    
    std::cout << "\n  Split final: Train=" << d.Xtrain.size() 
              << " Val=" << d.Xval.size() 
              << " Test=" << d.Xtest.size() << "\n\n";
    
    return d;
}

// Loader para Iris
static Dataset loadIris(const std::string& path, double trainRatio, double valRatio) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("No se pudo abrir " + path);
    std::string line; std::getline(file, line); 
    MatDouble_t Xall; MatDouble_t Yall;
    const std::vector<std::string> classes = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};
    while (std::getline(file, line)) {
        std::stringstream ss(line); std::string v; std::vector<double> row;
        std::getline(ss, v, ',');
        for (int i = 0; i < 4; ++i) { std::getline(ss, v, ','); row.push_back(std::stod(v)); }
        std::getline(ss, v, ','); 
        Xall.push_back(row); 
        std::vector<double> oh(classes.size(), -1.0);
        auto it = std::find(classes.begin(), classes.end(), v);
        if (it != classes.end()) oh[std::distance(classes.begin(), it)] = 1.0;
        Yall.push_back(oh);
    }
    Dataset d; d.inputSize = 4; d.outputSize = 3; d.classNames = classes;
    shuffleSplit3(Xall, Yall, trainRatio, valRatio, d.Xtrain, d.Ytrain, d.Xval, d.Yval, d.Xtest, d.Ytest);
    return d;
}

// Loader para Cancer
static Dataset loadCancer(const std::string& path, double trainRatio, double valRatio) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("No se pudo abrir " + path);
    std::string line; std::getline(file, line); MatDouble_t Xall, Yall;
    while (std::getline(file, line)) {
        std::stringstream ss(line); std::string v; std::vector<double> row;
        std::getline(ss, v, ','); std::getline(ss, v, ',');
        double label = (v == "M") ? 1.0 : -1.0;
        for (int i = 0; i < 30; ++i) {
            if (!std::getline(ss, v, ',')) throw std::runtime_error("Error cancer");
            row.push_back(std::stod(v));
        }
        Xall.push_back(row); Yall.push_back({label});
    }
    Dataset d; d.inputSize = 30; d.outputSize = 1; d.classNames = {"Benigno", "Maligno"};
    shuffleSplit3(Xall, Yall, trainRatio, valRatio, d.Xtrain, d.Ytrain, d.Xval, d.Yval, d.Xtest, d.Ytest);
    return d;
}

// Loader para Wine
static Dataset loadWine(const std::string& path, double trainRatio, double valRatio) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("No se pudo abrir " + path);
    std::string line; std::getline(file, line); 
    MatDouble_t Xall; std::vector<int> labelsInt;
    while (std::getline(file, line)) {
        std::stringstream ss(line); std::string v; std::vector<double> row;
        for (int i = 0; i < 11; ++i) {
            if (!std::getline(ss, v, ',')) throw std::runtime_error("Error wine");
            row.push_back(std::stod(v));
        }
        if (!std::getline(ss, v, ',')) throw std::runtime_error("Sin etiqueta");
        labelsInt.push_back(std::stoi(v));
        Xall.push_back(row);
    }
    MatDouble_t Yall;
    // Normalizacion simple
    const int featCount = 11;
    std::vector<double> mean(featCount, 0.0), stdv(featCount, 0.0);
    for (const auto& row : Xall) for (int c = 0; c < featCount; ++c) mean[c] += row[c];
    for (int c = 0; c < featCount; ++c) mean[c] /= Xall.size();
    for (const auto& row : Xall) for (int c = 0; c < featCount; ++c) stdv[c] += pow(row[c] - mean[c], 2);
    for (int c = 0; c < featCount; ++c) stdv[c] = sqrt(stdv[c] / Xall.size());
    for (auto& row : Xall) for (int c = 0; c < featCount; ++c) row[c] = (row[c] - mean[c]) / (stdv[c] + 1e-9);
    
    for (int q : labelsInt) Yall.push_back({(q >= 6) ? 1.0 : -1.0});
    
    Dataset d; d.inputSize = 11; d.outputSize = 1; d.classNames = {"malo", "bueno"};
    shuffleSplit3(Xall, Yall, trainRatio, valRatio, d.Xtrain, d.Ytrain, d.Xval, d.Yval, d.Xtest, d.Ytest);
    return d;
}

// Loader para MNIST
static Dataset loadMNIST(const std::string& path, double trainRatio, double valRatio) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("No se pudo abrir " + path);
    std::string line; std::getline(file, line); MatDouble_t Xall, Yall;
    const std::vector<std::string> classes = {"0","1","2","3","4","5","6","7","8","9"};
    while (std::getline(file, line)) {
        std::stringstream ss(line); std::string v; std::vector<double> row;
        std::getline(ss, v, ','); int label = std::stoi(v);
        for (int i = 0; i < 784; ++i) {
            if (!std::getline(ss, v, ',')) throw std::runtime_error("Error MNIST");
            row.push_back(std::stod(v) / 255.0);
        }
        Xall.push_back(row);
        std::vector<double> oh(10, -1.0); oh[label] = 1.0; Yall.push_back(oh);
    }
    Dataset d; d.inputSize = 784; d.outputSize = 10; d.classNames = classes;
    shuffleSplit3(Xall, Yall, trainRatio, valRatio, d.Xtrain, d.Ytrain, d.Xval, d.Yval, d.Xtest, d.Ytest);
    return d;
}

// Accuracy Genérico
static double accuracy(const Perceptron& model, const MatDouble_t& X, const MatDouble_t& Y, bool isMultiLabel) {
    if (X.empty()) return 0.0;
    int ok = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        auto scores = model.predict(X[i]);
        
        if (isMultiLabel) {
            // Assault: Comprobar cada salida (-1 o 1)
            bool allCorrect = true;
            for(size_t j=0; j<scores.size(); ++j) {
                // Predicción del perceptrón (-1 o 1)
                double pred = scores[j] >= 0 ? 1.0 : -1.0;
                // Target del dataset (-1 o 1)
                double target = Y[i][j]; 
                
                if (std::abs(pred - target) > 0.1) {
                    allCorrect = false;
                    break;
                }
            }
            if (allCorrect) ok++;
        } 
        else {
            if (scores.size() == 1) {
                double pred = scores[0] >= 0 ? 1.0 : -1.0;
                if (pred == Y[i][0]) ok++;
            } else {
                size_t pred = static_cast<size_t>(std::distance(scores.begin(),
                                         std::max_element(scores.begin(), scores.end())));
                size_t truth = static_cast<size_t>(std::distance(Y[i].begin(),
                                          std::max_element(Y[i].begin(), Y[i].end())));
                if (pred == truth) ok++;
            }
        }
    }
    return (static_cast<double>(ok) / X.size()) * 100.0;
}

static void trainDataset(const std::string& dataset, const std::string& dataPath,
                         const std::string& modelPath, double trainRatio, double valRatio) {
    Dataset d;
    if (dataset == "atari") d = loadAtari(dataPath, trainRatio, valRatio);
    else if (dataset == "iris") d = loadIris(dataPath, trainRatio, valRatio);
    else if (dataset == "cancer") d = loadCancer(dataPath, trainRatio, valRatio);
    else if (dataset == "wine") d = loadWine(dataPath, trainRatio, valRatio);
    else if (dataset == "mnist") d = loadMNIST(dataPath, trainRatio, valRatio);
    else throw std::runtime_error("Dataset no soportado: " + dataset);

    Perceptron model(static_cast<int>(d.inputSize), static_cast<int>(d.outputSize));
    std::cout << "Entrenando perceptron para dataset " << dataset << "...\n";
    
    model.train(d.Xtrain, d.Ytrain, d.Xval, d.Yval);

    double accTrain = accuracy(model, d.Xtrain, d.Ytrain, d.isMultiLabel);
    double accVal = accuracy(model, d.Xval, d.Yval, d.isMultiLabel);
    double accTest = d.Xtest.empty() ? 0.0 : accuracy(model, d.Xtest, d.Ytest, d.isMultiLabel);

    std::cout << "\n\nMejor Modelo:\n";
    std::cout << "Precision train: " << accTrain << "% (" << d.Xtrain.size() << " muestras)\n";
    std::cout << "Precision val:   " << accVal << "% (" << d.Xval.size() << " muestras)\n";
    if (!d.Xtest.empty()) {
        std::cout << "Precision test:  " << accTest << "% (" << d.Xtest.size() << " muestras)\n";
    }

    model.save(modelPath);
    std::cout << "Modelo guardado en: " << modelPath << "\n";
}

static void usage() {
    std::cout << "USO: RunPerceptron --dataset atari|iris|cancer|wine "
                 "[--data ruta_csv] [--model ruta_modelo]\n";
}

int main(int argc, char** argv) {
    std::string dataset;
    std::string dataPath;
    std::string modelPath;
    double trainSplit = 0.7;
    double valSplit = 0.15;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--dataset" && i + 1 < argc) dataset = argv[++i];
        else if (a == "--data" && i + 1 < argc) dataPath = argv[++i];
        else if (a == "--model" && i + 1 < argc) modelPath = argv[++i];
        else if (a == "--train-split" && i + 1 < argc) trainSplit = std::stod(argv[++i]);
        else if (a == "--val-split" && i + 1 < argc) valSplit = std::stod(argv[++i]);
        else if (a == "--split" && i + 1 < argc) {
            trainSplit = std::stod(argv[++i]);
            valSplit = 1.0 - trainSplit;
        }
        else if (a == "--help") { usage(); return 0; }
    }

    try {
        if (dataset.empty()) {
            usage();
            return 1;
        }
        
        if (dataPath.empty()) {
            if (dataset == "atari") dataPath = "datasets_juntos.csv"; 
            else if (dataset == "iris") dataPath = "data/Iris.csv";
            else if (dataset == "cancer") dataPath = "data/cancermama.csv";
            else if (dataset == "wine") dataPath = "data/winequality-red.csv";
            else if (dataset == "mnist") dataPath = "data/MNIST/train.csv";
        }
        if (modelPath.empty()) {
            if (dataset == "atari") modelPath = "models/assault_perceptron.txt";
            else if (dataset == "iris") modelPath = "models/iris_perceptron.txt";
            else if (dataset == "cancer") modelPath = "models/cancer_perceptron.txt";
            else if (dataset == "wine") modelPath = "models/wine_perceptron.txt";
            else if (dataset == "mnist") modelPath = "models/mnist_perceptron.txt";
        }

        if (trainSplit < 0 || valSplit < 0 || trainSplit + valSplit > 1.0) {
            throw std::runtime_error("Ratios invalidos: train+val debe ser <= 1");
        }
        trainDataset(dataset, dataPath, modelPath, trainSplit, valSplit);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}