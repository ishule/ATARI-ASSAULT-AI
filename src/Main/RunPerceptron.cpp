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
    bool isMultiLabel = false; // Para saber si usar accuracy especial
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

// ---------------------------------------------------------
// PARSER MULTI-SALIDA PARA ASSAULT (COLUMNAS 80, 81, 82)
// ---------------------------------------------------------
static void parseCSV_MultiOutput(const std::string& path, MatDouble_t& X, MatDouble_t& Y, char delimiter) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("No se pudo abrir " + path);

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string v;
        std::vector<double> values;

        // Leer todos los valores de la fila
        while (std::getline(ss, v, delimiter)) {
            try { values.push_back(std::stod(v)); } catch (...) {}
        }
        
        // Esperamos 83 columnas (63 RAM + 17 basura + 3 Botones)
        // Si hay menos, ignoramos la fila por seguridad
        if (values.size() < 83) continue;

        std::vector<double> rowInputs;
        std::vector<double> rowOutputs;

        // 1. INPUTS: Usamos solo las primeras 63 columnas (tu ramImportant)
        for (int i = 0; i < 63; ++i) {
            rowInputs.push_back(values[i] / 255.0); // Normalizar
        }
        X.push_back(rowInputs);

        // 2. OUTPUTS: Usamos las columnas 80, 81 y 82
        // Col 80: Derecha (Mapping A)
        // Col 81: Izquierda (Mapping B)
        // Col 82: Fuego
        // Convertimos 0 -> -1.0 y 1 -> 1.0 para el Perceptrón
        double val80 = (values[80] > 0.5) ? 1.0 : -1.0;
        double val81 = (values[81] > 0.5) ? 1.0 : -1.0;
        double val82 = (values[82] > 0.5) ? 1.0 : -1.0;

        rowOutputs = {val80, val81, val82};
        Y.push_back(rowOutputs);
    }
}

static Dataset loadAssault() {
    Dataset d;
    d.classNames = {"Right", "Left", "Fire"}; 
    d.isMultiLabel = true; // Activa accuracy especial

    std::string path = "data/data_manual_01.csv";
    std::cout << "Cargando Dataset Manual Multi-Salida: " << path << " ...\n";

    MatDouble_t Xall, Yall;
    parseCSV_MultiOutput(path, Xall, Yall, ';'); // Leemos con ;

    if (Xall.empty()) throw std::runtime_error("Dataset vacio o formato incorrecto.");

    // Dividimos 80% Train, 20% Val
    shuffleSplit3(Xall, Yall, 0.8, 0.2, d.Xtrain, d.Ytrain, d.Xval, d.Yval, d.Xtest, d.Ytest);

    d.inputSize = Xall[0].size();  // 63
    d.outputSize = Yall[0].size(); // 3

    std::cout << "Dimensiones -> Inputs: " << d.inputSize 
              << " (RAM) | Outputs: " << d.outputSize << " (Der, Izq, Fuego)\n";
    std::cout << "Total muestras: " << Xall.size() << "\n";
              
    return d;
}
// ---------------------------------------------------------


// Loader para Iris
static Dataset loadIris(const std::string& path, double trainRatio, double valRatio) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("No se pudo abrir " + path);
    std::string line;
    std::getline(file, line); 
    MatDouble_t Xall;
    MatDouble_t Yall;
    const std::vector<std::string> classes = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string v;
        std::vector<double> row;
        std::getline(ss, v, ',');
        for (int i = 0; i < 4; ++i) {
            std::getline(ss, v, ',');
            row.push_back(std::stod(v));
        }
        std::getline(ss, v, ','); 
        Xall.push_back(row); 
        std::vector<double> oh(classes.size(), -1.0);
        auto it = std::find(classes.begin(), classes.end(), v);
        if (it == classes.end()) throw std::runtime_error("Etiqueta desconocida: " + v);
        oh[static_cast<size_t>(std::distance(classes.begin(), it))] = 1.0;
        Yall.push_back(oh);
    }
    Dataset d;
    d.inputSize = 4; d.outputSize = 3; d.classNames = classes;
    shuffleSplit3(Xall, Yall, trainRatio, valRatio, d.Xtrain, d.Ytrain, d.Xval, d.Yval, d.Xtest, d.Ytest);
    return d;
}

// Loader para Cancer
static Dataset loadCancer(const std::string& path, double trainRatio, double valRatio) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("No se pudo abrir " + path);
    std::string line; std::getline(file, line); 
    MatDouble_t Xall, Yall;
    while (std::getline(file, line)) {
        std::stringstream ss(line); std::string v; std::vector<double> row;
        std::getline(ss, v, ','); std::getline(ss, v, ',');
        double label = (v == "M") ? 1.0 : -1.0;
        for (int i = 0; i < 30; ++i) {
            if (!std::getline(ss, v, ',')) throw std::runtime_error("Fila incompleta en cancer");
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
            if (!std::getline(ss, v, ',')) throw std::runtime_error("Fila incompleta en wine");
            row.push_back(std::stod(v));
        }
        if (!std::getline(ss, v, ',')) throw std::runtime_error("Sin etiqueta");
        labelsInt.push_back(std::stoi(v));
        Xall.push_back(row);
    }
    MatDouble_t Yall;
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
    std::string line; std::getline(file, line); 
    MatDouble_t Xall, Yall;
    const std::vector<std::string> classes = {"0","1","2","3","4","5","6","7","8","9"};
    while (std::getline(file, line)) {
        std::stringstream ss(line); std::string v; std::vector<double> row;
        std::getline(ss, v, ','); int label = std::stoi(v);
        for (int i = 0; i < 784; ++i) {
            if (!std::getline(ss, v, ',')) throw std::runtime_error("Fila incompleta en MNIST");
            row.push_back(std::stod(v) / 255.0);
        }
        Xall.push_back(row);
        std::vector<double> oh(10, -1.0); oh[label] = 1.0; Yall.push_back(oh);
    }
    Dataset d; d.inputSize = 784; d.outputSize = 10; d.classNames = classes;
    shuffleSplit3(Xall, Yall, trainRatio, valRatio, d.Xtrain, d.Ytrain, d.Xval, d.Yval, d.Xtest, d.Ytest);
    return d;
}

static std::string stripQuotes(const std::string& s) {
    std::string r = s;
    if (!r.empty() && r.front() == '"') r.erase(0, 1);
    if (!r.empty() && r.back() == '"') r.pop_back();
    return r;
}

// Accuracy Genérico
// Si es multi-label (Assault), requiere que todas las salidas coincidan en signo
// Si es binario/multi-clase (Iris/MNIST), usa lógica estándar
static double accuracy(const Perceptron& model, const MatDouble_t& X, const MatDouble_t& Y, bool isMultiLabel) {
    if (X.empty()) return 0.0;
    int ok = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        auto scores = model.predict(X[i]);
        
        if (isMultiLabel) {
            // Assault: Comprobar cada salida independientemente
            bool allCorrect = true;
            for(size_t j=0; j<scores.size(); ++j) {
                double pred = scores[j] >= 0 ? 1.0 : -1.0;
                // Si discrepan en signo, fallamos
                if (std::abs(pred - Y[i][j]) > 0.1) {
                    allCorrect = false;
                    break;
                }
            }
            if (allCorrect) ok++;
        } 
        else {
            // Estándar (Iris, MNIST, Cancer...)
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
    if (dataset == "assault") d = loadAssault();
    else if (dataset == "iris") d = loadIris(dataPath, trainRatio, valRatio);
    else if (dataset == "cancer") d = loadCancer(dataPath, trainRatio, valRatio);
    else if (dataset == "wine") d = loadWine(dataPath, trainRatio, valRatio);
    else if (dataset == "mnist") d = loadMNIST(dataPath, trainRatio, valRatio);
    else throw std::runtime_error("Dataset no soportado: " + dataset);

    Perceptron model(static_cast<int>(d.inputSize), static_cast<int>(d.outputSize));
    std::cout << "Entrenando perceptron para dataset " << dataset << "...\n";
    
    // Entrenar
    model.train(d.Xtrain, d.Ytrain, d.Xval, d.Yval);

    // Calcular precisiones pasando el flag isMultiLabel
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
    std::cout << "USO: RunPerceptron --dataset assault|iris|cancer|wine "
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
        // Defaults de ruta solo si no es assault (assault ya tiene rutas hardcodeadas en su loader)
        if (dataPath.empty()) {
            if (dataset == "iris") dataPath = "data/Iris.csv";
            else if (dataset == "cancer") dataPath = "data/cancermama.csv";
            else if (dataset == "wine") dataPath = "data/winequality-red.csv";
            else if (dataset == "mnist") dataPath = "data/MNIST/train.csv";
        }
        if (modelPath.empty()) {
            if (dataset == "assault") modelPath = "models/assault_perceptron.txt";
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