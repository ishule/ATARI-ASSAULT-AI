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
    size_t inputSize{};   // Sin bias
    size_t outputSize{};  // Numero de clases/salidas
    std::vector<std::string> classNames;
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

// Loader para Iris (3 clases, 4 features + bias, one-hot {-1,1})
static Dataset loadIris(const std::string& path, double trainRatio, double valRatio) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("No se pudo abrir " + path);

    std::string line;
    // Saltar encabezado
    std::getline(file, line);

    MatDouble_t Xall;
    MatDouble_t Yall;
    const std::vector<std::string> classes = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string v;
        std::vector<double> row;

        // ID
        std::getline(ss, v, ',');
        // 4 features
        for (int i = 0; i < 4; ++i) {
            std::getline(ss, v, ',');
            row.push_back(std::stod(v));
        }
        // label
        std::getline(ss, v, ',');

        // bias
        row.push_back(1.0);
        Xall.push_back(row);

        std::vector<double> oh(classes.size(), -1.0);
        auto it = std::find(classes.begin(), classes.end(), v);
        if (it == classes.end()) throw std::runtime_error("Etiqueta desconocida: " + v);
        oh[static_cast<size_t>(std::distance(classes.begin(), it))] = 1.0;
        Yall.push_back(oh);
    }

    Dataset d;
    d.inputSize = 4;
    d.outputSize = 3;
    d.classNames = classes;
    shuffleSplit3(Xall, Yall, trainRatio, valRatio, d.Xtrain, d.Ytrain, d.Xval, d.Yval, d.Xtest, d.Ytest);
    return d;
}

// Loader para cáncer mama (binario M/B). 30 features + bias. Salida 1 valor (-1 benigno, 1 maligno)
static Dataset loadCancer(const std::string& path, double trainRatio, double valRatio) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("No se pudo abrir " + path);

    std::string line;
    std::getline(file, line);  // encabezado

    MatDouble_t Xall;
    MatDouble_t Yall;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string v;
        std::vector<double> row;

        // id
        std::getline(ss, v, ',');
        // diagnosis
        std::getline(ss, v, ',');
        double label = (v == "M") ? 1.0 : -1.0;

        // 30 features
        for (int i = 0; i < 30; ++i) {
            if (!std::getline(ss, v, ',')) throw std::runtime_error("Fila incompleta en cancer");
            row.push_back(std::stod(v));
        }
        // bias
        row.push_back(1.0);
        Xall.push_back(row);
        Yall.push_back({label});
    }

    Dataset d;
    d.inputSize = 30;
    d.outputSize = 1;
    d.classNames = {"Benigno", "Maligno"};
    shuffleSplit3(Xall, Yall, trainRatio, valRatio, d.Xtrain, d.Ytrain, d.Xval, d.Yval, d.Xtest, d.Ytest);
    return d;
}

// Loader para Wine Quality (multiclase). 11 features + bias. One-hot {-1,1} para calidad.
static Dataset loadWine(const std::string& path, double trainRatio, double valRatio) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("No se pudo abrir " + path);

    std::string line;
    std::getline(file, line);  // encabezado

    MatDouble_t Xall;
    std::vector<int> labelsInt;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string v;
        std::vector<double> row;

        // 11 features
        for (int i = 0; i < 11; ++i) {
            if (!std::getline(ss, v, ',')) throw std::runtime_error("Fila incompleta en wine");
            row.push_back(std::stod(v));
        }
        // quality
        if (!std::getline(ss, v, ',')) throw std::runtime_error("Sin etiqueta de calidad en wine");
        int q = std::stoi(v);
        labelsInt.push_back(q);

        Xall.push_back(row);
    }

    MatDouble_t Yall;
    // normalizar (estandarizar) las 11 features: (x - mean) / std
    const int featCount = 11;
    std::vector<double> mean(featCount, 0.0), stdv(featCount, 0.0);
    for (const auto& row : Xall) {
        for (int c = 0; c < featCount; ++c) mean[c] += row[c];
    }
    for (int c = 0; c < featCount; ++c) mean[c] /= static_cast<double>(Xall.size());
    for (const auto& row : Xall) {
        for (int c = 0; c < featCount; ++c) {
            double d = row[c] - mean[c];
            stdv[c] += d * d;
        }
    }
    for (int c = 0; c < featCount; ++c) {
        stdv[c] = std::sqrt(stdv[c] / static_cast<double>(Xall.size()));
        if (stdv[c] < 1e-12) stdv[c] = 1.0;  // evitar división por cero
    }
    for (auto& row : Xall) {
        for (int c = 0; c < featCount; ++c) {
            row[c] = (row[c] - mean[c]) / stdv[c];
        }
        // añadir bias después de escalar
        row.push_back(1.0);
    }

    // Binarizar la calidad: >=6 buena (1), <6 mala (-1)
    for (int q : labelsInt) {
        double y = (q >= 6) ? 1.0 : -1.0;
        Yall.push_back({y});
    }

    Dataset d;
    d.inputSize = 11;
    d.outputSize = 1;
    d.classNames = {"malo", "bueno"};
    shuffleSplit3(Xall, Yall, trainRatio, valRatio, d.Xtrain, d.Ytrain, d.Xval, d.Yval, d.Xtest, d.Ytest);
    return d;
}

// Loader para MNIST (10 clases, 784 pixels). Normaliza 0-255 a 0-1, one-hot {-1,1}
static Dataset loadMNIST(const std::string& path, double trainRatio, double valRatio) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("No se pudo abrir " + path);

    std::string line;
    std::getline(file, line);  // encabezado (label,pixel0,pixel1,...,pixel783)

    MatDouble_t Xall;
    MatDouble_t Yall;
    const std::vector<std::string> classes = {"0","1","2","3","4","5","6","7","8","9"};

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string v;
        std::vector<double> row;

        // Label (primera columna)
        std::getline(ss, v, ',');
        int label = std::stoi(v);

        // 784 pixels (normalizar a 0-1)
        for (int i = 0; i < 784; ++i) {
            if (!std::getline(ss, v, ',')) throw std::runtime_error("Fila incompleta en MNIST");
            row.push_back(std::stod(v) / 255.0);
        }
        // bias
        row.push_back(1.0);
        Xall.push_back(row);

        // One-hot encoding
        std::vector<double> oh(10, -1.0);
        oh[label] = 1.0;
        Yall.push_back(oh);
    }

    Dataset d;
    d.inputSize = 784;
    d.outputSize = 10;
    d.classNames = classes;
    shuffleSplit3(Xall, Yall, trainRatio, valRatio, d.Xtrain, d.Ytrain, d.Xval, d.Yval, d.Xtest, d.Ytest);
    return d;
}

// Quitar comillas de un string (si las tiene)
static std::string stripQuotes(const std::string& s) {
    std::string r = s;
    if (!r.empty() && r.front() == '"') r.erase(0, 1);
    if (!r.empty() && r.back() == '"') r.pop_back();
    return r;
}

// Loader para Credit Card Fraud (30 features PCA'd + binario). Normalizar, bias, label {-1,1}
static Dataset loadCreditCard(const std::string& path, double trainRatio, double valRatio) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("No se pudo abrir " + path);

    std::string line;
    std::getline(file, line);  // encabezado (Time,V1,V2,...,V28,Amount,Class)

    MatDouble_t Xall;
    std::vector<int> labelsInt;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string v;
        std::vector<double> row;

        // Time (saltar)
        std::getline(ss, v, ',');
        // 28 PCA features (V1-V28)
        for (int i = 0; i < 28; ++i) {
            if (!std::getline(ss, v, ',')) throw std::runtime_error("Fila incompleta en creditcard");
            row.push_back(std::stod(stripQuotes(v)));
        }
        // Amount (normalizar junto con PCA features)
        if (!std::getline(ss, v, ',')) throw std::runtime_error("Sin Amount en creditcard");
        row.push_back(std::stod(stripQuotes(v)));
        // Class (0 = no fraude, 1 = fraude)
        if (!std::getline(ss, v, ',')) throw std::runtime_error("Sin Class en creditcard");
        // Trim espacios y quitar comillas
        v.erase(0, v.find_first_not_of(" \t\r\n"));
        v.erase(v.find_last_not_of(" \t\r\n") + 1);
        v = stripQuotes(v);
        if (v.empty()) throw std::runtime_error("Label vacío en creditcard");
        int label = static_cast<int>(std::stod(v));
        labelsInt.push_back(label);

        Xall.push_back(row);
    }

    MatDouble_t Yall;
    // Normalizar (estandarizar) las 29 features: (x - mean) / std
    const int featCount = 29;
    std::vector<double> mean(featCount, 0.0), stdv(featCount, 0.0);
    for (const auto& row : Xall) {
        for (int c = 0; c < featCount; ++c) mean[c] += row[c];
    }
    for (int c = 0; c < featCount; ++c) mean[c] /= static_cast<double>(Xall.size());
    for (const auto& row : Xall) {
        for (int c = 0; c < featCount; ++c) {
            double d = row[c] - mean[c];
            stdv[c] += d * d;
        }
    }
    for (int c = 0; c < featCount; ++c) {
        stdv[c] = std::sqrt(stdv[c] / static_cast<double>(Xall.size()));
        if (stdv[c] < 1e-12) stdv[c] = 1.0;
    }
    for (auto& row : Xall) {
        for (int c = 0; c < featCount; ++c) {
            row[c] = (row[c] - mean[c]) / stdv[c];
        }
        // bias
        row.push_back(1.0);
    }

    // Labels: 0 (no fraude) -> -1, 1 (fraude) -> 1
    for (int label : labelsInt) {
        double y = (label == 1) ? 1.0 : -1.0;
        Yall.push_back({y});
    }

    Dataset d;
    d.inputSize = 29;
    d.outputSize = 1;
    d.classNames = {"legit", "fraud"};
    shuffleSplit3(Xall, Yall, trainRatio, valRatio, d.Xtrain, d.Ytrain, d.Xval, d.Yval, d.Xtest, d.Ytest);
    return d;
}

// Accuracy: multiclase usa argmax, binario usa signo
static double accuracy(const Perceptron& model, const MatDouble_t& X, const MatDouble_t& Y) {
    int ok = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        auto scores = model.predict(X[i]);
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
    return X.empty() ? 0.0 : (static_cast<double>(ok) / X.size()) * 100.0;
}

static void trainDataset(const std::string& dataset, const std::string& dataPath,
                         const std::string& modelPath, double trainRatio, double valRatio) {
    Dataset d;
    if (dataset == "iris") d = loadIris(dataPath, trainRatio, valRatio);
    else if (dataset == "cancer") d = loadCancer(dataPath, trainRatio, valRatio);
    else if (dataset == "wine") d = loadWine(dataPath, trainRatio, valRatio);
    else if (dataset == "creditcard") d = loadCreditCard(dataPath, trainRatio, valRatio);
    else if (dataset == "mnist") d = loadMNIST(dataPath, trainRatio, valRatio);
    else throw std::runtime_error("Dataset no soportado: " + dataset);

    // d.inputSize ya incluye el bias en cada fila; no volver a sumarlo al construir
    Perceptron model(static_cast<int>(d.inputSize), static_cast<int>(d.outputSize));
    std::cout << "Entrenando perceptron para dataset " << dataset << "...\n";
    model.train(d.Xtrain, d.Ytrain, d.Xval, d.Yval);

    double accTrain = accuracy(model, d.Xtrain, d.Ytrain);
    double accVal = accuracy(model, d.Xval, d.Yval);
    double accTest = d.Xtest.empty() ? 0.0 : accuracy(model, d.Xtest, d.Ytest);

    std::cout << "\n\nMejor Modelo:\n";  // Separador solicitado antes de resultados finales
    std::cout << "Precision train: " << accTrain << "% (" << d.Xtrain.size() << " muestras)\n";
    std::cout << "Precision val:   " << accVal << "% (" << d.Xval.size() << " muestras)\n";
    if (!d.Xtest.empty()) {
        std::cout << "Precision test:  " << accTest << "% (" << d.Xtest.size() << " muestras)\n";
    }


    model.save(modelPath);
    std::cout << "Modelo guardado en: " << modelPath << "\n";
}

static void usage() {
    std::cout << "USO: RunPerceptron --dataset iris|cancer|wine "
                 "[--data ruta_csv] [--model ruta_modelo] [--train-split 0.7] [--val-split 0.15]\n";
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
        // Compat: si usan --split antiguo, se interpreta como train y sin test
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
            if (dataset == "iris") dataPath = "data/Iris.csv";
            else if (dataset == "cancer") dataPath = "data/cancermama.csv";
            else if (dataset == "wine") dataPath = "data/winequality-red.csv";
            else if (dataset == "creditcard") dataPath = "data/creditcard.csv";
            else if (dataset == "mnist") dataPath = "data/MNIST/train.csv";
        }
        if (modelPath.empty()) {
            if (dataset == "iris") modelPath = "models/iris_perceptron.txt";
            else if (dataset == "cancer") modelPath = "models/cancer_perceptron.txt";
            else if (dataset == "wine") modelPath = "models/wine_perceptron.txt";
            else if (dataset == "creditcard") modelPath = "models/creditcard_perceptron.txt";
            else if (dataset == "mnist") modelPath = "models/mnist_perceptron.txt";
        }

        // validar ratios
        if (trainSplit < 0 || valSplit < 0 || trainSplit + valSplit > 1.0) {
            throw std::runtime_error("Ratios invalidos: train+val debe ser <= 1");
        }
        double testSplit = 1.0 - trainSplit - valSplit;

        (void)testSplit;  // reservado por si se hace explícito en CLI
        trainDataset(dataset, dataPath, modelPath, trainSplit, valSplit);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
