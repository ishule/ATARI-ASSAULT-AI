#include "MLP.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <iomanip>
#include <sys/stat.h>
#include <cmath>

static void createDirectory(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        #ifdef _WIN32
            _mkdir(path.c_str());
        #else
            mkdir(path.c_str(), 0755);
        #endif
    }
}

static bool fileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

static std::string generateModelFilename(const std::string& dataset,
                                        const std::vector<int>& arch,
                                        const std::string& activation,
                                        const std::string& phase,
                                        int expNum) {
    std::stringstream ss;
    ss << "models/mlp/" << dataset << "_exp" << expNum << "_";
    
    for (size_t i = 0; i < arch.size(); ++i) {
        ss << arch[i];
        if (i < arch.size() - 1) ss << "-";
    }
    ss << "_" << activation;
    
    if (!phase.empty()) {
        ss << "_" << phase;
    }
    
    ss << ".txt";
    return ss.str();
}

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

struct Dataset {
    MatDouble_t Xtrain, Ytrain, Xval, Yval, Xtest, Ytest;
    size_t inputSize, outputSize;
    std::string name;
};

static Dataset loadIris(const std::string& path, double trainRatio, double valRatio) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("No se pudo abrir " + path);

    std::string line;
    std::getline(file, line);

    MatDouble_t Xall, Yall;
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

        std::vector<double> oh(3, 0.0);
        auto it = std::find(classes.begin(), classes.end(), v);
        if (it != classes.end()) {
            oh[std::distance(classes.begin(), it)] = 1.0;
        }
        Yall.push_back(oh);
    }

    Dataset d;
    d.name = "iris";
    d.inputSize = 4;
    d.outputSize = 3;
    shuffleSplit3(Xall, Yall, trainRatio, valRatio, 
                  d.Xtrain, d.Ytrain, d.Xval, d.Yval, d.Xtest, d.Ytest);
    return d;
}

static Dataset loadCancer(const std::string& path, double trainRatio, double valRatio) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("No se pudo abrir " + path);

    std::string line;
    std::getline(file, line);

    MatDouble_t Xall, Yall;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string v;
        std::vector<double> row;

        std::getline(ss, v, ',');
        std::getline(ss, v, ',');
        double label = (v == "M") ? 1.0 : 0.0;

        for (int i = 0; i < 30; ++i) {
            std::getline(ss, v, ',');
            if (!v.empty()) row.push_back(std::stod(v));
        }
        
        if (row.size() == 30) {
            Xall.push_back(row);
            Yall.push_back({label});
        }
    }

    Dataset d;
    d.name = "cancer";
    d.inputSize = 30;
    d.outputSize = 1;
    shuffleSplit3(Xall, Yall, trainRatio, valRatio, 
                  d.Xtrain, d.Ytrain, d.Xval, d.Yval, d.Xtest, d.Ytest);
    return d;
}

static Dataset loadWine(const std::string& path, double trainRatio, double valRatio) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("No se pudo abrir " + path);

    std::string line;
    std::getline(file, line);

    MatDouble_t Xall, Yall;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string v;
        std::vector<double> row;

        for (int i = 0; i < 11; ++i) {
            std::getline(ss, v, ';');
            if (!v.empty()) row.push_back(std::stod(v));
        }
        std::getline(ss, v, ';');
        if (!v.empty()) {
            double quality = (std::stod(v) >= 6) ? 1.0 : 0.0;
            
            if (row.size() == 11) {
                Xall.push_back(row);
                Yall.push_back({quality});
            }
        }
    }

    Dataset d;
    d.name = "wine";
    d.inputSize = 11;
    d.outputSize = 1;
    shuffleSplit3(Xall, Yall, trainRatio, valRatio, 
                  d.Xtrain, d.Ytrain, d.Xval, d.Yval, d.Xtest, d.Ytest);
    return d;
}

// ============================================================================
// MNIST Dataset - IGUAL QUE EN RunPerceptron.cpp
// ============================================================================
static Dataset loadMNIST(const std::string& trainPath, const std::string& testPath, 
                        double trainRatio, double valRatio) {
    std::cout << "Cargando MNIST...\n";
    
    Dataset d;
    d.name = "mnist";
    d.inputSize = 784;  // 28x28 pixeles
    d.outputSize = 10;  // 10 dígitos (0-9)
    
    // ========================================================================
    // Cargar datos de TRAIN
    // ========================================================================
    std::ifstream trainFile(trainPath);
    if (!trainFile) throw std::runtime_error("No se pudo abrir " + trainPath);
    
    std::string line;
    std::getline(trainFile, line);  // Skip header si existe
    
    MatDouble_t XtrainAll, YtrainAll;
    int trainCount = 0;
    int maxTrainSamples = 10000;  // Limitar para velocidad
    
    std::cout << "  Leyendo train..." << std::flush;
    
    while (std::getline(trainFile, line) && trainCount < maxTrainSamples) {
        if (line.empty()) continue;
        
        std::stringstream ss(line);
        std::string v;
        
        // Primera columna: label (0-9)
        std::getline(ss, v, ',');
        int label = std::stoi(v);
        
        // One-hot encoding
        std::vector<double> oh(10, 0.0);
        oh[label] = 1.0;
        
        // Siguientes 784 columnas: pixeles
        std::vector<double> pixels;
        pixels.reserve(784);
        
        for (int i = 0; i < 784; ++i) {
            if (!std::getline(ss, v, ',')) break;
            // Normalizar de [0, 255] a [0, 1]
            pixels.push_back(std::stod(v) / 255.0);
        }
        
        if (pixels.size() == 784) {
            XtrainAll.push_back(pixels);
            YtrainAll.push_back(oh);
            trainCount++;
        }
        
        if (trainCount % 2000 == 0) {
            std::cout << "." << std::flush;
        }
    }
    
    trainFile.close();
    std::cout << " " << trainCount << " muestras\n";
    
    // ========================================================================
    // Cargar datos de TEST
    // ========================================================================
    std::ifstream testFile(testPath);
    if (!testFile) throw std::runtime_error("No se pudo abrir " + testPath);
    
    std::getline(testFile, line);  // Skip header
    
    MatDouble_t XtestAll, YtestAll;
    int testCount = 0;
    int maxTestSamples = 2000;  // Limitar para velocidad
    
    std::cout << "  Leyendo test..." << std::flush;
    
    while (std::getline(testFile, line) && testCount < maxTestSamples) {
        if (line.empty()) continue;
        
        std::stringstream ss(line);
        std::string v;
        
        std::getline(ss, v, ',');
        int label = std::stoi(v);
        
        std::vector<double> oh(10, 0.0);
        oh[label] = 1.0;
        
        std::vector<double> pixels;
        pixels.reserve(784);
        
        for (int i = 0; i < 784; ++i) {
            if (!std::getline(ss, v, ',')) break;
            pixels.push_back(std::stod(v) / 255.0);
        }
        
        if (pixels.size() == 784) {
            XtestAll.push_back(pixels);
            YtestAll.push_back(oh);
            testCount++;
        }
        
        if (testCount % 500 == 0) {
            std::cout << "." << std::flush;
        }
    }
    
    testFile.close();
    std::cout << " " << testCount << " muestras\n";
    
    // ========================================================================
    // Dividir TRAIN en train y validation
    // ========================================================================
    std::vector<size_t> idx(XtrainAll.size());
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
    
    // Test ya está separado
    d.Xtest = XtestAll;
    d.Ytest = YtestAll;
    
    std::cout << "MNIST cargado: Train=" << d.Xtrain.size() 
              << " Val=" << d.Xval.size() 
              << " Test=" << d.Xtest.size() << "\n\n";
    
    return d;
}

// ============================================================================
// Estructura para guardar información del mejor modelo
// ============================================================================
struct BestModel {
    double trainAcc = 0.0;
    double valAcc = 0.0;
    double testAcc = 0.0;
    std::string filename;
    int expNum = 0;
    std::vector<int> architecture;
    std::string activation;
    std::string phase;
    MLP* model = nullptr;
};

void runExperiments(const std::string& datasetName, const std::string& dataPath,
                   const std::string& resultsFile, double trainRatio, double valRatio) {
    
    Dataset dataset;
    if (datasetName == "iris") {
        dataset = loadIris(dataPath, trainRatio, valRatio);
    } else if (datasetName == "cancer") {
        dataset = loadCancer(dataPath, trainRatio, valRatio);
    } else if (datasetName == "wine") {
        dataset = loadWine(dataPath, trainRatio, valRatio);
    } else if (datasetName == "mnist") {
        std::string trainPath = "data/MNIST/train.csv";
        std::string testPath = "data/MNIST/test.csv";
        dataset = loadMNIST(trainPath, testPath, trainRatio, valRatio);
    } else {
        throw std::runtime_error("Dataset no soportado: " + datasetName);
    }
    
    createDirectory("models");
    createDirectory("models/mlp");
    
    std::ofstream results(resultsFile, std::ios::app);
    if (!results) throw std::runtime_error("No se pudo abrir " + resultsFile);
    
    results << "\n\n================================================\n";
    results << "DATASET: " << dataset.name << "\n";
    results << "Train: " << dataset.Xtrain.size() << " | Val: " << dataset.Xval.size() 
            << " | Test: " << dataset.Xtest.size() << "\n";
    results << "Input Features: " << dataset.inputSize << " | Output Classes: " << dataset.outputSize << "\n";
    results << "================================================\n\n";
    
    // Arquitecturas específicas por dataset
    std::vector<std::vector<int>> architectures;
    int maxEpochs = 100;
    
    if (datasetName == "mnist") {
        // MNIST: redes más grandes
        architectures = {
            {784, 128, 10},
            {784, 256, 10},
            {784, 512, 10},
            {784, 256, 128, 10},
            {784, 512, 256, 10}
        };
        maxEpochs = 30;  // Menos épocas para MNIST (tarda más por muestra)
    } else {
        // Iris, Cancer, Wine: redes estándar
        architectures = {
            {static_cast<int>(dataset.inputSize), 10, static_cast<int>(dataset.outputSize)},
            {static_cast<int>(dataset.inputSize), 20, static_cast<int>(dataset.outputSize)},
            {static_cast<int>(dataset.inputSize), 50, static_cast<int>(dataset.outputSize)},
            {static_cast<int>(dataset.inputSize), 10, 10, static_cast<int>(dataset.outputSize)},
            {static_cast<int>(dataset.inputSize), 20, 10, static_cast<int>(dataset.outputSize)},
            {static_cast<int>(dataset.inputSize), 50, 20, static_cast<int>(dataset.outputSize)},
            {static_cast<int>(dataset.inputSize), 100, 50, static_cast<int>(dataset.outputSize)}
        };
    }
    
    std::vector<ActivationType> activations = {
        ActivationType::SIGMOID,
        ActivationType::TANH,
        ActivationType::RELU,
        ActivationType::LEAKY_RELU
    };
    
    int expNum = 0;
    BestModel bestModel;
    
    // ========================================================================
    // FASE 1: Forward Propagation Only
    // ========================================================================
    results << "===== FASE 1: Forward Propagation Only =====\n\n";
    
    for (auto& arch : architectures) {
        for (auto& act : activations) {
            expNum++;
            std::string actName = ActivationFunctions::toString(act);
            
            std::cout << "\n[" << expNum << "] Testing: ";
            for (int l : arch) std::cout << l << " ";
            std::cout << "with " << actName << "\n";
            
            MLPConfig cfg;
            cfg.layerSizes = arch;
            cfg.activation = act;
            cfg.maxEpochs = 0;
            cfg.verbose = false;
            
            MLP* model = new MLP(cfg);
            
            double trainAcc = model->evaluate(dataset.Xtrain, dataset.Ytrain);
            double valAcc = model->evaluate(dataset.Xval, dataset.Yval);
            double testAcc = model->evaluate(dataset.Xtest, dataset.Ytest);
            
            if (testAcc > bestModel.testAcc) {
                if (bestModel.model) delete bestModel.model;
                bestModel.model = model;
                bestModel.trainAcc = trainAcc;
                bestModel.valAcc = valAcc;
                bestModel.testAcc = testAcc;
                bestModel.expNum = expNum;
                bestModel.architecture = arch;
                bestModel.activation = actName;
                bestModel.phase = "forward";
                bestModel.filename = generateModelFilename(dataset.name, arch, actName, "forward", expNum);
            } else {
                delete model;
            }
            
            results << "Exp " << expNum << " | ";
            for (int l : arch) results << l << "-";
            results << " | " << actName << "\n";
            results << "  Train: " << std::fixed << std::setprecision(2) << trainAcc << "% | ";
            results << "Val: " << valAcc << "% | ";
            results << "Test: " << testAcc << "%\n\n";
        }
    }
    
    // ========================================================================
    // FASE 2: Con Backpropagation
    // ========================================================================
    results << "\n===== FASE 2: Con Backpropagation =====\n\n";
    
    for (auto& arch : architectures) {
        for (auto& act : activations) {
            expNum++;
            std::string actName = ActivationFunctions::toString(act);
            
            std::cout << "\n[" << expNum << "] Training: ";
            for (int l : arch) std::cout << l << " ";
            std::cout << "with " << actName << "\n";
            
            MLPConfig cfg;
            cfg.layerSizes = arch;
            cfg.activation = act;
            cfg.learningRate = 0.01;
            cfg.maxEpochs = maxEpochs;
            cfg.batchSize = 32;
            cfg.verbose = true;
            cfg.printEvery = (datasetName == "mnist") ? 2 : 10;
            
            MLP* model = new MLP(cfg);
            model->train(dataset.Xtrain, dataset.Ytrain, dataset.Xval, dataset.Yval);
            
            double trainAcc = model->evaluate(dataset.Xtrain, dataset.Ytrain);
            double valAcc = model->evaluate(dataset.Xval, dataset.Yval);
            double testAcc = model->evaluate(dataset.Xtest, dataset.Ytest);
            
            if (testAcc > bestModel.testAcc) {
                if (bestModel.model) delete bestModel.model;
                bestModel.model = model;
                bestModel.trainAcc = trainAcc;
                bestModel.valAcc = valAcc;
                bestModel.testAcc = testAcc;
                bestModel.expNum = expNum;
                bestModel.architecture = arch;
                bestModel.activation = actName;
                bestModel.phase = "backprop";
                bestModel.filename = generateModelFilename(dataset.name, arch, actName, "backprop", expNum);
            } else {
                delete model;
            }
            
            results << "Exp " << expNum << " | ";
            for (int l : arch) results << l << "-";
            results << " | " << actName << "\n";
            results << "  Train: " << trainAcc << "% | ";
            results << "Val: " << valAcc << "% | ";
            results << "Test: " << testAcc << "%\n\n";
            
            std::cout << "  Results: Train=" << trainAcc << "% Val=" << valAcc 
                     << "% Test=" << testAcc << "%\n";
        }
    }
    
    // ========================================================================
    // FASE 3: Con Dropout (solo 2 arquitecturas)
    // ========================================================================
    results << "\n===== FASE 3: Con Regularización (Dropout) =====\n\n";
    
    std::vector<double> dropoutRates = {0.2, 0.3, 0.5};
    size_t numArchs = std::min(size_t(2), architectures.size());
    
    for (size_t i = 0; i < numArchs; ++i) {
        auto& arch = architectures[i];
        for (auto& rate : dropoutRates) {
            expNum++;
            std::cout << "\n[" << expNum << "] Training with Dropout=" << rate << "\n";
            
            MLPConfig cfg;
            cfg.layerSizes = arch;
            cfg.activation = ActivationType::RELU;
            cfg.learningRate = 0.01;
            cfg.maxEpochs = maxEpochs;
            cfg.batchSize = 32;
            cfg.useDropout = true;
            cfg.dropoutRate = rate;
            cfg.verbose = false;
            
            MLP* model = new MLP(cfg);
            model->train(dataset.Xtrain, dataset.Ytrain, dataset.Xval, dataset.Yval);
            
            double trainAcc = model->evaluate(dataset.Xtrain, dataset.Ytrain);
            double valAcc = model->evaluate(dataset.Xval, dataset.Yval);
            double testAcc = model->evaluate(dataset.Xtest, dataset.Ytest);
            
            std::stringstream ss;
            ss << "dropout" << std::fixed << std::setprecision(1) << rate;
            
            if (testAcc > bestModel.testAcc) {
                if (bestModel.model) delete bestModel.model;
                bestModel.model = model;
                bestModel.trainAcc = trainAcc;
                bestModel.valAcc = valAcc;
                bestModel.testAcc = testAcc;
                bestModel.expNum = expNum;
                bestModel.architecture = arch;
                bestModel.activation = "RELU";
                bestModel.phase = ss.str();
                bestModel.filename = generateModelFilename(dataset.name, arch, "RELU", ss.str(), expNum);
            } else {
                delete model;
            }
            
            results << "Exp " << expNum << " | Dropout=" << rate << "\n";
            results << "  Train: " << trainAcc << "% | ";
            results << "Val: " << valAcc << "% | ";
            results << "Test: " << testAcc << "%\n\n";
        }
    }
    
    // ========================================================================
    // FASE 4: Con L2 Regularization
    // ========================================================================
    results << "\n===== FASE 4: Con L2 Regularization =====\n\n";
    
    std::vector<double> l2Lambdas = {0.001, 0.01, 0.1};
    
    for (size_t i = 0; i < numArchs; ++i) {
        auto& arch = architectures[i];
        for (auto& lambda : l2Lambdas) {
            expNum++;
            std::cout << "\n[" << expNum << "] Training with L2 lambda=" << lambda << "\n";
            
            MLPConfig cfg;
            cfg.layerSizes = arch;
            cfg.activation = ActivationType::RELU;
            cfg.learningRate = 0.01;
            cfg.maxEpochs = maxEpochs;
            cfg.batchSize = 32;
            cfg.useL2 = true;
            cfg.l2Lambda = lambda;
            cfg.verbose = false;
            
            MLP* model = new MLP(cfg);
            model->train(dataset.Xtrain, dataset.Ytrain, dataset.Xval, dataset.Yval);
            
            double trainAcc = model->evaluate(dataset.Xtrain, dataset.Ytrain);
            double valAcc = model->evaluate(dataset.Xval, dataset.Yval);
            double testAcc = model->evaluate(dataset.Xtest, dataset.Ytest);
            
            std::stringstream ss;
            ss << "l2_" << lambda;
            
            if (testAcc > bestModel.testAcc) {
                if (bestModel.model) delete bestModel.model;
                bestModel.model = model;
                bestModel.trainAcc = trainAcc;
                bestModel.valAcc = valAcc;
                bestModel.testAcc = testAcc;
                bestModel.expNum = expNum;
                bestModel.architecture = arch;
                bestModel.activation = "RELU";
                bestModel.phase = ss.str();
                bestModel.filename = generateModelFilename(dataset.name, arch, "RELU", ss.str(), expNum);
            } else {
                delete model;
            }
            
            results << "Exp " << expNum << " | L2 lambda=" << lambda << "\n";
            results << "  Train: " << trainAcc << "% | ";
            results << "Val: " << valAcc << "% | ";
            results << "Test: " << testAcc << "%\n\n";
        }
    }
    
    // ========================================================================
    // FASE 5: Con Early Stopping (solo 3 arquitecturas)
    // ========================================================================
    results << "\n===== FASE 5: Con Early Stopping =====\n\n";
    
    for (size_t i = 0; i < std::min(size_t(3), architectures.size()); ++i) {
        auto& arch = architectures[i];
        expNum++;
        std::cout << "\n[" << expNum << "] Training with Early Stopping\n";
        
        MLPConfig cfg;
        cfg.layerSizes = arch;
        cfg.activation = ActivationType::RELU;
        cfg.learningRate = 0.01;
        cfg.maxEpochs = 150;
        cfg.batchSize = 32;
        cfg.useEarlyStopping = true;
        cfg.patience = 15;
        cfg.minDelta = 0.001;
        cfg.verbose = true;
        cfg.printEvery = 5;
        
        MLP* model = new MLP(cfg);
        model->train(dataset.Xtrain, dataset.Ytrain, dataset.Xval, dataset.Yval);
        
        double trainAcc = model->evaluate(dataset.Xtrain, dataset.Ytrain);
        double valAcc = model->evaluate(dataset.Xval, dataset.Yval);
        double testAcc = model->evaluate(dataset.Xtest, dataset.Ytest);
        
        if (testAcc > bestModel.testAcc) {
            if (bestModel.model) delete bestModel.model;
            bestModel.model = model;
            bestModel.trainAcc = trainAcc;
            bestModel.valAcc = valAcc;
            bestModel.testAcc = testAcc;
            bestModel.expNum = expNum;
            bestModel.architecture = arch;
            bestModel.activation = "RELU";
            bestModel.phase = "earlystop";
            bestModel.filename = generateModelFilename(dataset.name, arch, "RELU", "earlystop", expNum);
        } else {
            delete model;
        }
        
        results << "Exp " << expNum << " | Early Stopping (stopped at epoch " 
                << model->getBestEpoch() + 1 << ")\n";
        results << "  Train: " << trainAcc << "% | ";
        results << "Val: " << valAcc << "% | ";
        results << "Test: " << testAcc << "%\n\n";
    }
    
    // ========================================================================
    // GUARDAR MEJOR MODELO
    // ========================================================================
    if (bestModel.model) {
        bestModel.model->save(bestModel.filename);
        std::cout << "\n✓ Mejor modelo guardado: " << bestModel.filename << "\n";
    }
    
    results << "\n========================================\n";
    results << "MEJOR MODELO PARA " << dataset.name << "\n";
    results << "========================================\n";
    results << "Experimento: " << bestModel.expNum << "\n";
    results << "Arquitectura: ";
    for (size_t i = 0; i < bestModel.architecture.size(); ++i) {
        results << bestModel.architecture[i];
        if (i < bestModel.architecture.size() - 1) results << "-";
    }
    results << "\n";
    results << "Activación: " << bestModel.activation << "\n";
    results << "Fase: " << bestModel.phase << "\n";
    results << "Archivo: " << bestModel.filename << "\n";
    results << "Train Accuracy: " << std::fixed << std::setprecision(2) << bestModel.trainAcc << "%\n";
    results << "Val Accuracy: " << bestModel.valAcc << "%\n";
    results << "Test Accuracy: " << bestModel.testAcc << "%\n\n";
    
    results.close();
    
    std::cout << "\n\n========================================\n";
    std::cout << "EXPERIMENTOS COMPLETADOS\n";
    std::cout << "========================================\n";
    std::cout << "Mejor modelo: " << bestModel.filename << "\n";
    std::cout << "  Train: " << bestModel.trainAcc << "%\n";
    std::cout << "  Val:   " << bestModel.valAcc << "%\n";
    std::cout << "  Test:  " << bestModel.testAcc << "%\n\n";
    
    if (bestModel.model) delete bestModel.model;
}

void usage() {
    std::cout << "USO: RunMLP --dataset <nombre> [opciones]\n\n";
    std::cout << "Datasets disponibles:\n";
    std::cout << "  iris    - Iris Flowers (150 samples, 4 features, 3 classes)\n";
    std::cout << "  cancer  - Breast Cancer (569 samples, 30 features, 2 classes)\n";
    std::cout << "  wine    - Wine Quality (1599 samples, 11 features, 2 classes)\n";
    std::cout << "  mnist   - MNIST Digits (60000 samples, 784 features, 10 classes)\n\n";
    std::cout << "Opciones:\n";
    std::cout << "  --results <path>       Archivo de resultados (default: results/mlp_results.txt)\n";
    std::cout << "  --train-split <0-1>    Ratio de train (default: 0.7)\n";
    std::cout << "  --val-split <0-1>      Ratio de val (default: 0.15)\n";
    std::cout << "  --help                 Mostrar esta ayuda\n\n";
    std::cout << "Ejemplos:\n";
    std::cout << "  ./RunMLP --dataset iris\n";
    std::cout << "  ./RunMLP --dataset mnist\n";
    std::cout << "  ./RunMLP --dataset cancer --train-split 0.8\n";
}

int main(int argc, char** argv) {
    std::string dataset, dataPath;
    std::string resultsFile = "results/mlp_results.txt";
    double trainSplit = 0.7, valSplit = 0.15;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) dataset = argv[++i];
        else if (arg == "--data" && i + 1 < argc) dataPath = argv[++i];
        else if (arg == "--results" && i + 1 < argc) resultsFile = argv[++i];
        else if (arg == "--train-split" && i + 1 < argc) trainSplit = std::stod(argv[++i]);
        else if (arg == "--val-split" && i + 1 < argc) valSplit = std::stod(argv[++i]);
        else if (arg == "--help") { usage(); return 0; }
    }
    
    if (dataset.empty()) {
        usage();
        return 1;
    }
    
    // Validar dataset
    if (dataset != "iris" && dataset != "cancer" && dataset != "wine" && dataset != "mnist") {
        std::cerr << "ERROR: Dataset '" << dataset << "' no soportado\n\n";
        usage();
        return 1;
    }
    
    // Validar archivos
    if (dataPath.empty()) {
        if (dataset == "iris") dataPath = "data/Iris.csv";
        else if (dataset == "cancer") dataPath = "data/cancermama.csv";
        else if (dataset == "wine") dataPath = "data/winequality-red.csv";
        else if (dataset == "mnist") dataPath = "data/MNIST/train.csv";
    }
    
    if (!fileExists(dataPath)) {
        std::cerr << "ERROR: No se encontró el archivo: " << dataPath << "\n";
        return 1;
    }
    
    if (dataset == "mnist") {
        std::string testPath = "data/MNIST/test.csv";
        if (!fileExists(testPath)) {
            std::cerr << "ERROR: No se encontró " << testPath << "\n";
            return 1;
        }
    }
    
    try {
        runExperiments(dataset, dataPath, resultsFile, trainSplit, valSplit);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}