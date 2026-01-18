#include "MLP.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <iomanip>
#include <sys/stat.h>
#include <cmath>
#include <map>

// Crea un directorio si no existe
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

/* Genera un nombre de archivo para guardar modelos entrenados dependiendo de:
    - Dataset
    - Número de experimento
    - Arquitectura
    - Función de activación
    - Fase (forward, backward, earlystop ...)
*/
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

// Divide dataset en train/val/test con shuffle de forma aleatoria
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

// FUNCIONES DE CARGA DE DATASETS

struct Dataset {
    MatDouble_t Xtrain, Ytrain, Xval, Yval, Xtest, Ytest;
    size_t inputSize, outputSize;
    std::string name;
};

// Cargar dataset Iris desde CSV
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


// Carga dataset Cancer desde CSV
// Aplica normalización z-score a los features porque es crítico para este dataset ya que las features tienen escalas muy diferentes
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

    std::cout << "Cancer cargado: " << Xall.size() << " muestras\n";

    // Normalización (z-score)
    if (!Xall.empty()) {
        const size_t numFeatures = 30;
        std::vector<double> means(numFeatures, 0.0);
        std::vector<double> stds(numFeatures, 0.0);

        // Calcular medias
        for (const auto& sample : Xall) {
            for (size_t j = 0; j < numFeatures; ++j) {
                means[j] += sample[j];
            }
        }
        for (auto& m : means) m /= Xall.size();

        // Calcular desviaciones estándar
        for (const auto& sample : Xall) {
            for (size_t j = 0; j < numFeatures; ++j) {
                double diff = sample[j] - means[j];
                stds[j] += diff * diff;
            }
        }
        for (auto& s : stds) s = std::sqrt(s / Xall.size());

        // Normalizar (z-score)
        for (auto& sample : Xall) {
            for (size_t j = 0; j < numFeatures; ++j) {
                if (stds[j] > 1e-8) {
                    sample[j] = (sample[j] - means[j]) / stds[j];
                }
            }
        }

        std::cout << "  Features normalizadas (z-score)\n";
    }

    Dataset d;
    d.name = "cancer";
    d.inputSize = 30;
    d.outputSize = 1;
    shuffleSplit3(Xall, Yall, trainRatio, valRatio,
                  d.Xtrain, d.Ytrain, d.Xval, d.Yval, d.Xtest, d.Ytest);

    std::cout << "  Split: Train=" << d.Xtrain.size()
              << " Val=" << d.Xval.size()
              << " Test=" << d.Xtest.size() << "\n\n";

    return d;
}

// Carga dataset Wine desde CSV
// Convierte quality en clasificación binaria (good vs bad)
static Dataset loadWine(const std::string& path, double trainRatio, double valRatio) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("No se pudo abrir " + path);

    std::string line;
    std::getline(file, line);  // Skip header

    // DEBUG: Ver header
    std::cout << "Wine header: " << line.substr(0, 80) << "...\n";

    MatDouble_t Xall, Yall;
    int countBad = 0, countGood = 0;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string v;
        std::vector<double> row;

        // Leer las 11 primeras columnas (features)
        for (int i = 0; i < 11; ++i) {
            if (!std::getline(ss, v, ',')) break;  // ← IMPORTANTE: coma, no punto y coma
            row.push_back(std::stod(v));
        }

        // Leer columna 12 (quality)
        std::getline(ss, v, ',');
        int quality = std::stoi(v);

        // Clasificación binaria: good (>=6) vs bad (<6)
        double label = (quality >= 6) ? 1.0 : 0.0;

        if (label == 1.0) countGood++;
        else countBad++;

        if (row.size() == 11) {
            Xall.push_back(row);
            Yall.push_back({label});
        }
    }

    file.close();

    std::cout << "Wine cargado:\n";
    std::cout << "  Total: " << Xall.size() << " muestras\n";
    std::cout << "  Bad (quality<6): " << countBad << " (" << (100.0*countBad/Xall.size()) << "%)\n";
    std::cout << "  Good (quality>=6): " << countGood << " (" << (100.0*countGood/Xall.size()) << "%)\n";

    // Normalización (z-score)
    if (!Xall.empty()) {
        size_t numFeatures = Xall[0].size();

        // Calcular media y desviación estándar
        std::vector<double> means(numFeatures, 0.0);
        std::vector<double> stds(numFeatures, 0.0);

        for (const auto& sample : Xall) {
            for (size_t j = 0; j < numFeatures; ++j) {
                means[j] += sample[j];
            }
        }
        for (auto& m : means) m /= Xall.size();

        for (const auto& sample : Xall) {
            for (size_t j = 0; j < numFeatures; ++j) {
                double diff = sample[j] - means[j];
                stds[j] += diff * diff;
            }
        }
        for (auto& s : stds) s = std::sqrt(s / Xall.size());

        // Normalizar (z-score)
        for (auto& sample : Xall) {
            for (size_t j = 0; j < numFeatures; ++j) {
                if (stds[j] > 1e-8) {
                    sample[j] = (sample[j] - means[j]) / stds[j];
                }
            }
        }

        std::cout << "  Features normalizadas (z-score)\n";
    }

    Dataset d;
    d.name = "wine";
    d.inputSize = 11;
    d.outputSize = 1;
    shuffleSplit3(Xall, Yall, trainRatio, valRatio,
                  d.Xtrain, d.Ytrain, d.Xval, d.Yval, d.Xtest, d.Ytest);

    std::cout << "  Split: Train=" << d.Xtrain.size()
              << " Val=" << d.Xval.size()
              << " Test=" << d.Xtest.size() << "\n\n";

    return d;
}

// Carga dataset MNIST desde CSV
// Normaliza píxeles a [0,1] y aplica one-hot encoding a labels
static Dataset loadMNIST(const std::string& trainPath,
                        double trainRatio, double valRatio) {
    std::cout << "Cargando MNIST...\n";

    Dataset d;
    d.name = "mnist";
    d.inputSize = 784;
    d.outputSize = 10;

    // Cargamos solo train.csv (que SÍ tiene labels) a diferencia de test.csv
    std::ifstream trainFile(trainPath);
    if (!trainFile) throw std::runtime_error("No se pudo abrir " + trainPath);

    std::string line;
    std::getline(trainFile, line);  // Skip header

    MatDouble_t Xall, Yall;
    int count = 0;

    std::cout << "  Leyendo MNIST..." << std::flush;

    while (std::getline(trainFile, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string val;

        // Leer label (primera columna en train.csv)
        std::getline(ss, val, ',');
        int label = std::stoi(val);

        // One-hot encoding
        VecDouble_t oh(10, 0.0);
        oh[label] = 1.0;

        // Leer 784 píxeles
        VecDouble_t pixels;
        while (std::getline(ss, val, ',')) {
            if (!val.empty()) {
                pixels.push_back(std::stod(val) / 255.0);
            }
        }

        if (pixels.size() == 784) {
            Xall.push_back(pixels);
            Yall.push_back(oh);
            count++;
        }

        if (count % 2000 == 0) std::cout << "." << std::flush;
    }

    trainFile.close();
    std::cout << " " << count << " muestras\n";

    // Tenemos que dividir en Train / Val / Test
    std::vector<size_t> idx(Xall.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::mt19937 g(std::random_device{}());
    std::shuffle(idx.begin(), idx.end(), g);

    size_t trainSize = static_cast<size_t>(idx.size() * trainRatio);
    size_t valSize   = static_cast<size_t>(idx.size() * valRatio);

    for (size_t i = 0; i < idx.size(); ++i) {
        size_t j = idx[i];
        if (i < trainSize) {
            d.Xtrain.push_back(Xall[j]);
            d.Ytrain.push_back(Yall[j]);
        } else if (i < trainSize + valSize) {
            d.Xval.push_back(Xall[j]);
            d.Yval.push_back(Yall[j]);
        } else {
            d.Xtest.push_back(Xall[j]);
            d.Ytest.push_back(Yall[j]);
        }
    }

    std::cout << "MNIST cargado: Train=" << d.Xtrain.size()
              << " Val=" << d.Xval.size()
              << " Test=" << d.Xtest.size() << "\n\n";

    return d;
}


// Balanceo extra para el dataset de Atari Assault
// El balanceo consiste principalmente en ajustar el número de muestras por clase (acción) a la tercera clase más común
// De esta forma evitamos que las clases más comunes (sobre todo la clase NOOP) dominen el entrenamiento
static void balanceAtariData(MatDouble_t& X, MatDouble_t& Y, size_t minSamplesPerClass = 5) {
    std::cout << "\n=== BALANCEANDO DATASET DE ATARI (modo suave) ===\n";

    // Paso 1: Filtrar acciones inválidas [0,1,1] (LEFT + RIGHT simultáneamente)
    MatDouble_t X_filtered, Y_filtered;
    int invalidCount = 0;

    for (size_t i = 0; i < Y.size(); ++i) {
        if (Y[i][1] == 1.0 && Y[i][2] == 1.0) {
            invalidCount++;
            continue;
        }
        X_filtered.push_back(X[i]);
        Y_filtered.push_back(Y[i]);
    }

    if (invalidCount > 0) {
        std::cout << "  ⚠️  Eliminadas " << invalidCount
                  << " muestras con conflicto [0,1,1]\n";
    }

    // Paso 2: Agrupar por clase (acción)
    std::map<VecDouble_t, std::vector<size_t>> classIndices;

    for (size_t i = 0; i < Y_filtered.size(); ++i) {
        classIndices[Y_filtered[i]].push_back(i);
    }

    // Función helper para nombres de acciones
    auto getActionName = [](const VecDouble_t& action) -> std::string {
        if (action[0] == 1 && action[1] == 1) return "LEFTFIRE";
        else if (action[0] == 1 && action[2] == 1) return "RIGHTFIRE";
        else if (action[0] == 1) return "FIRE";
        else if (action[1] == 1) return "LEFT";
        else if (action[2] == 1) return "RIGHT";
        else return "NOOP";
    };

    // Paso 3: Mostrar distribución antes del balance
    std::cout << "\n  Distribución ANTES del balance:\n";
    std::vector<std::pair<std::string, size_t>> classSizes;

    for (const auto& [action, indices] : classIndices) {
        std::string name = getActionName(action);
        size_t count = indices.size();
        double pct = (100.0 * count) / Y_filtered.size();

        std::cout << "    " << std::left << std::setw(12) << name
                  << ": " << std::right << std::setw(5) << count
                  << " (" << std::fixed << std::setprecision(1) << pct << "%)\n";

        classSizes.push_back({name, count});
    }

    // Paso 4: Filtrar clases con muy pocas muestras
    std::vector<VecDouble_t> validClasses;
    std::cout << "\n  Clases válidas (>= " << minSamplesPerClass << " muestras):\n";

    for (const auto& [action, indices] : classIndices) {
        std::string name = getActionName(action);

        if (indices.size() >= minSamplesPerClass) {
            validClasses.push_back(action);
            std::cout << "    ✓ " << std::left << std::setw(12) << name
                      << ": " << indices.size() << " muestras\n";
        } else {
            std::cout << "    ✗ " << std::left << std::setw(12) << name
                      << ": " << indices.size() << " muestras (eliminada)\n";
        }
    }

    if (validClasses.empty()) {
        throw std::runtime_error("No hay clases válidas después del filtrado");
    }

    // Paso 5: Calcular tamaño objetivo (tercera clase más común)
    std::vector<size_t> sizes;
    for (const auto& action : validClasses) {
        sizes.push_back(classIndices[action].size());
    }
    std::sort(sizes.begin(), sizes.end(), std::greater<size_t>());  // Ordenar descendente

    // Elegir el tamaño objetivo:
    // - Si hay >= 3 clases: usar la 3ª más común (percentil ~50%)
    // - Si hay 2 clases: usar la 2ª (la mínima)
    // - Si hay 1 clase: no balancear
    size_t targetSize;
    if (sizes.size() >= 3) {
        targetSize = sizes[2];  // Tercera más común
        std::cout << "\n  → Usando tamaño de la 3ª clase más común: " << targetSize << " muestras\n";
    } else if (sizes.size() == 2) {
        targetSize = sizes[1];  // La mínima de las 2
        std::cout << "\n  → Usando tamaño de la clase minoritaria: " << targetSize << " muestras\n";
    } else {
        std::cout << "\n  → Solo 1 clase válida, no se balancea\n";
        return;  // No balancear si solo hay 1 clase
    }

    std::cout << "  → Total clases válidas: " << validClasses.size() << "\n";

    // Paso 6: Submuestrear cada clase a 'targetSize' (o mantener si es menor)
    MatDouble_t X_balanced, Y_balanced;
    std::mt19937 g(std::random_device{}());

    for (const auto& action : validClasses) {
        auto indices = classIndices[action];  // Copia
        std::shuffle(indices.begin(), indices.end(), g);

        // Tomar hasta 'targetSize' muestras (o todas si son menos)
        size_t numToTake = std::min(targetSize, indices.size());

        for (size_t i = 0; i < numToTake; ++i) {
            size_t idx = indices[i];
            X_balanced.push_back(X_filtered[idx]);
            Y_balanced.push_back(Y_filtered[idx]);
        }
    }

    // Paso 7: Mezclar el dataset balanceado
    std::vector<size_t> allIndices(X_balanced.size());
    std::iota(allIndices.begin(), allIndices.end(), 0);
    std::shuffle(allIndices.begin(), allIndices.end(), g);

    MatDouble_t X_shuffled, Y_shuffled;
    for (size_t idx : allIndices) {
        X_shuffled.push_back(X_balanced[idx]);
        Y_shuffled.push_back(Y_balanced[idx]);
    }

    // Paso 8: Reemplazar datos originales
    X = X_shuffled;
    Y = Y_shuffled;

    // Paso 9: Mostrar distribución después del balance
    std::cout << "\n  Distribución DESPUÉS del balance:\n";
    std::map<VecDouble_t, int> newCounts;
    for (const auto& y : Y) {
        newCounts[y]++;
    }

    for (const auto& [action, count] : newCounts) {
        std::string name = getActionName(action);
        double pct = (100.0 * count) / Y.size();
        std::cout << "    " << std::left << std::setw(12) << name
                  << ": " << std::right << std::setw(5) << count
                  << " (" << std::fixed << std::setprecision(1) << pct << "%)\n";
    }

    size_t originalSize = Y_filtered.size();
    size_t finalSize = Y.size();
    double reductionPct = 100.0 * (1.0 - (double)finalSize / originalSize);

    std::cout << "\n  → Total ANTES: " << originalSize << " muestras\n";
    std::cout << "  → Total DESPUÉS: " << finalSize << " muestras\n";
    std::cout << "  → Reducción: " << std::fixed << std::setprecision(1)
              << reductionPct << "%\n";
}


// Carga dataset Atari Assault desde CSV
// Lee el CSV, normaliza features de RAM a [0,1], aplica balanceo y split
static Dataset loadAtari(const std::string& path, double trainRatio, double valRatio) {
    std::cout << "Cargando Atari Assault...\n";

    std::ifstream file(path);
    if (!file) throw std::runtime_error("No se pudo abrir " + path);

    std::string line;
    MatDouble_t Xall, Yall;

    const int NUM_RAM_FEATURES = 80;
    const int NUM_ACTIONS = 3;

    int lineCount = 0;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string value;
        std::vector<double> features;
        std::vector<double> actions(NUM_ACTIONS, 0.0);

        // Leer features de RAM
        for (int i = 0; i < NUM_RAM_FEATURES; ++i) {
            if (!std::getline(ss, value, ';')) break;
            features.push_back(std::stod(value) / 255.0);
        }

        // Leer 3 acciones
        for (int i = 0; i < NUM_ACTIONS; ++i) {
            if (!std::getline(ss, value, ';')) break;
            actions[i] = std::stod(value);
        }

        if (features.size() == NUM_RAM_FEATURES && actions.size() == NUM_ACTIONS) {
            Xall.push_back(features);
            Yall.push_back(actions);
            lineCount++;
        }
    }

    file.close();

    std::cout << "  Total muestras cargadas: " << lineCount << "\n";

    balanceAtariData(Xall, Yall, 5);  // Mínimo 5 muestras por clase

    // Split (train/val/test)
    Dataset d;
    d.name = "atari";
    d.inputSize = NUM_RAM_FEATURES;
    d.outputSize = NUM_ACTIONS;

    shuffleSplit3(Xall, Yall, trainRatio, valRatio,
                  d.Xtrain, d.Ytrain, d.Xval, d.Yval, d.Xtest, d.Ytest);

    std::cout << "\n  Split final: Train=" << d.Xtrain.size()
              << " Val=" << d.Xval.size()
              << " Test=" << d.Xtest.size() << "\n\n";

    return d;
}


// Estructura para guardar información del mejor modelo
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


/* Función principal para ejecutar experimentos con MLP en un dataset dado. Está dividida en varias fases:
    1. Forward propagation only (sin entrenamiento): crea un modelo sin entrenar (pesos aleatorios)
    2. Backward propagation (entrenamiento estándar): entrena el modelo con backpropagation y RELU
    3. Regularización con Dropout: entrena con dropour (0.3 y 0.5) para prevenir overfitting
    4. Regularización L2 (lambda=0.01 y 0.1): entrena con regularización L2 para prevenir overfitting
    5. Early Stopping: entrena con early stopping basado en validación
    6. Mejor modelo global: guarda el mejor modelo obtenido en todas las fases
*/
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
        std::string trainPath = "data/MNIST/train_small.csv";
        dataset = loadMNIST(trainPath, trainRatio, valRatio);
    } else if (datasetName == "atari") {
        dataset = loadAtari(dataPath, trainRatio, valRatio);
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
    results << "Input: " << dataset.inputSize << " | Output: " << dataset.outputSize << "\n";
    results << "================================================\n\n";

    // Ajuste de arquitecturas y parámetros según dataset
    std::vector<std::vector<int>> architectures;
    int maxEpochs = 100;

    if (datasetName == "mnist") {
        architectures = {
            {784, 128, 10},
            {784, 256, 128, 10},
            {784, 512, 256, 10}
        };
        maxEpochs = 30;
    } else if (datasetName == "atari") {
        // Arquitecturas específicas para Atari
        architectures = {
            {80, 128, 64, 3},      // Mediana
            {80, 256, 128, 3},     // Grande
        };
        maxEpochs = 200;
    } else {
        int in = static_cast<int>(dataset.inputSize);
        int out = static_cast<int>(dataset.outputSize);

        architectures = {
            {in, 20, out},
            {in, 50, 20, out},
            {in, 100, 50, out}
        };
    }


    ActivationType activation = ActivationType::RELU;
    std::string actName = "RELU";

    int expNum = 0;
    BestModel bestModel;

    // Fase 1
    results << "===== FASE 1: Forward Propagation (sin entrenar) =====\n\n";

    // Solo probar 1 arquitectura para ver pesos aleatorios
    {
        expNum++;
        auto& arch = architectures[0];  // Arquitectura pequeña

        std::cout << "\n[" << expNum << "] Forward Only: ";
        for (int l : arch) std::cout << l << " ";
        std::cout << "\n";

        MLPConfig cfg;
        cfg.layerSizes = arch;
        cfg.activation = activation;
        cfg.learningRate = 0.01;
        cfg.maxEpochs = 0;
        cfg.verbose = false;

        MLP* model = new MLP(cfg);

        double trainAcc = model->evaluate(dataset.Xtrain, dataset.Ytrain);
        double valAcc = model->evaluate(dataset.Xval, dataset.Yval);
        double testAcc = model->evaluate(dataset.Xtest, dataset.Ytest);

        // std::string currentFilename = generateModelFilename(dataset.name, arch, actName, "forward", expNum);
        // model->save(currentFilename);
        // std::cout << "  Modelo guardado: " << currentFilename << "\n";


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
        results << " | Forward Only (pesos aleatorios)\n";
        results << "  Train: " << std::fixed << std::setprecision(2) << trainAcc << "% | ";
        results << "Val: " << valAcc << "% | ";
        results << "Test: " << testAcc << "%\n\n";
    }

    // Fase 2
    results << "\n===== FASE 2: Backpropagation =====\n\n";

    for (auto& arch : architectures) {
        expNum++;

        std::cout << "\n[" << expNum << "] Training: ";
        for (int l : arch) std::cout << l << " ";
        std::cout << "\n";

        MLPConfig cfg;
        cfg.layerSizes = arch;
        cfg.activation = activation;
        cfg.learningRate = 0.01;
        cfg.maxEpochs = maxEpochs;
        cfg.batchSize = 32;
        cfg.verbose = true;
        cfg.printEvery = (datasetName == "mnist") ? 5 : 10;

        MLP* model = new MLP(cfg);
        model->train(dataset.Xtrain, dataset.Ytrain, dataset.Xval, dataset.Yval);

        double trainAcc = model->evaluate(dataset.Xtrain, dataset.Ytrain);
        double valAcc = model->evaluate(dataset.Xval, dataset.Yval);
        double testAcc = model->evaluate(dataset.Xtest, dataset.Ytest);


        // std::string currentFilename = generateModelFilename(dataset.name, arch, actName, "backprop", expNum);
        // model->save(currentFilename);
        // std::cout << "  Modelo guardado: " << currentFilename << "\n";


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
        results << " | Backpropagation\n";
        results << "  Train: " << trainAcc << "% | ";
        results << "Val: " << valAcc << "% | ";
        results << "Test: " << testAcc << "%\n\n";

        std::cout << "  Results: Train=" << trainAcc << "% Val=" << valAcc
                  << "% Test=" << testAcc << "%\n";
    }

    // Fase 3
    results << "\n===== FASE 3: Regularización (Dropout) =====\n\n";

    auto& arch_reg = architectures[1];  // Arquitectura mediana
    std::vector<double> dropoutRates = {0.3, 0.5};

    for (auto& rate : dropoutRates) {
        expNum++;

        std::cout << "\n[" << expNum << "] Dropout=" << rate << ": ";
        for (int l : arch_reg) std::cout << l << " ";
        std::cout << "\n";

        MLPConfig cfg;
        cfg.layerSizes = arch_reg;
        cfg.activation = activation;
        cfg.learningRate = 0.01;
        cfg.maxEpochs = maxEpochs;
        cfg.batchSize = 32;
        cfg.useDropout = true;
        cfg.dropoutRate = rate;
        cfg.verbose = true;
        cfg.printEvery = (datasetName == "mnist") ? 5 : 10;

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
            bestModel.architecture = arch_reg;
            bestModel.activation = actName;
            bestModel.phase = ss.str();
            bestModel.filename = generateModelFilename(dataset.name, arch_reg, actName, ss.str(), expNum);
        } else {
            delete model;
        }

        results << "Exp " << expNum << " | Dropout=" << rate << "\n";
        results << "  Train: " << trainAcc << "% | ";
        results << "Val: " << valAcc << "% | ";
        results << "Test: " << testAcc << "%\n\n";
    }

    // Fase 4
    results << "\n===== FASE 4: Regularización (L2) =====\n\n";

    std::vector<double> l2Lambdas = {0.01, 0.1};

    for (auto& lambda : l2Lambdas) {
        expNum++;

        std::cout << "\n[" << expNum << "] L2 lambda=" << lambda << "\n";

        MLPConfig cfg;
        cfg.layerSizes = arch_reg;
        cfg.activation = activation;
        cfg.learningRate = 0.01;
        cfg.maxEpochs = maxEpochs;
        cfg.batchSize = 32;
        cfg.useL2 = true;
        cfg.l2Lambda = lambda;
        cfg.verbose = true;
        cfg.printEvery = (datasetName == "mnist") ? 5 : 10;

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
            bestModel.architecture = arch_reg;
            bestModel.activation = actName;
            bestModel.phase = ss.str();
            bestModel.filename = generateModelFilename(dataset.name, arch_reg, actName, ss.str(), expNum);
        } else {
            delete model;
        }

        results << "Exp " << expNum << " | L2 lambda=" << lambda << "\n";
        results << "  Train: " << trainAcc << "% | ";
        results << "Val: " << valAcc << "% | ";
        results << "Test: " << testAcc << "%\n\n";
    }

    // Fase 5
    results << "\n===== FASE 5: Early Stopping =====\n\n";

    for (size_t i = 0; i < 2; ++i) {  // Pequeña y mediana
        auto& arch = architectures[i];
        expNum++;

        std::cout << "\n[" << expNum << "] Early Stopping: ";
        for (int l : arch) std::cout << l << " ";
        std::cout << "\n";

        MLPConfig cfg;
        cfg.layerSizes = arch;
        cfg.activation = activation;
        cfg.learningRate = 0.01;
        cfg.maxEpochs = 150;  // Más épocas para ver el early stopping
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
            bestModel.activation = actName;
            bestModel.phase = "earlystop";
            bestModel.filename = generateModelFilename(dataset.name, arch, actName, "earlystop", expNum);
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
    std::cout << "  atari   - Atari Assault (manual gameplay, 59 RAM features, 3 actions)\n\n";
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

    resultsFile = "results/mlp_" + dataset + ".txt";

    if (dataset.empty()) {
        usage();
        return 1;
    }

    // Validar dataset
    if (dataset != "iris" && dataset != "cancer" && dataset != "wine" && dataset != "mnist" && dataset != "atari") {
        std::cerr << "ERROR: Dataset '" << dataset << "' no soportado\n\n";
        usage();
        return 1;
    }

    // Validar archivos
    if (dataPath.empty()) {
        if (dataset == "iris") dataPath = "data/Iris.csv";
        else if (dataset == "cancer") dataPath = "data/cancermama.csv";
        else if (dataset == "wine") dataPath = "data/winequality-red.csv";
        else if (dataset == "mnist") dataPath = "data/MNIST/train_small.csv";
        else if (dataset == "atari") dataPath = "datasets_juntos.csv";
    }

    if (!fileExists(dataPath)) {
        std::cerr << "ERROR: No se encontró el archivo: " << dataPath << "\n";
        return 1;
    }


    try {
        runExperiments(dataset, dataPath, resultsFile, trainSplit, valSplit);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}