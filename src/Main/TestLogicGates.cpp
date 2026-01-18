#include "Perceptron.hpp"
#include <iostream>
#include <vector>

using Mat = std::vector<std::vector<double>>;

static double accuracy(const Perceptron& model, const Mat& X, const Mat& Y) {
    if (X.empty()) return 0.0;
    int ok = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        auto out = model.predict(X[i]);
        double pred = out[0] >= 0.0 ? 1.0 : -1.0;
        if (std::abs(pred - Y[i][0]) < 0.1) ok++;
    }
    return (static_cast<double>(ok) / X.size()) * 100.0;
}

int main() {
    // Datos para AND
    Mat X_and = {{0.0,0.0}, {0.0,1.0}, {1.0,0.0}, {1.0,1.0}};
    Mat Y_and = {{-1.0}, {-1.0}, {-1.0}, {1.0}};

    // Datos para OR
    Mat X_or = X_and;
    Mat Y_or = {{-1.0}, {1.0}, {1.0}, {1.0}};

    std::cout << "== Test Perceptron: AND ==\n";
    Perceptron p_and(2,1);
    p_and.train(X_and, Y_and, Mat(), Mat());
    std::cout << "Pesos AND:\n";
    p_and.save("models/perceptron_and_weights.txt");
    std::cout << "Acc train (AND): " << accuracy(p_and, X_and, Y_and) << "%\n";
    for (size_t i=0;i<X_and.size();++i) {
        auto out = p_and.predict(X_and[i]);
        double pred = out[0] >= 0.0 ? 1.0 : -1.0;
        std::cout << "  In: ("<<X_and[i][0]<<","<<X_and[i][1]<<") -> "<< pred << " (raw="<< out[0] <<")\n";
    }

    std::cout << "\n== Test Perceptron: OR ==\n";
    Perceptron p_or(2,1);
    p_or.train(X_or, Y_or, Mat(), Mat());
    p_or.save("models/perceptron_or_weights.txt");
    std::cout << "Acc train (OR): " << accuracy(p_or, X_or, Y_or) << "%\n";
    for (size_t i=0;i<X_or.size();++i) {
        auto out = p_or.predict(X_or[i]);
        double pred = out[0] >= 0.0 ? 1.0 : -1.0;
        std::cout << "  In: ("<<X_or[i][0]<<","<<X_or[i][1]<<") -> "<< pred << " (raw="<< out[0] <<")\n";
    }

    // Datos para XOR (no linealmente separable — el perceptrón simple debería fallar)
    Mat X_xor = X_and;
    Mat Y_xor = {{-1.0}, {1.0}, {1.0}, {-1.0}}; // XOR: 1 cuando inputs distintos

    std::cout << "\n== Test Perceptron: XOR ==\n";
    Perceptron p_xor(2,1);
    p_xor.train(X_xor, Y_xor, Mat(), Mat());
    p_xor.save("data/xor_weights.txt");
    std::cout << "Acc train (XOR): " << accuracy(p_xor, X_xor, Y_xor) << "%\n";
    for (size_t i=0;i<X_xor.size();++i) {
        auto out = p_xor.predict(X_xor[i]);
        double pred = out[0] >= 0.0 ? 1.0 : -1.0;
        std::cout << "  In: ("<<X_xor[i][0]<<","<<X_xor[i][1]<<") -> "<< pred << " (raw="<< out[0] <<")\n";
    }

    return 0;
}
