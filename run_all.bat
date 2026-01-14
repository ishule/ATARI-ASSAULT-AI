@echo off
if not exist results mkdir results

echo Running Perceptron experiments...
bin\RunPerceptron --dataset iris --data data/Iris.csv --model models/iris_perceptron.txt --train-split 0.7 --val-split 0.15 > results\perceptron_iris.txt
bin\RunPerceptron --dataset cancer --data data/cancermama.csv --model models/cancer_perceptron.txt --train-split 0.7 --val-split 0.15 > results\perceptron_cancer.txt
bin\RunPerceptron --dataset wine --data data/winequality-red.csv --model models/wine_perceptron.txt --train-split 0.7 --val-split 0.15 > results\perceptron_wine.txt

echo Running GA experiments (weights)...
bin\RunGA --mode weights --dataset iris --train-split 0.7 --val-split 0.15 > results\ga_weights_iris.txt
bin\RunGA --mode weights --dataset cancer --train-split 0.7 --val-split 0.15 > results\ga_weights_cancer.txt
bin\RunGA --mode weights --dataset wine --train-split 0.7 --val-split 0.15 > results\ga_weights_wine.txt

echo Running GA experiments (neuroevolution)...
bin\RunGA --mode neuro --dataset iris --train-split 0.7 --val-split 0.15 > results\ga_neuro_iris.txt
bin\RunGA --mode neuro --dataset cancer --train-split 0.7 --val-split 0.15 > results\ga_neuro_cancer.txt
bin\RunGA --mode neuro --dataset wine --train-split 0.7 --val-split 0.15 > results\ga_neuro_wine.txt

echo Done. Results are in results\
pause
