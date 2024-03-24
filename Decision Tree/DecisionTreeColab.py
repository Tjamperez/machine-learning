###########################################Parte 1 Colab##########################################
from sklearn.datasets import load_breast_cancer


data = load_breast_cancer()
X = data.data  # matriz contendo os atributos
y = data.target  # vetor contendo a classe (0 para maligno e 1 para benigno) de cada instância
feature_names = data.feature_names  # nome de cada atributo
target_names = data.target_names  # nome de cada classe

print(f"Dimensões de X: {X.shape}\n")
print(f"Dimensões de y: {y.shape}\n")
print(f"Nomes dos atributos: {feature_names}\n")
print(f"Nomes das classes: {target_names}")
###########################################Parte 1 Colab##########################################

###########################################Parte 2 Colab##########################################
import numpy as np

n_malign = np.sum(y == 0)
n_benign = np.sum(y == 1)

print("\nNúmero de exemplos malignos: %d" % n_malign)
print("\nNúmero de exemplos benignos: %d" % n_benign)
###########################################Parte 2 Colab##########################################

###########################################Parte 3 Colab##########################################
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split  # função do scikit-learn que implementa um holdout

def get_root_node(dt, feature_names):
    feature_idx = dt.tree_.feature[0]
    return feature_names[feature_idx]


n_repeats = 20
root_nodes = []

# variando o seed do holdout, geramos conjuntos de treino e teste um pouco diferentes a cada iteração
for split_random_state in range(0, n_repeats):
  # Holdout com 20% de dados de teste
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=split_random_state)

  # Treinamento da árvore usando os dados de treino
  dt = DecisionTreeClassifier(random_state=0)
  dt.fit(X_train, y_train)

  # Obtemos o atributo usado na raiz e o salvamos na lista
  root_node = get_root_node(dt, feature_names)
  root_nodes.append(root_node)

print(root_nodes)

###########################################Parte 3 Colab##########################################

###########################################Parte 4 Colab##########################################

from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia nos dados de teste: %.3f" % accuracy)

###########################################Parte 4 Colab##########################################

###########################################Parte 5 Colab##########################################
n_repeats = 20
accuracies = []

# variando o seed do holdout, geramos conjuntos de treino e teste um pouco diferentes a cada iteração
for split_random_state in range(0, n_repeats):
  # Holdout com 20% de dados de teste
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=split_random_state)

  # Nova instância da árvore de decisão
  dt = DecisionTreeClassifier(random_state=0)
  
  # Treine a árvore de decisão usando os dados de treino
  # ...
  dt.fit(X_train, y_train)
  y_pred = dt.predict(X_test)

  # Calcule a acurácia nos dados de teste
  # ...
  accuracy = accuracy_score(y_test, y_pred)
  print("Acurácia nos dados de teste: %.3f" % accuracy)
  accuracies.append(accuracy)
# Calcule a média, desvio padrão, máximo e mínimo das acurácias (pode usar numpy)
# ...

max_acc = max(accuracies)
min_acc = min(accuracies)

print("Max Accuracy: %.3f" % max_acc)
print("Min Accuracy: %.3f" % min_acc)

average = np.average(accuracies)
print("Average in data: %.3f" % average)

variance = np.var(accuracies)
print("Average in data: %.3f" % variance)

deviation = np.std(accuracies)
print("Average in data: %.3f" % deviation)

###########################################Parte 5 Colab##########################################

###########################################Parte 6 Colab##########################################

X_interesting = X[[40, 86, 297, 135, 73], :]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=split_random_state)

dt = DecisionTreeClassifier()
  
# Treine a árvore de decisão usando os dados de treino
# ...
dt.fit(X_train, y_train)
y_pred = dt.predict(X_interesting)

print(y_pred)

#[1,1,0,0,0]


# 1. Instancie uma nova árvore de decisão, dessa vez sem especificar o valor de random_state
# 2. Separe o conjunto em treino e teste, dessa vez sem especificar o valor de random_state
# 3. Treine a nova árvore usando o conjunto de treino
# 4. Use a nova árvore treinada para obter predições para os valores de X_interesting acima.

###########################################Parte 6 Colab##########################################

###########################################Parte 7 Colab##########################################

dt = DecisionTreeClassifier(max_depth=2)
dt.fit(X, y)

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(12,6))
_ = plot_tree(dt, feature_names=feature_names, class_names=target_names)

###########################################Parte 7 Colab##########################################

###########################################Parte 8 Colab##########################################

max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, None]  # None faz com que essa poda não seja aplicada
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
for depth in max_depths:
  dt = DecisionTreeClassifier(max_depth=depth, random_state=0)
  dt.fit(X_train, y_train)
  
  y_pred = dt.predict(X_test)
  acc = accuracy_score(y_test, y_pred)
  print(acc)
  
###########################################Parte 8 Colab##########################################
  
###########################################Parte 9 Colab##########################################

def plot_acc_vs_ccp(accuracies_train, accuracies_test, ccps):
  fig, ax = plt.subplots(figsize=(8, 4))
  ax.set_xlabel("alpha")
  ax.set_ylabel("accuracy")
  ax.set_title("Accuracy vs alpha for training and testing sets")
  ax.plot(ccps, accuracies_train, marker="o", label="train", drawstyle="steps-post")
  ax.plot(ccps, accuracies_test, marker="o", label="test", drawstyle="steps-post")
  ax.legend()
  ax.grid()
  plt.show()


accs_train = []
accs_test = []
ccps = [k * 0.001 for k in range(0, 200, 2)]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
for ccp in ccps:
  dt = DecisionTreeClassifier(ccp_alpha=ccp, random_state=0)
  dt.fit(X_train, y_train)
  
  y_pred_train = dt.predict(X_train)
  acc_train = accuracy_score(y_train, y_pred_train)

  y_pred_test = dt.predict(X_test)
  acc_test = accuracy_score(y_test, y_pred_test)

  accs_train.append(acc_train)
  accs_test.append(acc_test)

plot_acc_vs_ccp(accs_train, accs_test, ccps)

###########################################Parte 9 Colab##########################################