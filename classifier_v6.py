import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

def plot_confusion_matrix(confusion, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Rótulos Preditos')
    plt.ylabel('Rótulos Verdadeiros')
    plt.title('Matriz de Confusão')
    plt.show()

df = pd.read_csv('dataset_sdn_v2.csv')

X = df.drop(['Attack'], axis=1)
y = df['Attack']

X = pd.get_dummies(X, columns=['Source IP', 'Destination IP', 'Protocol'], drop_first=True)

pca = PCA(n_components=0.95) 
X_pca = pca.fit_transform(X)

selector = SelectKBest(score_func=f_classif, k=10)
X_kbest = selector.fit_transform(X, y)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_kbest, y)

stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in stratified_split.split(X_resampled, y_resampled):
    X_train, X_test = X_resampled[train_index], X_resampled[test_index]
    y_train, y_test = y_resampled[train_index], y_resampled[test_index]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Decision Tree': (DecisionTreeClassifier(random_state=42), {
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }),
    'Logistic Regression': (LogisticRegression(random_state=42), {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2']
    }),
    'Naive Bayes': (GaussianNB(), {}),
    'Random Forest': (RandomForestClassifier(random_state=42), {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }),
    'KNN': (KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    }),
    'Bagging': (BaggingClassifier(estimator=DecisionTreeClassifier(), random_state=42), {
        'n_estimators': [10, 50, 100],
        'estimator__max_depth': [5, 10, 15]
    }),
    'AdaBoost': (AdaBoostClassifier(estimator=DecisionTreeClassifier(), random_state=42), {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    })
}

labels = ['Normal', 'Attack']

for model_name, (model, params) in models.items():
    print(f"{'='*40}\n{model_name}\n{'='*40}")
    
    if params:
        grid_search = GridSearchCV(model, params, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        print("Melhores Parâmetros:", grid_search.best_params_)
    else:
        best_model = model
        best_model.fit(X_train_scaled, y_train)
    
    y_pred = best_model.predict(X_test_scaled)
    
    if hasattr(best_model, "predict_proba"):
        y_pred_proba = best_model.predict_proba(X_test_scaled)
        std_dev = np.std(y_pred_proba, axis=0)
        print(f"Desvio Padrão das Probabilidades de Predição: {std_dev}")

    confusion = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, target_names=labels)
    
    print("\nMatriz de Confusão:")
    print(confusion)
    print("\nRelatório de Classificação:")
    print(classification_rep)
    
    if model_name == 'Decision Tree':
        plt.figure(figsize=(12, 8))
        plot_tree(best_model, filled=True, feature_names=X.columns.tolist(), class_names=labels, max_depth=5)
        plt.title('Árvore de Decisão')
        plt.show()
    plot_confusion_matrix(confusion, labels)

plt.figure(figsize=(12, 6))
sns.scatterplot(x='Packet Rate', y='Bytes Sent', hue='Attack', data=df, palette=['blue', 'red'], alpha=0.7)
plt.title('Distribuição das Amostras')
plt.xlabel('Packet Rate')
plt.ylabel('Bytes Sent')
plt.legend(['Normal', 'Attack'])
plt.show()
