import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

def createPipeline():
    # Définition du pipeline
    pipeline = Pipeline([
        ('selection', SelectFromModel(RandomForestClassifier(n_estimators=1000, random_state=1), max_features=8)),
        ('normalizer', StandardScaler()),  # Étape de normalisation
        # ('pca', PCA(n_components=2)),     # Étape d'ACP
        ('classifier', MLPClassifier(hidden_layer_sizes=(40, 20), random_state=1))  # Étape de classification
    ])

    return pipeline

def loadData(path):
    dataset = pd.read_csv(path)
    numpyArray = dataset.values
    y = numpyArray[:, -1]
    X = np.delete(numpyArray, -1, axis=1)
    return X, y