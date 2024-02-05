import numpy as np
np.set_printoptions(threshold=10000,suppress=True)
import warnings
warnings.filterwarnings('ignore')
import pickle
import functions as f
import os

#Script de mise à jour de la pipeline à partir des données de production

X,y = f.loadData("data/ref_data.csv")
Xprod, yprod = f.loadData("data/prod_data.csv")

X,y = np.concatenate((X, Xprod)), np.concatenate((y, yprod))
pipeline = f.createPipeline()

# Entraînement du pipeline sur l'ensemble d'apprentissage
pipeline.fit(X, y)

# Sauvegarde du pipeline
with open('artifacts/pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

dataset = np.concatenate(X, y, axis=1)
dataset.to_csv("data/ref_data.csv", index=False)
os.remove("data/prod_data.csv")