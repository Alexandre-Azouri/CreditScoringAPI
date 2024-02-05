import numpy as np
np.set_printoptions(threshold=10000,suppress=True)
import warnings
warnings.filterwarnings('ignore')
import pickle
import functions as f

#Ce script se base sur une étude préalable, d'où l'absence de données de test

X,y = f.loadData("data/ref_data.csv")

pipeline = f.createPipeline()

# Entraînement du pipeline sur l'ensemble d'apprentissage
pipeline.fit(X, y)

# Sauvegarde du pipeline
with open('artifacts/pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

