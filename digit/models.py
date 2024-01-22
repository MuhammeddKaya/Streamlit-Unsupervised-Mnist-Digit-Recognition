from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from keras.datasets import mnist
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.mixture import GaussianMixture

def load_model_asset():
    x_train_path = 'x_train.npy'
    y_train_path = 'y_train.npy'
    if os.path.exists(x_train_path) and os.path.exists(y_train_path):
        x_train = np.load(x_train_path)
        y_train = np.load(y_train_path)
    else:
        (x_train, y_train), (_, _) = mnist.load_data()
        x_train = x_train.reshape(-1, 784).astype('float32') / 255
        y_train = y_train.reshape(-1)
        np.save(x_train_path, x_train)
        np.save(y_train_path, y_train)
    return x_train, y_train

def load_model_kmean():
    model_path = 'kmeans_model.pkl'
    if os.path.exists(model_path):
        kmeans = joblib.load(model_path)
    else:
        x_train, _ = load_model_asset()
        kmeans = KMeans(n_clusters=10, random_state=0)
        kmeans.fit(x_train)
        joblib.dump(kmeans, model_path)
    return kmeans

# def load_random_forest_model():
#     rf_model_path = 'random_forest_model.pkl'
#     if os.path.exists(rf_model_path):
#         rf = joblib.load(rf_model_path)
#     else:
#         x_train, y_train = load_model_asset()
#         rf = RandomForestClassifier(n_estimators=100, random_state=0)
#         rf.fit(x_train, y_train)
#         joblib.dump(rf, rf_model_path)
#     return rf

def train_random_forest_with_grid_search(x_train, y_train):
    # Veriyi eğitim ve test setine ayırma (örneğin, 70-30 bölme)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestClassifier(random_state=0)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(x_train, y_train)

    # Test seti üzerinde performansı değerlendirme
    test_score = grid_search.score(x_test, y_test)
    print(f"Test seti doğruluk oranı: {test_score}")

    return grid_search.best_estimator_
def load_random_forest_model():
    rf_model_path = 'random_forest_model.pkl'
    if os.path.exists(rf_model_path):
        rf = joblib.load(rf_model_path)
    else:
        x_train, y_train = load_model_asset()
        rf = train_random_forest_with_grid_search(x_train, y_train)
        joblib.dump(rf, rf_model_path)
    return rf

def load_gmm_model():
    model_path = 'gmm_model.pkl'
    if os.path.exists(model_path):
        gmm = joblib.load(model_path)
    else:
        x_train, _ = load_model_asset()
        gmm = GaussianMixture(n_components=10, random_state=0)
        gmm.fit(x_train)
        joblib.dump(gmm, model_path)
    return gmm