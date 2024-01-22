from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from keras.datasets import mnist
import numpy as np
import joblib
import os

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

def load_random_forest_model():
    rf_model_path = 'random_forest_model.pkl'
    if os.path.exists(rf_model_path):
        rf = joblib.load(rf_model_path)
    else:
        x_train, y_train = load_model_asset()
        rf = RandomForestClassifier(n_estimators=100, random_state=0)
        rf.fit(x_train, y_train)
        joblib.dump(rf, rf_model_path)
    return rf



# from sklearn.cluster import KMeans
# from keras.datasets import mnist
# import numpy as np
# import joblib
# import os
# from keras.layers import Input, Dense
# from keras.models import Model
# from keras.models import load_model
# from sklearn.decomposition import PCA
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.cluster import DBSCAN

# def load_model_asset():
#     x_train_path = 'x_train.npy'
#     if os.path.exists(x_train_path):
#         x_train = np.load(x_train_path)
#     else:
#         (x_train, _), (_, _) = mnist.load_data()
#         x_train = x_train.reshape(-1, 784).astype('float32') / 255
#         np.save(x_train_path, x_train)
#     return x_train

# def load_model_kmean():
#     model_path = 'kmeans_model.pkl'
#     if os.path.exists(model_path):
#         kmeans = joblib.load(model_path)
#     else:
#         x_train = load_model_asset()
#         kmeans = KMeans(n_clusters=10, random_state=0)  # Küme sayısı 10 olarak ayarlandı
#         kmeans.fit(x_train)
#         joblib.dump(kmeans, model_path)
#     return kmeans


# from sklearn.cluster import DBSCAN

# def create_dbscan_model(x_train, eps=0.5, min_samples=5):
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#     dbscan_result = dbscan.fit_predict(x_train)
#     return dbscan, dbscan_result

# def load_dbscan_model():
#     dbscan_path = 'dbscan_result.npy'
#     if os.path.exists(dbscan_path):
#         dbscan_result = np.load(dbscan_path)
#     else:
#         x_train = load_model_asset()
#         _, dbscan_result = create_dbscan_model(x_train)
#         np.save(dbscan_path, dbscan_result)
#     return dbscan_result

# def create_dbscan_model(x_train, eps=0.5, min_samples=5):
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#     dbscan_result = dbscan.fit_predict(x_train)
#     return dbscan, dbscan_result



# def load_dbscan_model():
#     dbscan_path = 'dbscan_result.npy'
#     if os.path.exists(dbscan_path):
#         dbscan_result = np.load(dbscan_path)
#     else:
#         x_train = load_model_asset()
#         _, dbscan_result = create_dbscan_model(x_train)
#         np.save(dbscan_path, dbscan_result)
#     return dbscan_result

# def load_pca_for_dbscan():
#     pca_path = 'pca_for_dbscan.npy'
#     if os.path.exists(pca_path):
#         pca_result = np.load(pca_path)
#     else:
#         x_train = load_model_asset()
#         pca, pca_result = create_pca_model(x_train, n_components=2)
#         np.save(pca_path, pca_result)
#     return pca_result


# from sklearn.ensemble import RandomForestClassifier

# def train_random_forest(x_train, y_train, n_estimators=100, random_state=0):
#     rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
#     rf.fit(x_train, y_train)
#     return rf

# def load_random_forest_model():
#     rf_model_path = 'random_forest_model.pkl'
#     if os.path.exists(rf_model_path):
#         rf = joblib.load(rf_model_path)
#     else:
#         (x_train, y_train), (_, _) = mnist.load_data()
#         x_train = x_train.reshape(-1, 784).astype('float32') / 255
#         rf = train_random_forest(x_train, y_train)
#         joblib.dump(rf, rf_model_path)
#     return rf




# # from sklearn.cluster import KMeans
# # from keras.datasets import mnist
# # import numpy as np
# # import joblib
# # import os
# # from sklearn.model_selection import GridSearchCV

# # def load_model_asset():
# #     x_train_path = 'x_train.npy'
# #     if os.path.exists(x_train_path):
# #         # Eğer x_train zaten kaydedilmişse, dosyadan yükle
# #         x_train = np.load(x_train_path)
# #     else:
# #         # x_train'i yükle, ön işle yap ve diske kaydet
# #         (x_train, _), (_, _) = mnist.load_data()
# #         x_train = x_train.reshape(-1, 784).astype('double') / 255
# #         np.save(x_train_path, x_train)
# #     return x_train

# # def load_model_kmean():
# #     model_path = 'kmeans_model.pkl'
# #     if os.path.exists(model_path):
# #         # Eğer model zaten kaydedilmişse, dosyadan yükle
# #         kmeans = joblib.load(model_path)
# #     else:
# #         # Modeli eğit ve diske kaydet
# #         x_train = load_model_asset()
        
# #         # Hiperparametre tuning için kullanılacak parametre kombinasyonları
# #         param_grid = {
# #             'n_clusters': [5, 8, 10, 12],  # İstediğin küme sayıları
# #             'max_iter': [100, 500, 1000],    # İterasyon sayıları
# #             'n_init': [10, 20, 30]           # İlk küme merkezlerinin seçilme sayıları
# #         }
        
# #         # GridSearchCV uygula
# #         grid_search = GridSearchCV(estimator=KMeans(random_state=0), param_grid=param_grid, cv=3)
# #         grid_search.fit(x_train)
        
# #         # En iyi parametreleri ve modeli al
# #         best_params = grid_search.best_params_
# #         kmeans = grid_search.best_estimator_

# #         # Modeli kaydet
# #         joblib.dump(kmeans, model_path)
    
# #     return kmeans



















# #----------------------------v1----------------------------------

# # from sklearn.cluster import KMeans
# # from keras.datasets import mnist
# # import numpy as np
# # import joblib
# # import os
# # from keras.layers import Input, Dense
# # from keras.models import Model
# # from keras.models import load_model


# # def load_model_asset():
# #     x_train_path = 'x_train.npy'
# #     if os.path.exists(x_train_path):
# #         # Eğer x_train zaten kaydedilmişse, dosyadan yükle
# #         x_train = np.load(x_train_path)
# #     else:
# #         # x_train'i yükle, ön işle yap ve diske kaydet
# #         (x_train, _), (_, _) = mnist.load_data()
# #         x_train = x_train.reshape(-1, 784).astype('double') / 255
# #         np.save(x_train_path, x_train)
# #     return x_train

# # def load_model_kmean():
# #     model_path = 'kmeans_model.pkl'
# #     if os.path.exists(model_path):
# #         # Eğer model zaten kaydedilmişse, dosyadan yükle
# #         kmeans = joblib.load(model_path)
# #     else:
# #         # Modeli eğit ve diske kaydet
# #         x_train = load_model_asset()
# #         kmeans = KMeans(n_clusters=100, random_state=0)
# #         kmeans.fit(x_train)
# #         joblib.dump(kmeans, model_path)
# #     return kmeans


# # # models.py'ye ekleyin

# # def create_autoencoder():
# #     input_img = Input(shape=(784,))
# #     encoded = Dense(128, activation='relu')(input_img)
# #     encoded = Dense(64, activation='relu')(encoded)

# #     decoded = Dense(128, activation='relu')(encoded)
# #     decoded = Dense(784, activation='sigmoid')(decoded)

# #     autoencoder = Model(input_img, decoded)
# #     autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# #     return autoencoder

# # def train_autoencoder(x_train):
# #     autoencoder = create_autoencoder()
# #     autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True)
# #     return autoencoder

# # def load_autoencoder():
# #     autoencoder_path = 'autoencoder_model.h5'
# #     if os.path.exists(autoencoder_path):
# #         # Eğer model zaten kaydedilmişse, dosyadan yükle
# #         autoencoder = load_model(autoencoder_path)
# #     else:
# #         # Modeli eğit ve diske kaydet
# #         x_train = load_model_asset()
# #         autoencoder = train_autoencoder(x_train)
# #         autoencoder.save(autoencoder_path)
# #     return autoencoder
