# gui.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from models import load_model_asset, load_model_kmean, load_random_forest_model,load_gmm_model

# Model yüklemesi
kmeans = load_model_kmean()
random_forest = load_random_forest_model()
x_train, _ = load_model_asset()  # x_train ve y_train'i ayrı ayrı al
gmm = load_gmm_model()

# Model seçimi için kullanıcı arayüzü elemanları
st.title("MNIST Rakam Tahmini")
model_option = st.selectbox("Modeli Seçin", ('K-Means (usv)', 'Random Forest (sv)', 'Gaussian Mixture (usv)'))

st.sidebar.title("MNIST Verisetinden Örnekler")
minist_samples = np.random.choice(len(x_train), 5, replace=False)
for sample_idx in minist_samples:
    st.sidebar.image(x_train[sample_idx].reshape(28, 28), caption=f"Örnek {sample_idx}", width=150, use_column_width=True)

# Çizim alanı
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    width=400,
    height=400,
    drawing_mode="freedraw",
    key="canvas",
)

# Gönder butonu
if st.button("Resmi Gönder"):
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype('uint8'))
        img = ImageOps.grayscale(img)
        img = img.resize((28, 28))

        img_np = np.array(img).reshape(1, -1).astype('float64') / 255

        if model_option == 'K-Means (usv)':
            kmeans.cluster_centers_ = kmeans.cluster_centers_.astype('float64')
            cluster = kmeans.predict(img_np)[0]
            indices = np.where(kmeans.labels_ == cluster)[0]
            sample_indices = np.random.choice(indices, 32, replace=False)
            
            fig, axes = plt.subplots(1, 32, figsize=(16, 12))
            for ax, idx in zip(axes, sample_indices):
                ax.imshow(x_train[idx].reshape(28, 28), cmap='gray')
                ax.axis('off')
            st.pyplot(fig)
        if model_option == 'Gaussian Mixture (usv)':
            cluster = gmm.predict(img_np)[0]

            # Kümedeki örneklerden rastgele seçim
            indices = np.where(gmm.predict(x_train) == cluster)[0]
            sample_indices = np.random.choice(indices, 32, replace=False)

            # Görselleştirme
            fig, axes = plt.subplots(1, 32, figsize=(16, 12))
            for ax, idx in zip(axes, sample_indices):
                ax.imshow(x_train[idx].reshape(28, 28), cmap='gray')
                ax.axis('off')
            st.pyplot(fig)

        elif model_option == 'Random Forest (sv)':
            prediction = random_forest.predict(img_np)
            st.write(f"Random Forest tahmini: {prediction[0]}")

