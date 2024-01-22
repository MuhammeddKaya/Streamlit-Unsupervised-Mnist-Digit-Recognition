
# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image, ImageOps
# from streamlit_drawable_canvas import st_canvas
# from models import load_model_asset, load_model_kmean, load_random_forest_model

# # Model yüklemesi
# kmeans = load_model_kmean()
# random_forest = load_random_forest_model()
# x_train = load_model_asset()

# # Model seçimi için kullanıcı arayüzü elemanları
# st.title("MNIST Rakam Tahmini")
# model_option = st.selectbox("Modeli Seçin", ('K-Means', 'Random Forest'))

# # Çizim alanı
# canvas_result = st_canvas(
#     fill_color="black",
#     stroke_width=15,
#     stroke_color="white",
#     background_color="black",
#     width=400,
#     height=400,
#     drawing_mode="freedraw",
#     key="canvas",
# )

# # Gönder butonu
# if st.button("Resmi Gönder"):
#     if canvas_result.image_data is not None:
#         img = Image.fromarray(canvas_result.image_data.astype('uint8'))
#         img = ImageOps.grayscale(img)
#         img = img.resize((28, 28))

#         img_np = np.array(img).reshape(1, -1).astype('float64') / 255

#         if model_option == 'K-Means':
#             kmeans.cluster_centers_ = kmeans.cluster_centers_.astype('float64')
#             cluster = kmeans.predict(img_np)[0]
#             indices = np.where(kmeans.labels_ == cluster)[0]
#             sample_indices = np.random.choice(indices, 8, replace=False)
            
#             fig, axes = plt.subplots(1, 8, figsize=(16, 2))
#             for ax, idx in zip(axes, sample_indices):
#                 ax.imshow(x_train[idx].reshape(28, 28), cmap='gray')
#                 ax.axis('off')
#             st.pyplot(fig)

#         elif model_option == 'Random Forest':
#             prediction = random_forest.predict(img_np)
#             st.write(f"Random Forest tahmini: {prediction[0]}")




# # import streamlit as st
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from PIL import Image, ImageOps
# # from streamlit_drawable_canvas import st_canvas
# # from models import load_model_asset, load_model_kmean
# # from models import load_random_forest_model

# # # Model yüklemesi
# # kmeans = load_model_kmean()

# # x_train = load_model_asset()
# # random_forest = load_random_forest_model()

# # # Model seçimi için kullanıcı arayüzü elemanları
# # st.title("MNIST Rakam Tahmini")
# # model_option = st.selectbox("Modeli Seçin", ('K-Means', 'Random Forest'))


# # # Çizim alanı
# # canvas_result = st_canvas(
# #     fill_color="black",
# #     stroke_width=15,
# #     stroke_color="white",
# #     background_color="black",
# #     width=400,
# #     height=400,
# #     drawing_mode="freedraw",
# #     key="canvas",
# # )

# # # Gönder butonu
# # if st.button("Resmi Gönder"):
# #     if canvas_result.image_data is not None:
# #         img = Image.fromarray(canvas_result.image_data.astype('uint8'))
# #         img = ImageOps.grayscale(img)
# #         img = img.resize((28, 28))

# #         img_np = np.array(img).reshape(1, -1).astype('float64') / 255

# #         if model_option == 'K-Means':
# #             kmeans.cluster_centers_ = kmeans.cluster_centers_.astype('float64')
# #             cluster = kmeans.predict(img_np)[0]
# #             indices = np.where(kmeans.labels_ == cluster)[0]
# #             sample_indices = np.random.choice(indices, 8, replace=False)

# #         elif model_option == 'Random Forest':
# #             img_np = np.array(img).reshape(1, -1).astype('float32') / 255
# #             prediction = random_forest.predict(img_np)
# #             st.write(f"Random Forest tahmini: {prediction[0]}")




# #         fig, axes = plt.subplots(1, 8, figsize=(16, 2))
# #         for ax, idx in zip(axes, sample_indices):
# #             ax.imshow(x_train[idx].reshape(28, 28), cmap='gray')
# #             ax.axis('off')
# #         st.pyplot(fig)




# # import streamlit as st
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from PIL import Image, ImageOps
# # from streamlit_drawable_canvas import st_canvas
# # from models import load_model_asset, load_model_kmean, load_autoencoder

# # # Model yüklemesi
# # kmeans = load_model_kmean()
# # autoencoder = load_autoencoder()
# # x_train = load_model_asset()

# # # Model seçimi için kullanıcı arayüzü elemanları
# # st.title("MNIST Rakam Tahmini")
# # model_option = st.selectbox("Modeli Seçin", ('K-Means', 'Autoencoder'))

# # # Çizim alanı
# # canvas_result = st_canvas(
# #     fill_color="black",  # Canvas arkaplan rengi
# #     stroke_width=15,  # Çizim kalem kalınlığı
# #     stroke_color="white",  # Çizim rengi
# #     background_color="black",  # Gerçek arkaplan rengi
# #     width=400,
# #     height=400,
# #     drawing_mode="freedraw",
# #     key="canvas",
# # )

# # # Gönder butonu
# # if st.button("Resmi Gönder"):
# #     if canvas_result.image_data is not None:
# #         # Resmi işle
# #         img = Image.fromarray(canvas_result.image_data.astype('uint8'))
# #         img = ImageOps.grayscale(img)  # Griye çevir
# #         img = img.resize((28, 28))  # MNIST formatına boyutlandır

# #         # Modelle tahmin yap
# #         img_np = np.array(img).reshape(1, -1).astype('float32') / 255

# #         if model_option == 'K-Means':
# #             # K-means modeli ile tahmin yap
# #             cluster = kmeans.predict(img_np)[0].astype('float64')
# #             # İlgili kümeden örnekleri göster
# #             indices = np.where(kmeans.labels_ == cluster)[0]
# #             sample_indices = np.random.choice(indices, 8, replace=False)

# #         elif model_option == 'Autoencoder':
# #             # Otoenkoder modeli ile tahmin yap (burada yeniden oluşturma hatasını kullanabilirsiniz)
# #             reconstructed_img = autoencoder.predict(img_np)
# #             reconstruction_error = np.mean(np.abs(img_np - reconstructed_img))

# #             st.write(f"Yeniden Oluşturma Hatası: {reconstruction_error:.4f}")
# #             # Otoenkoder için örnek gösterim henüz belirlenmedi, bu yüzden rastgele resimler gösterilebilir
# #             sample_indices = np.random.choice(range(len(x_train)), 8, replace=False)

# #         # Seçilen örnekleri göster
# #         fig, axes = plt.subplots(1, 8, figsize=(16, 2))
# #         for ax, idx in zip(axes, sample_indices):
# #             ax.imshow(x_train[idx].reshape(28, 28), cmap='gray')
# #             ax.axis('off')
# #         st.pyplot(fig)




# # import streamlit as st
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from PIL import Image, ImageOps
# # import joblib
# # from streamlit_drawable_canvas import st_canvas
# # from models import load_model_asset, load_model_kmean

# # # Modeli yükleme
# # kmeans = load_model_kmean()
# # x_train = load_model_asset()

# # st.title("MNIST Rakam Tahmini")

# # # Çizim alanı
# # canvas_result = st_canvas(
# #     fill_color="black",  # Canvas arkaplan rengi
# #     stroke_width=15,  # Çizim kalem kalınlığı
# #     stroke_color="white",  # Çizim rengi
# #     background_color="black",  # Gerçek arkaplan rengi
# #     width=400,
# #     height=400,
# #     drawing_mode="freedraw",
# #     key="canvas",
# # )

# # # Gönder butonu
# # if st.button("Resmi Gönder"):
# #     if canvas_result.image_data is not None:
# #         # Resmi işle
# #         img = Image.fromarray(canvas_result.image_data.astype('uint8'))
# #         img = ImageOps.grayscale(img)  # Griye çevir
# #         img = img.resize((28, 28))  # MNIST formatına boyutlandır

# #         # Modelle tahmin yap
# #         img_np = np.array(img).reshape(1, -1).astype('float64') / 255  # Değişiklik burada

# #         # Küme merkezlerini uygun veri türüne çevir
# #         kmeans.cluster_centers_ = kmeans.cluster_centers_.astype('float64')  # Değişiklik burada

# #         cluster = kmeans.predict(img_np)[0]

# #         # İlgili kümeden örnekleri göster
# #         indices = np.where(kmeans.labels_ == cluster)[0]
# #         sample_indices = np.random.choice(indices, 8, replace=False)

# #         fig, axes = plt.subplots(1, 8, figsize=(16, 2))
# #         for ax, idx in zip(axes, sample_indices):
# #             ax.imshow(x_train[idx].reshape(28, 28), cmap='gray')
# #             ax.axis('off')
# #         st.pyplot(fig)



# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image, ImageOps
# import joblib
# from streamlit_drawable_canvas import st_canvas
# from models import load_model_asset, load_model_kmean

# # Modeli yükleme
# kmeans = load_model_kmean()
# kmeans.cluster_centers_ = kmeans.cluster_centers_.astype('float64')  # Değişiklik burada
# x_train = load_model_asset()

# st.title("MNIST Rakam Tahmini")

# # Çizim alanı
# canvas_result = st_canvas(
#     fill_color="black",  # Canvas arkaplan rengi
#     stroke_width=15,  # Çizim kalem kalınlığı
#     stroke_color="white",  # Çizim rengi
#     background_color="black",  # Gerçek arkaplan rengi
#     width=400,
#     height=400,
#     drawing_mode="freedraw",
#     key="canvas",
# )

# # Çizim alanından veri alındığında
# if canvas_result.image_data is not None:
#     # Resmi işle
#     img = Image.fromarray(canvas_result.image_data.astype('uint8'))
#     img = ImageOps.grayscale(img)  # Griye çevir
#     img = img.resize((28, 28))  # MNIST formatına boyutlandır

#     # Modelle tahmin yap
#     img_np = np.array(img).reshape(1, -1).astype('float64') / 255  # Değişiklik burada

#     # Küme merkezlerini uygun veri türüne çevir
#     kmeans.cluster_centers_ = kmeans.cluster_centers_.astype('float64')  # Değişiklik burada

#     cluster = kmeans.predict(img_np)[0]

#     # İlgili kümeden örnekleri göster
#     indices = np.where(kmeans.labels_ == cluster)[0]
#     sample_indices = np.random.choice(indices, 8, replace=False)

#     fig, axes = plt.subplots(1, 8, figsize=(16, 2))
#     for ax, idx in zip(axes, sample_indices):
#         ax.imshow(x_train[idx].reshape(28, 28), cmap='gray')
#         ax.axis('off')
#     st.pyplot(fig)