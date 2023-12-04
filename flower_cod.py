import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical


# Folder paths
data_folder = r'C:\\TIA\\tema 1 tia\\flowers'  # Schimbă aceasta cu calea către folderul cu setul tău de date

# Funcție pentru încărcarea și preprocesarea imaginilor
def preprocess_images(folder_path, target_size=(100, 100)):
    images = []
    labels = []
    for class_name in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_name)
        if os.path.isdir(class_path):
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                img = cv2.imread(image_path)
                if img is not None:
                    # Redimensionare imagine la dimensiunea specificată (100x100)
                    img = cv2.resize(img, target_size)
                    # Normalizare valorilor pixelilor în intervalul [0, 1] menținând valoarea maximă originală a pixelilor (255)
                    img = img.astype('float32') / img.max()
                    images.append(img)
                    labels.append(class_name)
                else:
                    print(f"Warning: Unable to load {image_name}")
    return np.array(images), np.array(labels)


# Funcție pentru încărcarea, afișarea și identificarea imaginilor de test aleatorii
def display_random_test_images(folder_path, num_images=3, target_size=(100, 100)):
    for class_name in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_name)
        if os.path.isdir(class_path):
            print(f"Trebuie să afișăm 3 imagini aleatoare pentru clasa {class_name}:")
            image_paths = [os.path.join(class_path, img) for img in os.listdir(class_path)]
            random_images = random.sample(image_paths, num_images)
            plt.figure(figsize=(8, 4))
            for i, img_path in enumerate(random_images):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, target_size)
                    plt.subplot(1, num_images, i+1)
                    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    plt.title(class_name)  # Adăugarea titlului cu numele categoriei sub fiecare imagine
                    plt.axis('off')
            plt.show()
            
            
# Încărcare și preprocesare imagini
images, labels = preprocess_images(data_folder, target_size=(100, 100))

# Obținerea etichetelor numerice
label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(labels)

# Împărțirea setului de date în set de antrenare și set de testare (de exemplu, 80% pentru antrenare și 20% pentru testare)
train_images, test_images, train_labels, test_labels = train_test_split(images, numeric_labels, test_size=0.2, random_state=42)

# Convertirea etichetelor în formatul one-hot encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Afișare dimensiuni pentru seturile de antrenare și de testare
print(f"Dimensiune set de antrenare (imagini): {train_images.shape}")
print(f"Dimensiune set de antrenare (etichete): {train_labels.shape}")
print(f"Dimensiune set de testare (imagini): {test_images.shape}")
print(f"Dimensiune set de testare (etichete): {test_labels.shape}")


# Afișare a numărului de imagini și etichete încărcate
print(f"Numărul de imagini încărcate: {len(images)}")
print(f"Numărul de etichete încărcate: {len(labels)}")


# Definirea arhitecturii CNN

# Inițializare model secvențial
model = Sequential()

# Adăugarea straturilor convoluționale și de pooling
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Transformarea matricii într-un vector unidimensional
model.add(Flatten())

# Adăugarea straturilor complet conectate (fully connected)
model.add(Dense(128, activation='relu'))
model.add(Dense(5, activation='softmax'))  # Stratul de ieșire cu 5 neuroni pentru cele 5 clase de flori, activare softmax

# Compilarea modelului
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Antrenarea modelului pe setul de date de antrenare
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))

# Afișare de 3 poze aleatoare pentru fiecare clasă de flori
display_random_test_images(data_folder, num_images=3, target_size=(100, 100))
