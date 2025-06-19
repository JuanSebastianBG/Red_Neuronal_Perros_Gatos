# ===============================================================================
# CLASIFICACIÓN DE GATOS VS PERROS CON TENSORFLOW Y KERAS
#
# Este código implementa y compara 3 arquitecturas de redes neuronales:
# 1. Red Neuronal Densa (Dense)
# 2. Red Neuronal Convolucional (CNN)
# 3. CNN con Dropout
#
# Cada modelo se entrena con y sin Data Augmentation para comparar resultados
# ===============================================================================

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard

# ===============================================================================
# 1. CONFIGURACIÓN INICIAL Y DESCARGA DE DATOS
# ===============================================================================

# Fix para el dataset de cats_vs_dogs (issue conocido)
# Más detalle aquí: https://github.com/tensorflow/datasets/issues/3918
setattr(tfds.image_classification.cats_vs_dogs, '_URL',
        "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-DEBA77B919F/kagglecatsanddogs_5340.zip")

# Descargar el dataset de perros y gatos
print("Descargando dataset cats_vs_dogs...")
datos, metadatos = tfds.load('cats_vs_dogs', as_supervised=True, with_info=True)

# Configuraciones
TAMANO_IMG = 100

# ===============================================================================
# 2. VISUALIZACIÓN INICIAL DE DATOS
# ===============================================================================

print("Visualizando 25 imágenes de muestra...")
plt.figure(figsize=(20, 20))

for i, (imagen, etiqueta) in enumerate(datos['train'].take(25)):
    imagen = cv2.resize(imagen.numpy(), (TAMANO_IMG, TAMANO_IMG))
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(imagen, cmap='gray')
    plt.title(f"{'Gato' if etiqueta.numpy() == 0 else 'Perro'}")

plt.tight_layout()
plt.show()

# ===============================================================================
# 3. PREPROCESAMIENTO DE DATOS
# ===============================================================================

print("Preprocesando todas las imágenes...")
datos_entrenamiento = []

# Procesar todas las imágenes
for i, (imagen, etiqueta) in enumerate(datos['train']):
    if i % 1000 == 0:
        print(f"Procesadas {i} imágenes...")
    
    # Redimensionar y convertir a escala de grises
    imagen = cv2.resize(imagen.numpy(), (TAMANO_IMG, TAMANO_IMG))
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen = imagen.reshape(TAMANO_IMG, TAMANO_IMG, 1)  # Cambiar tamaño a 100,100,1
    
    # Normalizar pixeles a rango 0-1
    imagen = imagen.astype('float32') / 255.0
    
    datos_entrenamiento.append([imagen, etiqueta])

print(f"Total de imágenes procesadas: {len(datos_entrenamiento)}")

# ===============================================================================
# 4. SEPARAR DATOS EN X (ENTRADAS) Y y (ETIQUETAS)
# ===============================================================================

print("Separando datos en entradas y etiquetas...")
X = []  # imágenes de entrada (pixeles)
y = []  # etiquetas (perro o gato)

for imagen, etiqueta in datos_entrenamiento:
    X.append(imagen)
    y.append(etiqueta)

# Convertir a arrays de numpy
X = np.array(X)
y = np.array(y)

print(f"Forma de X: {X.shape}")
print(f"Forma de y: {y.shape}")

# ===============================================================================
# 5. DEFINICIÓN DE MODELOS
# ===============================================================================

print("Definiendo arquitecturas de modelos...")

# MODELO 1: RED NEURONAL DENSA
# Usa sigmoid como salida (en lugar de softmax) para mostrar como podría funcionar
# Sigmoid regresa siempre datos entre 0 y 1. Si se acerca a 0 = gato, si se acerca a 1 = perro
modeloDenso = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(100, 100, 1)),
    tf.keras.layers.Dense(150, activation='relu'),
    tf.keras.layers.Dense(150, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# MODELO 2: RED NEURONAL CONVOLUCIONAL (CNN)
modeloCNN = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# MODELO 3: CNN CON DROPOUT
modeloCNN2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(250, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# ===============================================================================
# 6. COMPILACIÓN DE MODELOS
# ===============================================================================

print("Compilando modelos...")

# Usar crossentropy binario ya que tenemos solo 2 opciones (perro o gato)
modeloDenso.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

modeloCNN.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

modeloCNN2.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

# Mostrar resumen de modelos
print("\n=== RESUMEN MODELO DENSO ===")
modeloDenso.summary()

print("\n=== RESUMEN MODELO CNN ===")
modeloCNN.summary()

print("\n=== RESUMEN MODELO CNN2 ===")
modeloCNN2.summary()

# ===============================================================================
# 7. ENTRENAMIENTO SIN DATA AUGMENTATION
# ===============================================================================

print("\n" + "="*50)
print("FASE 1: ENTRENAMIENTO SIN DATA AUGMENTATION")
print("="*50)

# Configurar TensorBoard para cada modelo
tensorboardDenso = TensorBoard(log_dir='logs/denso')
tensorboardCNN = TensorBoard(log_dir='logs/cnn')
tensorboardCNN2 = TensorBoard(log_dir='logs/cnn2')

# Entrenar Modelo Denso
print("\n--- Entrenando Modelo Denso ---")
historial_denso = modeloDenso.fit(
    X, y, 
    batch_size=32,
    validation_split=0.15,
    epochs=20,  # Reducido para ejemplo, original era 100
    callbacks=[tensorboardDenso],
    verbose=1
)

# Entrenar Modelo CNN
print("\n--- Entrenando Modelo CNN ---")
historial_cnn = modeloCNN.fit(
    X, y, 
    batch_size=32,
    validation_split=0.15,
    epochs=20,  # Reducido para ejemplo, original era 100
    callbacks=[tensorboardCNN],
    verbose=1
)

# Entrenar Modelo CNN2
print("\n--- Entrenando Modelo CNN2 ---")
historial_cnn2 = modeloCNN2.fit(
    X, y, 
    batch_size=32,
    validation_split=0.15,
    epochs=20,  # Reducido para ejemplo, original era 100
    callbacks=[tensorboardCNN2],
    verbose=1
)

# ===============================================================================
# 8. VISUALIZACIÓN DE DATOS ORIGINALES
# ===============================================================================

print("\nVisualizando imágenes originales...")
plt.figure(figsize=(20, 8))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X[i].reshape(100, 100), cmap="gray")
    plt.title(f"{'Gato' if y[i] == 0 else 'Perro'}")

plt.suptitle("Imágenes Originales")
plt.tight_layout()
plt.show()

# ===============================================================================
# 9. CONFIGURACIÓN DE DATA AUGMENTATION
# ===============================================================================

print("\n" + "="*50)
print("FASE 2: CONFIGURACIÓN DE DATA AUGMENTATION")
print("="*50)

# Crear generador de aumento de datos con varias transformaciones
datagen = ImageDataGenerator(
    rotation_range=30,          # Rotación hasta 30 grados
    width_shift_range=0.2,      # Desplazamiento horizontal 20%
    height_shift_range=0.2,     # Desplazamiento vertical 20%
    shear_range=15,             # Inclinación hasta 15 grados
    zoom_range=[0.7, 1.4],      # Zoom entre 70% y 140%
    horizontal_flip=True,       # Voltear horizontalmente
    vertical_flip=True          # Voltear verticalmente
)

# Ajustar el generador a nuestros datos
datagen.fit(X)

# Visualizar ejemplos de data augmentation
print("Generando ejemplos de Data Augmentation...")
plt.figure(figsize=(20, 8))

for imagen, etiqueta in datagen.flow(X, y, batch_size=10, shuffle=False):
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(imagen[i].reshape(100, 100), cmap="gray")
        plt.title(f"{'Gato' if etiqueta[i] == 0 else 'Perro'} (Aumentado)")
    break

plt.suptitle("Imágenes con Data Augmentation")
plt.tight_layout()
plt.show()

# ===============================================================================
# 10. NUEVOS MODELOS PARA DATA AUGMENTATION
# ===============================================================================

print("Creando nuevos modelos para entrenamiento con Data Augmentation...")

# Crear modelos idénticos para entrenamiento con augmentation
modeloDenso_AD = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(100, 100, 1)),
    tf.keras.layers.Dense(150, activation='relu'),
    tf.keras.layers.Dense(150, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

modeloCNN_AD = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

modeloCNN2_AD = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(250, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar los nuevos modelos
modeloDenso_AD.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

modeloCNN_AD.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

modeloCNN2_AD.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

# ===============================================================================
# 11. SEPARACIÓN DE DATOS PARA ENTRENAMIENTO CON AUGMENTATION
# ===============================================================================

print("Separando datos para entrenamiento con Data Augmentation...")

# Calcular índices de separación
total_datos = len(X)
indice_separacion = int(total_datos * 0.85)  # 85% para entrenamiento

print(f"Total de datos: {total_datos}")
print(f"Entrenamiento: {indice_separacion} imágenes")
print(f"Validación: {total_datos - indice_separacion} imágenes")

# Separar los datos
X_entrenamiento = X[:indice_separacion]
X_validacion = X[indice_separacion:]

y_entrenamiento = y[:indice_separacion]
y_validacion = y[indice_separacion:]

# Crear generador para datos de entrenamiento
data_gen_entrenamiento = datagen.flow(X_entrenamiento, y_entrenamiento, batch_size=32)

# ===============================================================================
# 12. ENTRENAMIENTO CON DATA AUGMENTATION
# ===============================================================================

print("\n" + "="*50)
print("FASE 3: ENTRENAMIENTO CON DATA AUGMENTATION")
print("="*50)

# Configurar TensorBoard para modelos con augmentation
tensorboardDenso_AD = TensorBoard(log_dir='logs/denso_AD')
tensorboardCNN_AD = TensorBoard(log_dir='logs/cnn_AD')
tensorboardCNN2_AD = TensorBoard(log_dir='logs/cnn2_AD')

# Calcular steps per epoch
steps_per_epoch = int(np.ceil(len(X_entrenamiento) / 32))
validation_steps = int(np.ceil(len(X_validacion) / 32))

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

# Entrenar Modelo Denso con Data Augmentation
print("\n--- Entrenando Modelo Denso con Data Augmentation ---")
historial_denso_ad = modeloDenso_AD.fit(
    data_gen_entrenamiento,
    epochs=20,  # Reducido para ejemplo
    batch_size=32,
    validation_data=(X_validacion, y_validacion),
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[tensorboardDenso_AD],
    verbose=1
)

# Recrear generador para CNN (los generadores se "agotan")
data_gen_entrenamiento = datagen.flow(X_entrenamiento, y_entrenamiento, batch_size=32)

# Entrenar Modelo CNN con Data Augmentation
print("\n--- Entrenando Modelo CNN con Data Augmentation ---")
historial_cnn_ad = modeloCNN_AD.fit(
    data_gen_entrenamiento,
    epochs=20,  # Reducido para ejemplo
    batch_size=32,
    validation_data=(X_validacion, y_validacion),
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[tensorboardCNN_AD],
    verbose=1
)

# Recrear generador para CNN2
data_gen_entrenamiento = datagen.flow(X_entrenamiento, y_entrenamiento, batch_size=32)

# Entrenar Modelo CNN2 con Data Augmentation
print("\n--- Entrenando Modelo CNN2 con Data Augmentation ---")
historial_cnn2_ad = modeloCNN2_AD.fit(
    data_gen_entrenamiento,
    epochs=20,  # Reducido para ejemplo
    batch_size=32,
    validation_data=(X_validacion, y_validacion),
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[tensorboardCNN2_AD],
    verbose=1
)

# ===============================================================================
# 13. EVALUACIÓN Y COMPARACIÓN DE RESULTADOS
# ===============================================================================

print("\n" + "="*50)
print("EVALUACIÓN Y COMPARACIÓN DE RESULTADOS")
print("="*50)

# Función para obtener la mejor precisión de validación
def obtener_mejor_precision(historial):
    return max(historial.history['val_accuracy'])

# Obtener mejores precisiones
precision_denso = obtener_mejor_precision(historial_denso)
precision_cnn = obtener_mejor_precision(historial_cnn)
precision_cnn2 = obtener_mejor_precision(historial_cnn2)

precision_denso_ad = obtener_mejor_precision(historial_denso_ad)
precision_cnn_ad = obtener_mejor_precision(historial_cnn_ad)
precision_cnn2_ad = obtener_mejor_precision(historial_cnn2_ad)

# Mostrar resultados
print("\n=== RESULTADOS SIN DATA AUGMENTATION ===")
print(f"Modelo Denso:  {precision_denso:.4f} ({precision_denso*100:.2f}%)")
print(f"Modelo CNN:    {precision_cnn:.4f} ({precision_cnn*100:.2f}%)")
print(f"Modelo CNN2:   {precision_cnn2:.4f} ({precision_cnn2*100:.2f}%)")

print("\n=== RESULTADOS CON DATA AUGMENTATION ===")
print(f"Modelo Denso:  {precision_denso_ad:.4f} ({precision_denso_ad*100:.2f}%)")
print(f"Modelo CNN:    {precision_cnn_ad:.4f} ({precision_cnn_ad*100:.2f}%)")
print(f"Modelo CNN2:   {precision_cnn2_ad:.4f} ({precision_cnn2_ad*100:.2f}%)")

print("\n=== MEJORA CON DATA AUGMENTATION ===")
print(f"Modelo Denso:  {(precision_denso_ad - precision_denso)*100:+.2f}%")
print(f"Modelo CNN:    {(precision_cnn_ad - precision_cnn)*100:+.2f}%")
print(f"Modelo CNN2:   {(precision_cnn2_ad - precision_cnn2)*100:+.2f}%")

# ===============================================================================
# 14. VISUALIZACIÓN DE CURVAS DE ENTRENAMIENTO
# ===============================================================================

print("\nGenerando gráficas de entrenamiento...")

# Función para graficar historial
def graficar_entrenamiento(historial, titulo):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Precisión
    ax1.plot(historial.history['accuracy'], label='Entrenamiento')
    ax1.plot(historial.history['val_accuracy'], label='Validación')
    ax1.set_title(f'{titulo} - Precisión')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Precisión')
    ax1.legend()
    ax1.grid(True)
    
    # Pérdida
    ax2.plot(historial.history['loss'], label='Entrenamiento')
    ax2.plot(historial.history['val_loss'], label='Validación')
    ax2.set_title(f'{titulo} - Pérdida')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Pérdida')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Graficar resultados sin augmentation
graficar_entrenamiento(historial_denso, "Modelo Denso")
graficar_entrenamiento(historial_cnn, "Modelo CNN")
graficar_entrenamiento(historial_cnn2, "Modelo CNN2")

# Graficar resultados con augmentation
graficar_entrenamiento(historial_denso_ad, "Modelo Denso con Data Augmentation")
graficar_entrenamiento(historial_cnn_ad, "Modelo CNN con Data Augmentation")
graficar_entrenamiento(historial_cnn2_ad, "Modelo CNN2 con Data Augmentation")

# ===============================================================================
# 15. GUARDAR MODELOS ENTRENADOS
# ===============================================================================

print("\nGuardando modelos entrenados...")

# Guardar modelos sin augmentation
modeloDenso.save('modelo_denso.h5')
modeloCNN.save('modelo_cnn.h5')
modeloCNN2.save('modelo_cnn2.h5')

# Guardar modelos con augmentation
modeloDenso_AD.save('modelo_denso_ad.h5')
modeloCNN_AD.save('modelo_cnn_ad.h5')
modeloCNN2_AD.save('modelo_cnn2_ad.h5')

print("Modelos guardados exitosamente!")

# ===============================================================================
# 16. FUNCIÓN DE PREDICCIÓN PARA NUEVAS IMÁGENES
# ===============================================================================

def predecir_imagen(modelo, ruta_imagen):
    """
    Función para predecir si una nueva imagen es un gato o perro
    
    Args:
        modelo: Modelo entrenado de Keras
        ruta_imagen: Ruta a la imagen a predecir
    
    Returns:
        str: 'Gato' o 'Perro' con el porcentaje de confianza
    """
    import cv2
    
    # Cargar y preprocesar imagen
    imagen = cv2.imread(ruta_imagen)
    imagen = cv2.resize(imagen, (100, 100))
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen = imagen.reshape(1, 100, 100, 1)
    imagen = imagen.astype('float32') / 255.0
    
    # Hacer predicción
    prediccion = modelo.predict(imagen)
    confianza = prediccion[0][0]
    
    if confianza < 0.5:
        return f"Gato ({(1-confianza)*100:.1f}% confianza)"
    else:
        return f"Perro ({confianza*100:.1f}% confianza)"

# Ejemplo de uso:
# resultado = predecir_imagen(modeloCNN2_AD, 'mi_imagen.jpg')
# print(resultado)

print("\n" + "="*50)
print("¡ENTRENAMIENTO COMPLETADO!")
print("="*50)
print("\nPara visualizar los resultados en TensorBoard, ejecuta:")
print("tensorboard --logdir logs")
print("\nLuego abre tu navegador en: http://localhost:6006")
print("\n¡Los modelos están listos para hacer predicciones!")