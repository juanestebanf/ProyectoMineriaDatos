# Clasificación de Lesiones Cutáneas con Vision Transformer (ViT)

Proyecto de Minería de Datos enfocado en la **clasificación automática de lesiones dermatológicas** mediante Deep Learning utilizando un modelo **Vision Transformer (ViT)**.

La aplicación permite analizar imágenes clínicas de lesiones cutáneas humanas y estimar la probabilidad de pertenencia a una de las categorías dermatológicas entrenadas.

---

 **Este sistema NO realiza diagnóstico médico.**  
Es un prototipo académico de apoyo.

---

## Clases que el modelo puede detectar

El modelo fue entrenado con imágenes clínicas del dataset HAM10000 y puede clasificar:

- Melanoma
- Nevus Melanocítico (lunar común)
- Carcinoma Basocelular
- Queratosis Actínica
- Lesiones Vasculares
- Queratosis Benigna
- Dermatofibroma

---

## Alcance y Limitaciones (Importante)

Este modelo:

- Fue entrenado exclusivamente con imágenes dermatológicas humanas.
- No detecta objetos.
- No identifica si la imagen pertenece al dominio médico.
- No está diseñado para clasificar animales, objetos o fotografías generales.

Si se carga una imagen fuera del dominio (por ejemplo, un animal u objeto), el modelo igualmente devolverá una predicción, ya que los clasificadores siempre asignan una clase basada en patrones visuales aprendidos.

Esto no significa que el modelo esté “fallando”, sino que está operando fuera de su dominio de entrenamiento.

---

## Arquitectura

- **Modelo:** Vision Transformer (ViT)
- **Tipo de aprendizaje:** Transfer Learning + Fine Tuning
- **Framework:** Hugging Face Transformers
- **Interfaz:** Streamlit
- **Lenguaje:** Python

---

## Demo en Producción

Aplicación desplegada en Streamlit Cloud:

https://proyectomineriadatos-detecciontiposcancer.streamlit.app/

---

##  Estructura del Proyecto
PROYECTOMINERIADATOS/
│── modelo_vit/
├── app.py
├── requirements.txt
├── README.md
├── ejemplosFotos/


Los archivos de pesos del modelo no están incluidos en el repositorio debido a las limitaciones de tamaño de GitHub.  
El modelo se encuentra alojado en Hugging Face.

---
##  Instalar dependencias
pip install -r requirements.txt

##  Ejectar app
streamlit run app.py
