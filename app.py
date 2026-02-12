import streamlit as st
from transformers import pipeline
from PIL import Image

st.set_page_config(page_title="Detector de Lesiones de Piel", layout="wide")

st.title("Detector de Lesiones de Piel con Vision Transformer (ViT)")

st.markdown(
    "<hr style='margin-top:10px;margin-bottom:30px;'>",
    unsafe_allow_html=True
)

# ===============================
# TRADUCCIONES
# ===============================
traduccion = {
    "benign_keratosis-like_lesions": "Queratosis Benigna (tipo verruga / seborreica)",
    "basal_cell_carcinoma": "Carcinoma Basocelular",
    "actinic_keratoses": "Queratosis Actínica (Pre-cáncer)",
    "vascular_lesions": "Lesiones Vasculares",
    "melanocytic_Nevi": "Nevus Melanocítico (Lunar común)",
    "melanoma": "Melanoma",
    "dermatofibroma": "Dermatofibroma"
}

# ===============================
# INFORMACIÓN DE LESIONES
# ===============================
info_lesiones = {
    "benign_keratosis-like_lesions": "Crecimiento benigno común en adultos mayores. No es canceroso.",
    "basal_cell_carcinoma": "El cáncer de piel más frecuente. Crece lentamente y suele ser tratable.",
    "actinic_keratoses": "Lesión precancerosa causada por daño solar acumulado.",
    "vascular_lesions": "Generalmente benignas. Suelen ser manchas rojizas o violáceas.",
    "melanocytic_Nevi": "Lunar común. Generalmente benigno.",
    "melanoma": "Cáncer de piel agresivo. La detección temprana es clave.",
    "dermatofibroma": "Nódulo benigno firme, no canceroso."
}

# ===============================
# COLUMNAS SUPERIORES
# ===============================
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### Clases que el modelo puede identificar:")
    for clase in traduccion.values():
        st.markdown(f"- {clase}")

with col_right:
    st.markdown("### Uso exclusivo para lesiones cutáneas humanas")
    st.info("""
Este modelo fue entrenado exclusivamente con imágenes clínicas de lesiones dermatológicas humanas.

**Solo se deben subir fotografías donde:**
- Se observe claramente una lesión en piel humana. La imagen esté enfocada en la lesión. No existan objetos externos dominantes  

**No subir:**
- Animales | Objetos  
- Paisajes | Fotografías sin lesión clara 

El sistema no es un clasificador universal.
""")

st.markdown(
    "<hr style='margin-top:20px;margin-bottom:30px;'>",
    unsafe_allow_html=True
)

# ===============================
# EJEMPLOS VISUALES
# ===============================
st.markdown("### Ejemplos de imágenes válidas")

def cargar_y_redimensionar(ruta, size=(250, 250)):
    img = Image.open(ruta)
    img = img.resize(size)
    return img

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.image(cargar_y_redimensionar("ejemplosFotos/melanoma1.webp"), caption="Melanoma")

with col2:
    st.image(cargar_y_redimensionar("ejemplosFotos/lunarcomunn.png"), caption="Nevus Melanocítico")

with col3:
    st.image(cargar_y_redimensionar("ejemplosFotos/carcinoma.png"), caption="Carcinoma Basocelular")

with col4:
    st.image(cargar_y_redimensionar("ejemplosFotos/Queratoss.jpeg"), caption="Queratosis Actínica")

with col5:
    st.image(cargar_y_redimensionar("ejemplosFotos/dermatofbroma.png"), caption="Dermatofibroma")

st.markdown(
    "<hr style='margin-top:30px;margin-bottom:30px;'>",
    unsafe_allow_html=True
)

# ===============================
# CARGA DEL MODELO
# ===============================
@st.cache_resource
def load_vit_model():
    return pipeline(
        "image-classification",
        model="Fuentesjes/SkinCancer-ViT",
        device=-1
    )

classifier = load_vit_model()

# ===============================
# SUBIDA DE IMAGEN
# ===============================
st.markdown("## Analizar nueva imagen")

uploaded_file = st.file_uploader(
    "Sube la imagen de la lesión...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and classifier is not None:

    st.warning("Asegúrese de que la imagen corresponda a una lesión cutánea humana.")

    img = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(img, caption="Imagen subida", use_container_width=True)

    with col2:
        with st.spinner("Analizando la imagen..."):
            results = classifier(img)

        st.subheader("Resultados del análisis:")

        results = sorted(results, key=lambda x: x["score"], reverse=True)

        for i, res in enumerate(results):
            label_original = res["label"]
            label_espanol = traduccion.get(label_original, label_original)
            score = res["score"]

            if i == 0:
                st.success(
                    f"Predicción principal: {label_espanol} "
                    f"({round(score * 100, 1)}% de confianza)"
                )

                descripcion = info_lesiones.get(
                    label_original,
                    "No hay descripción disponible para esta categoría."
                )
                st.info(descripcion)

                st.markdown(
                    "**Nota:** Este resultado es solo una predicción del modelo. "
                    "**NO constituye un diagnóstico médico.**"
                )
            else:
                st.write(f"{label_espanol} — {round(score * 100, 1)}%")
                st.progress(float(score))

            st.write("---")

# ===============================
# ADVERTENCIA FINAL
# ===============================
st.markdown(
    "<hr style='margin-top:30px;margin-bottom:20px;'>",
    unsafe_allow_html=True
)

st.warning("""
IMPORTANTE  
Este es un prototipo educativo basado en inteligencia artificial y minería de datos.  
NO sustituye la evaluación de un médico especialista.

Ante cualquier cambio en una lesión, consulte a un dermatólogo.
""")

st.caption("Modelo basado en Vision Transformer (ViT) - Autoría propia")
