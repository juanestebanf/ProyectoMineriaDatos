import streamlit as st
from transformers import pipeline
from PIL import Image

# Configuración de la página
st.set_page_config(page_title="Detector de Lesiones de Piel")
st.title("Detector de Lesiones de Piel (ViT)")

# Traducción de etiquetas
traduccion = {
    "benign_keratosis-like_lesions": "Queratosis Benigna (tipo verruga / seborreica)",
    "basal_cell_carcinoma": "Carcinoma Basocelular",
    "actinic_keratoses": "Queratosis Actínica (Pre-cáncer)",
    "vascular_lesions": "Lesiones Vasculares",
    "melanocytic_Nevi": "Nevus Melanocítico (Lunar común)",
    "melanoma": "Melanoma",
    "dermatofibroma": "Dermatofibroma"
}

# Información breve por clase
info_lesiones = {
    "benign_keratosis-like_lesions": "Crecimiento benigno común en adultos mayores. Suele ser verrugoso, elevado, marrón o negro. No es canceroso y rara vez requiere tratamiento salvo por estética o irritación.",
    "basal_cell_carcinoma": "El cáncer de piel más frecuente. Crece lentamente, casi nunca hace metástasis. Suele aparecer como nódulo brillante, perlado o úlcera que no cicatriza. Tratamiento temprano suele ser curativo.",
    "actinic_keratoses": "Lesión precancerosa por daño solar acumulado. Manchas ásperas, escamosas, rosadas o rojizas. Sin tratamiento, un pequeño porcentaje puede evolucionar a cáncer escamoso.",
    "vascular_lesions": "Generalmente benignas (ej. hemangiomas). Pueden ser manchas rojas o violáceas. Rara vez malignas.",
    "melanocytic_Nevi": "Lunar común. Generalmente benigno. Vigilar cambios según la regla ABCDE.",
    "melanoma": "Cáncer de piel agresivo. Puede hacer metástasis. La detección temprana es clave.",
    "dermatofibroma": "Nódulo benigno firme, usualmente marrón. No es canceroso."
}

# ===============================
# CARGA DEL MODELO DESDE HUGGING FACE
# ===============================
@st.cache_resource
def load_vit_model():
    return pipeline(
        "image-classification",
        model="Anwarkh1/Skin_Cancer-Image_Classification"
    )

classifier = load_vit_model()

# Subir imagen
uploaded_file = st.file_uploader(
    "Sube la imagen de la lesión...",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None and classifier is not None:
    img = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(img, caption="Imagen subida", use_container_width=True)

    with col2:
        with st.spinner("Analizando la imagen, espere..."):
            results = classifier(img)

        st.subheader("Resultados del análisis:")

        results = sorted(results, key=lambda x: x["score"], reverse=True)

        for i, res in enumerate(results):
            label_original = res["label"]
            label_espanol = traduccion.get(label_original, label_original)
            score = res["score"]

            if i == 0:
                st.success(
                    f"**Predicción principal: {label_espanol}** "
                    f"({round(score * 100, 1)}% de confianza)"
                )

                descripcion = info_lesiones.get(
                    label_original,
                    "No hay descripción disponible para esta categoría."
                )
                st.info(descripcion)

                st.markdown(
                    "**Atención:** Este resultado es solo una predicción del modelo. "
                    "**NO es un diagnóstico médico**."
                )
            else:
                st.write(f"{label_espanol} — {round(score * 100, 1)}%")
                st.progress(float(score))

            st.write("---")

# Advertencia final
st.divider()

st.warning("""
**IMPORTANTE – LEE ESTO CON ATENCIÓN**  
Este es un prototipo educativo basado en inteligencia artificial y minería de datos.  
**NO sustituye la opinión de un médico especialista**.

- Ante resultados preocupantes, acude a un dermatólogo.
- No tomes decisiones médicas basándote solo en esta herramienta.
- El sistema puede cometer errores.
""")

st.caption("Modelo basado en Vision Transformer (ViT)")
