import streamlit as st
from cerebro import ClasificadorEmociones
from generador import generar_respuesta_optimizada
import time

# Configuración de página
st.set_page_config(page_title="Empathic AI", page_icon="🧠")

# Título y Descripción
st.title("🧠 Empathic AI")
st.subheader("Sistema de Análisis Emocional y Respuesta Dirigida")
st.markdown("""
Esta aplicación integra dos Inteligencias Artificiales:
1. **BERT (Local):** Analiza tu texto y clasifica tu emoción.
2. **Gemini (Cloud):** Genera una respuesta empática personalizada basada en esa emoción.
""")

# Carga del modelo (con Cache para que no recargue cada vez)
@st.cache_resource
def cargar_modelo():
    return ClasificadorEmociones()

with st.spinner("Cargando cerebro emocional (BERT)..."):
    clf = cargar_modelo()

# Área de interacción
texto = st.text_area("¿Cómo te sientes hoy? (Escribe en Inglés)", height=100, placeholder="Ej: I feel overwhelmed with my exams...")

col1, col2 = st.columns([1, 4])

with col1:
    analizar = st.button("Analizar", type="primary")

if analizar and texto:
    # 1. Análisis con BERT
    st.markdown("---")
    st.write("### 🔍 Análisis del Modelo Interno")
    
    inicio = time.time()
    emocion, confianza = clf.predecir(texto)
    tiempo = time.time() - inicio
    
    # Métricas visuales
    st.metric(label="Emoción Detectada", value=emocion)
    st.progress(confianza, text=f"Nivel de Confianza: {confianza:.1%}")
    st.caption(f"Tiempo de inferencia BERT: {tiempo:.4f} seg")
    
    # 2. Generación con Gemini
    st.markdown("---")
    st.write("### 🤖 Respuesta Generativa (Gemini)")
    
    with st.spinner(f"Generando respuesta para {emocion}..."):
        respuesta = generar_respuesta_optimizada(texto, emocion)
        
        # Mostramos la respuesta en una cajita bonita
        st.info(respuesta, icon="✨")

elif analizar and not texto:
    st.warning("Por favor, escribe algo primero.")