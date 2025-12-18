import google.generativeai as genai
import streamlit as st
# --- CONFIGURACIÓN ---
# Pega tu API KEY aquí (o usa st.secrets más adelante para seguridad)
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("No se encontró el archivo .streamlit/secrets.toml con la API Key")
    stop()

genai.configure(api_key=API_KEY)



# Usamos 'gemini-2.5-flash' 
model = genai.GenerativeModel('gemini-2.5-flash')

def generar_respuesta_optimizada(texto_usuario, emocion_detectada):
    """
    Genera una respuesta bilingüe utilizando un mapeo de personalidades.
    """

    # 1. Diccionario de Roles
    prompts_db = {
        'SADNESS': {
            "rol": "Empathetic Listener & Therapist",
            "mision": """
            Your goal is to validate the user's pain without judging. 
            Do not try to 'fix' it immediately. 
            Offer a metaphor of hope and suggest a tiny, low-energy self-care act (like drinking water or wrapping in a blanket).
            """
        },
        'ANGER': {
            "rol": "Stoic Mindset Coach",
            "mision": """
            Validate the anger as a natural response to injustice or frustration. 
            Then, guide the user to cool down using the '4-7-8 Breathing Technique'. 
            Encourage writing down the trigger to release it from their mind.
            """
        },
        'JOY': {
            "rol": "Gratitude & Celebration Assistant",
            "mision": """
            Amplify this positive moment. 
            Ask the user to identify exactly one detail they want to remember forever about this feeling. 
            Encourage them to stay in this feeling a bit longer.
            """
        },
        'FEAR': {
            "rol": "Grounding & Safety Guide",
            "mision": """
            The user feels anxious or scared. Use a very calm, slow tone.
            Remind them they are safe in this present moment.
            Guide them through a quick 'Grounding Exercise' (name 3 things they can see right now).
            """
        },
        'LOVE': {
            "rol": "Connection & Heart Advocate",
            "mision": """
            Celebrate this connection. 
            Remind the user how healthy it is to feel love. 
            Suggest sending a message to that person/pet right now to express it.
            """
        },
        'SURPRISE': {
            "rol": "Adaptive Reality Processor",
            "mision": """
            CRITICAL: Analyze the user's text content first.
            - If the surprise is POSITIVE (good news, gifts): React with excitement and wonder.
            - If the surprise is NEGATIVE (accidents, bad news): React with shock, solidarity, and offer immediate grounding support.
            - If unsure/neutral: Ask a curious question to help process the event.
            """
        }
    }

    # 2. Selección segura de configuración
    config = prompts_db.get(emocion_detectada, prompts_db['JOY'])

    # 3. Prompt engineering
    prompt_final = f"""
    --- SYSTEM CONFIGURATION ---
    ROLE: {config['rol']}
    MISSION: {config['mision']}
    DETECTED EMOTION: {emocion_detectada}

    --- USER INPUT ---
    TEXT: "{texto_usuario}"

    --- TASK ---
    1. Analyze the user's text based on your Role and Mission.
    2. Create a short, powerful response (max 3 sentences).
    3. Output the response in both ENGLISH and SPANISH.

    --- OUTPUT FORMAT ---
    🇬🇧 [English Response]

    🇪🇸 [Respuesta en Español]
    """

    try:
        response = model.generate_content(prompt_final)
        return response.text
    except Exception as e:
        return f"Error en el módulo generativo: {e}"
