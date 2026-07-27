import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gen_ai
 
from app import retriever, formatear_contexto
 
load_dotenv()
 
st.set_page_config(page_title="Chat de la empresa",
                   layout="centered")
 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
 
gen_ai.configure(api_key=GEMINI_API_KEY)
model = gen_ai.GenerativeModel("gemini-3.5-flash")
 
 
def map_role(role):
    return "assistant" if role == "model" else role
 
 
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])
 
# Historial "limpio" solo para mostrar en pantalla: guardamos acá la
# pregunta original del usuario, sin el contexto inyectado, para no
# mostrarle al usuario el bloque de contexto que se le manda al modelo.
if "historial_visible" not in st.session_state:
    st.session_state.historial_visible = []
 
st.title("Chatbot")
 
for mensaje in st.session_state.historial_visible:
    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["content"])
        if mensaje.get("citaciones"):
            with st.expander("Fuentes utilizadas"):
                for doc in mensaje["citaciones"]:
                    st.caption(doc.metadata.get("source", "N/A"))
 
user_input = st.chat_input("Escriba su consulta...")
 
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.historial_visible.append({"role": "user", "content": user_input})
 
    # Buscar contexto relevante con el retriever de app.py
    documentos_relacionados = retriever.invoke(user_input)
    contexto = formatear_contexto(documentos_relacionados) if documentos_relacionados else ""
 
    # Armar el mensaje que recibe el modelo con contexto
    if documentos_relacionados:
        mensaje_aumentado = (
            f"Contexto:\n{contexto}\n\n"
            f"Pregunta del empleado: {user_input}\n\n"
            "Respondé usando únicamente la información del contexto. "
            "Si no hay información suficiente, respondé solo \"No lo sé\"."
        )
    else:
        mensaje_aumentado = (
            f"Pregunta del empleado: {user_input}\n\n"
            "No se encontró contexto relevante en la documentación. "
            "Respondé solo \"No lo sé\"."
        )
 
    # Envía al chat_session manteniendo la memoria de la conversación
    response = st.session_state.chat_session.send_message(mensaje_aumentado)
 
    with st.chat_message("assistant"):
        st.markdown(response.text)
        if documentos_relacionados:
            with st.expander("Fuentes utilizadas"):
                for doc in documentos_relacionados:
                    st.caption(doc.metadata.get("source", "N/A"))
 
    st.session_state.historial_visible.append(
        {
            "role": "assistant",
            "content": response.text,
            "citaciones": documentos_relacionados,
        }
    )
