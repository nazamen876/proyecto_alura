from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from typing import Dict

import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model= "gemini-3.5-flash",
    temperature = 0,
    google_api_key = GEMINI_API_KEY
)

docs = []

for documento in Path("docs").glob("*.pdf"):
  try:
    loader = PyMuPDFLoader(str(documento))
    docs.extend(loader.load())
    print(f"Archivo cargado: {documento.name}")
  except Exception as e:
    print(f"Error cargando archivo: {documento.name}: {e} ")

print(f"Total de archivos cargados: {len(docs)}")

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = splitter.split_documents(docs)

if not chunks:
  print("No hay archivos")
else:
  for chunk in chunks:
    print(chunk)
    print("-------------------------")

modelo_embeddings = GoogleGenerativeAIEmbeddings(
    model = "models/gemini-embedding-001",
    google_api_key = GEMINI_API_KEY
)

vectorstore = FAISS.from_documents(chunks, modelo_embeddings)

retriever = vectorstore.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {"score_threshold": 0.3, "k" : 4}
)




prompt_rag = ChatPromptTemplate(
    [
        ("system",
         """
         Eres el especialista en RR.HH. de la empresa Carraro Desarrollo de Software.
         Responde siempre utilizando los conocimientos del contexto que te fuere dado.
         Si no hay información sobre la pregunta pasada en los datos responde solo "No lo sé".
         """),
        ("human", "Contexto: {context}. \n Pregunta del empleado : {input}")
    ]
)

document_chain = create_stuff_documents_chain(llm, prompt=prompt_rag)

def formatear_contexto(documentos) -> str:
  """Junta el contenido de los documentos recuperados en un solo texto,
  para poder pasárselo como contexto a cualquier modelo (LangChain o
  google-generativeai directo)."""
  return "\n\n".join(doc.page_content for doc in documentos)


def busqueda_de_preguntas_RAG(pregunta) -> Dict:
  documentosRelacionados = retriever.invoke(pregunta)

  if not documentosRelacionados:
    return {
        "respuesta" : "No lo sé.",
        "citaciones" : [],
        "documentos_encontrados" : False
    }

  answer = document_chain.invoke({
      "input" : pregunta,
      "context" : documentosRelacionados
  })

  if answer.rstrip(".!?") == "No lo sé":
    return {
        "respuesta" : "No lo sé.",
        "citaciones" : [],
        "documentos_encontrados" : False
    }

  return {
        "respuesta" : answer,
        "citaciones" : documentosRelacionados,
        "documentos_encontrados" : True
    }

#mensajes_de_prueba = [
#    "Puedo obtener un reembolso por el internet de mi home office?",
#    "Quiero una excepción para teletrabajar durante 5 días",
#    "Como funciona la política de comidas para viajes?",
#    "Existe una política para anticipos de vacaciones?",
#    "Quien es Napoleón Bonaparte"
#]

#for pregunta in mensajes_de_prueba:
#  respuesta_RAG = busqueda_de_preguntas_RAG(pregunta)
#  print(f"PREGUNTA: {pregunta}")
#  print(f"RESPUESTA: {respuesta_RAG['respuesta']}")
#  if respuesta_RAG["documentos_encontrados"]:
#    for i, citacion in enumerate(respuesta_RAG['citaciones']):
#      print(f"CITACION {i+1}: ")
#      print(f"Camino del documento: {citacion.metadata['file_path']}")
#      print(f"Contenido: {citacion.page_content}")
#      print("------------------------------------------------------------")