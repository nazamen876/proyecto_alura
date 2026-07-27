# Guía de instalación 

Como requisito previo se debe tener python instalado.

### Crear un entorno virtual en la carpeta raíz del proyecto y activarlo
```bash
  py -m venv venv
  venv\Scripts\activate
```

### Instalar los requerimientos del proyecto
```bash
  py -m pip install -r requirements.txt
```

### Crear un archivo .env en la carpeta raíz y definir una llave API
```bash
  # Dentro del archivo .env
  GEMINI_API_KEY = <Tu clave API>
```

El proyecto puede correrse localmente mediante streamlit con el siguiente comando:

```bash
  streamlit run main.py
```
