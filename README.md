# Modelo de Clasificación de Intenciones

Este proyecto utiliza **spaCy** para entrenar un modelo de clasificación de texto que identifica intenciones en frases relacionadas con correos electrónicos. El modelo puede clasificar frases en las siguientes categorías:

- **ENVIAR_CORREO**
- **HISTORIAL_CORREOS**
- **CORREOS_RECIBIDOS**
- **ABRIR_CORREO**
- **RESPONDER_CORREO**
- **REENVIAR_CORREO**
- **ELIMINAR_CORREO**
- **AGREGAR_EVENTO**
- **LISTAR_EVENTOS**
- **ELIMINAR_EVENTO**
- **ACTUALIZAR_EVENTO**

Repositorio del proyecto: [https://github.com/Santiago-Ospina-Gonzalez/ModeloAI](https://github.com/Santiago-Ospina-Gonzalez/ModeloAI)

---

## ✨ Estructura del Proyecto

- `main.py`: Entrena el modelo y realiza predicciones básicas.
- `test_model.py`: Contiene ejemplos de prueba para evaluar el modelo con frases de cada categoría.
- `modelo_intenciones/`: Carpeta donde se guarda el modelo entrenado.
- `requirements.txt`: Lista de dependencias necesarias para ejecutar el proyecto.
- `entrenamiento.jsonl`: Archivo con los datos de entrenamiento en formato JSONL, que incluye frases relacionadas con correos y eventos.
---

## 🚀 Requisitos

- Python 3.8 o superior.
- Dependencias listadas en `requirements.txt`.

---

## 🔧 Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/Santiago-Ospina-Gonzalez/ModeloAI
   cd ModeloAI
   ```

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Descarga el modelo base de spaCy para español:
   ```bash
   python -m spacy download es_core_news_sm
   ```

---

## ⚙️ Uso

### Entrenar el modelo

Ejecuta el archivo `main.py` para entrenar el modelo:
```bash
python main.py
```
El modelo entrenado se guardará en la carpeta `modelo_intenciones`.

### Probar el modelo

Ejecuta `test_model.py` para probar el modelo con ejemplos predefinidos:
```bash
python test_model.py
```
Esto mostrará las predicciones del modelo para cada frase de prueba, junto con la categoría predicha y el nivel de confianza.

---

## 📂 Formato de los Datos de Entrenamiento

El archivo `entrenamiento.jsonl` contiene frases etiquetadas con las categorías correspondientes. Ejemplo:

```json
{"text": "Quiero agregar un evento para mañana.", "cats": {"ENVIAR_CORREO": 0, "HISTORIAL_CORREOS": 0, "CORREOS_RECIBIDOS": 0, "ABRIR_CORREO": 0, "RESPONDER_CORREO": 0, "REENVIAR_CORREO": 0, "ELIMINAR_CORREO": 0, "AGREGAR_EVENTO": 1, "LISTAR_EVENTOS": 0, "ELIMINAR_EVENTO": 0, "ACTUALIZAR_EVENTO": 0}}
```

---

## 📊 Dependencias

- Python 3.8+
- spaCy
- es_core_news_sm
- spacy-lookups-data

---

## 💼 Contribuciones

Si deseas contribuir, por favor abre un issue o envía un pull request. Toda ayuda es bienvenida para mejorar este proyecto.

