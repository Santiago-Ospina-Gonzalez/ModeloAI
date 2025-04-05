import spacy
from spacy.pipeline.textcat import Config
from spacy.training import Example
import json

def load_training_data(filepath):
    """Carga los datos de entrenamiento desde un archivo JSONL."""
    training_data = []
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            text, annotations = json.loads(line)
            training_data.append((text, annotations))
    return training_data

def train_intent_model():
    # Carga un modelo base de spaCy
    nlp = spacy.load("es_core_news_sm")  # Modelo preentrenado en español

    # Configuración del componente de categorización de texto
    textcat_config = {
        "threshold": 0.5,
        "model": {
            "@architectures": "spacy.TextCatBOW.v1",
            "exclusive_classes": True,
            "ngram_size": 1,
            "no_output_layer": False,
        },
    }
    textcat = nlp.add_pipe("textcat", config=textcat_config)

    # Define las etiquetas de intención
    textcat.add_label("ENVIAR_CORREO")
    textcat.add_label("HISTORIAL_CORREOS")
    textcat.add_label("CORREOS_RECIBIDOS")
    textcat.add_label("ABRIR_CORREO")
    textcat.add_label("RESPONDER_CORREO")
    textcat.add_label("REENVIAR_CORREO")
    textcat.add_label("ELIMINAR_CORREO")

    # Carga los datos de entrenamiento desde el archivo
    train_data = load_training_data("entrenamiento.jsonl")

    # Entrenamiento del modelo
    optimizer = nlp.begin_training()
    for epoch in range(10):  # Ajusta el número de épocas según sea necesario
        losses = {}
        for text, annotations in train_data:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], drop=0.5, losses=losses)
        print(f"Epoch {epoch + 1}, Losses: {losses}")

    # Guarda el modelo entrenado
    nlp.to_disk("modelo_intenciones")
    print("Modelo entrenado y guardado en 'modelo_intenciones'.")

def predict_intent(text):
    # Carga el modelo entrenado
    nlp = spacy.load("modelo_intenciones")
    doc = nlp(text)
    return doc.cats

if __name__ == "__main__":
    train_intent_model()
    # Ejemplo de predicción
    texto = "Quiero enviar un correo"
    prediccion = predict_intent(texto)
    print(f"Predicción para '{texto}': {prediccion}")
