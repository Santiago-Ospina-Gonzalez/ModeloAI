import spacy

def predict_intent(text):
    # Carga el modelo entrenado
    nlp = spacy.load("modelo_intenciones")
    doc = nlp(text)
    return doc.cats

if __name__ == "__main__":
    # Ejemplos de prueba para cada categoría
    ejemplos = [
        # ENVIAR_CORREO
        "Por favor, envía un correo a Juan.",
        "Necesito mandar un email urgente.",
        
        # HISTORIAL_CORREOS
        "Muéstrame el historial de correos enviados.",
        "Quiero revisar los correos antiguos.",
        
        # CORREOS_RECIBIDOS
        "¿Qué correos he recibido hoy?",
        "Dame la bandeja de entrada, por favor.",
        
        # ABRIR_CORREO
        "Abre el último correo que llegó.",
        "Quiero leer el mensaje más reciente.",
        
        # RESPONDER_CORREO
        "Responde este correo con más detalles.",
        "Quiero contestar este mensaje ahora mismo.",
        
        # REENVIAR_CORREO
        "Reenvía este correo a Ana, por favor.",
        "Necesito reenviar este email urgente.",
        
        # ELIMINAR_CORREO
        "Elimina este correo de mi bandeja de entrada.",
        "Por favor, borra este mensaje.",
        
        # AGREGAR_EVENTO
        "Quiero agregar un evento para mañana.",
        "Añade un evento a mi calendario para el lunes.",
        
        # LISTAR_EVENTOS
        "Muéstrame todos los eventos de esta semana.",
        "Quiero ver los eventos programados para hoy.",
        
        # ELIMINAR_EVENTO
        "Elimina el evento que tengo mañana.",
        "Borra el evento de la tarde, por favor.",
        
        # ACTUALIZAR_EVENTO
        "Actualiza el evento del lunes a las 3 PM.",
        "Cambia la hora del evento de mañana a las 5 PM."
    ]

    # Realiza predicciones para cada ejemplo
    for texto in ejemplos:
        prediccion = predict_intent(texto)
        # Encuentra la categoría con mayor probabilidad
        categoria_predicha = max(prediccion, key=prediccion.get)
        confianza = prediccion[categoria_predicha]
        
        print(f"Texto: '{texto}'")
        print(f"Predicción: {prediccion}")
        print(f"Categoría predicha: '{categoria_predicha}' con confianza de {confianza:.2f}")
        print("-" * 50)