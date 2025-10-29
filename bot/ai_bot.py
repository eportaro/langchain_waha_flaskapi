import os

from decouple import config

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

#Importanciones para trabajar con SUPABASE
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client


class AIBot:

    def __init__(self):
        self.__chat = ChatOpenAI(model= 'gpt-4o-mini')
        self.__retriever = self.__build_retriever()

    #Si vas a cambiar a Chroma o Pinecone o Qdrant, tienes que modificar esta función.
    def __build_retriever(self):
        embedding_model = OpenAIEmbeddings(model='text-embedding-ada-002')

        # Obtener credenciales de Supabase desde las variables de entorno
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")

        client = create_client(supabase_url, supabase_key)

        vector_store = SupabaseVectorStore(
            client=client,
            embedding=embedding_model,
            table_name="documents",
            query_name="match_documents"
        )

        return vector_store.as_retriever( #El retrieve busca los datos correspondientes en nuestro VectorStore
            search_kwargs={'k': 15}, #Busco hasta máximo 30 resultados de Chunks
        )
    

    def __build_messages(self, history_messages, question):
        messages = []
        
        # Validación y depuración
        if not history_messages:
            print("No hay historial de mensajes para procesar")
            messages.append(HumanMessage(content=question))
            return messages
        
        try:
            # Extraer información relevante del historial
            user_info = self.__extract_user_info(history_messages)
            print(f"Información extraída del usuario: {user_info}")
            
            for message in history_messages:
                # Verificamos que el mensaje tenga la estructura esperada
                if not isinstance(message, dict):
                    print(f"Mensaje no es un diccionario: {message}")
                    continue
                    
                # Verificamos que tenga las claves esperadas
                if 'body' not in message:
                    print(f"Mensaje sin clave 'body': {message}")
                    continue
                    
                # Verificamos que el body no sea None
                if message.get('body') is None:
                    print(f"Mensaje con 'body' None: {message}")
                    continue
                    
                # Usamos 'isUser' en lugar de 'fromMe'
                is_user = message.get('isUser', False)
                message_class = HumanMessage if is_user else AIMessage
                messages.append(message_class(content=message.get('body')))
                
            # Agregamos la pregunta actual
            messages.append(HumanMessage(content=question))
            
        except Exception as e:
            print(f"Error al construir mensajes: {e}")
            # En caso de error, al menos incluimos la pregunta actual
            messages = [HumanMessage(content=question)]
            
        return messages

    def __extract_user_info(self, history_messages):
        """Extrae información del usuario analizando todo el historial de mensajes"""
        user_info = {
            "nombre": None,
            "correo": None,
            "programa_interes": None
        }
        
        # Patrones de despedida
        despedida_patterns = [
            "eso es todo", "gracias por la información", 
            "muchas gracias", "eso sería todo", "adiós", 
            "hasta luego", "chao", "nos vemos", "listo",
            "bueno muchas gracias", "hasta pronto"
    ]

        # También analizar las respuestas del bot para detectar saludos personalizados
        bot_greeting_patterns = [
            "¡hola (.+)!",
            "hola (.+)!",
            "gracias por proporcionar tus datos, (.+)!"
        ]
        
        # Primero buscar en mensajes más recientes (ordenados de más nuevo a más antiguo)
        for message in reversed(history_messages):
            text = message.get('body', '').lower() if message.get('body') else ''
            
            # Si es mensaje del usuario
            if message.get('isUser', False) or message.get('sender') == 'user':
                # Buscar nombre
                if "mi nombre es" in text:
                    name_part = text.split("mi nombre es")[1].strip().split()[0]
                    if name_part and not user_info["nombre"]:
                        user_info["nombre"] = name_part.title()
                        
                # Buscar correo (mejorado para detectar emails)
                if "@" in text and "." in text and not user_info["correo"]:
                    words = text.split()
                    for word in words:
                        if "@" in word and "." in word:
                            user_info["correo"] = word
                            
                # Buscar programa de interés
                programs = ["ai engineer", "data engineer", "data analyst", "machine learning", "data scientist"]
                for program in programs:
                    if program in text.lower() and not user_info["programa_interes"]:
                        user_info["programa_interes"] = program
            
            # Si es mensaje del bot, buscar saludos personalizados
            elif not message.get('isUser', True) or message.get('sender') == 'bot':
                for pattern in bot_greeting_patterns:
                    import re
                    match = re.search(pattern, text.lower())
                    if match and not user_info["nombre"]:
                        user_info["nombre"] = match.group(1).title()
        
        # Añadir este bloque para detectar despedidas
        for message in history_messages:
            if message.get('isUser', False) or message.get('sender') == 'user':
                text = message.get('body', '').lower() if message.get('body') else ''
                
                # Verificar si es una despedida
                for pattern in despedida_patterns:
                    if pattern in text.lower():
                        user_info["despidiendose"] = True
                        break

        return user_info

    # Añadir esta nueva función
    def __generar_respuesta_despedida(self, nombre=None):
        """Genera una respuesta de despedida personalizada"""
        if nombre:
            return f"¡Ha sido un placer ayudarte, {nombre}! Gracias por contactar con DataPath. Si tienes más preguntas en el futuro o necesitas información adicional, no dudes en escribirnos nuevamente. ¡Que tengas un excelente día! 😊"
        else:
            return "¡Ha sido un placer ayudarte! Gracias por contactar con DataPath. Si tienes más preguntas en el futuro o necesitas información adicional, no dudes en escribirnos nuevamente. ¡Que tengas un excelente día! 😊"
    
    def invoke(self, history_messages, question):

        # Asegurarse de que history_messages no sea None
        if history_messages is None:
            history_messages = []
            
        # Agregar log para depuración
        print(f"AIBot.invoke recibió history_messages con {len(history_messages)} mensajes")
        
        # Extraer información del usuario para incluirla en el prompt
        user_info = self.__extract_user_info(history_messages)
        
        # Verificar si el usuario se está despidiendo
        if user_info.get("despidiendose", False) and "gracias" in question.lower():
            # Si es una despedida, generar respuesta personalizada directamente
            return self.__generar_respuesta_despedida(user_info.get("nombre"))

        SYSTEM_TEMPLATE = '''
        Eres un asistente especializado en resolver dudas sobre la empresa de educación online DataPath.

        **Rol y Objetivo**:
        1. Responde de forma natural, agradable y respetuosa a las preguntas o comentarios del usuario.
        2. MANTÉN Y USA EL CONTEXTO DE LA CONVERSACIÓN. Esto es crítico para dar una buena experiencia.
        3. Apóyate en el "contexto" (documentos relevantes) para resolver dudas específicas sobre DataPath.
        4. Mantén un tono amistoso y responde en español, usando emojis para mostrar cercanía cuando sea apropiado.

        **Información del usuario que has recopilado hasta ahora**:
        - Nombre: {user_info_nombre}
        - Correo: {user_info_correo}
        - Programa de interés: {user_info_programa}

        **Instrucciones Clave**:
        - IMPORTANTE: Si conoces el nombre del usuario, úsalo en tus respuestas para generar más confianza. Por ejemplo: "Hola {user_info_nombre}, aquí tienes la información..."
        - Si el usuario te pregunta si sabes su nombre, correo o intereses, responde con la información que tienes.
        - NO vuelvas a preguntar datos personales si ya los proporcionó.
        - Si ya se saludó o presentó anteriormente, NO repitas saludos. Continúa la conversación de manera fluida.

        **Instrucciones para despedidas**:
        - Si el usuario dice frases como "gracias, eso es todo", "listo", "ya no tengo más preguntas", "adiós", identifícalo como una despedida.
        - Al despedirte, agradece al usuario por su tiempo y SIEMPRE invítalo a volver si tiene más consultas en el futuro.
        - En las despedidas, usa frases como: "¡Gracias por contactarnos! Estamos aquí para cuando necesites más información." o "Ha sido un placer ayudarte. Puedes escribirnos nuevamente cuando tengas más consultas sobre DataPath."

        <context>
        {context}
        </context>
        '''

        try:
            # Obtener documentos relevantes desde Supabase
            docs = self.__retriever.invoke(question)

            # Agregamos print para depurar el contexto
            print("Contexto obtenido desde Supabase:", docs)

            # Construir los mensajes del historial y la pregunta
            constructed_messages = self.__build_messages(history_messages, question)
            print("Mensajes construidos para el prompt:", constructed_messages)
            
            # Formatear el template con la información del usuario
            formatted_template = SYSTEM_TEMPLATE.format(
                user_info_nombre=user_info["nombre"] or "No proporcionado aún",
                user_info_correo=user_info["correo"] or "No proporcionado aún",
                user_info_programa=user_info["programa_interes"] or "No proporcionado aún",
                context="{context}"  # Placeholder para que luego se rellene con los documentos
            )
            
            # Creamos el prompt del sistema para el chain
            question_answering_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        'system',
                        formatted_template,
                    ),
                    MessagesPlaceholder(variable_name='messages'),
                ]
            )

            document_chain = create_stuff_documents_chain(self.__chat, question_answering_prompt)
            
            # Invocamos la chain pasándole el contexto obtenido y el historial de mensajes
            response = document_chain.invoke(
                {
                    'context': docs,
                    'messages': constructed_messages,  # Usamos los mensajes ya construidos
                }
            )
            return response
            
        except Exception as e:
            print(f"Error en AIBot.invoke: {e}")
            # En caso de error, proporcionar una respuesta genérica
            return f"Lo siento, tuve un problema procesando tu consulta. ¿Podrías reformularla de otra manera? 😊"