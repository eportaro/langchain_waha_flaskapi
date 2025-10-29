from time import sleep

from langchain.agents import (
    AgentExecutor,
    create_tool_calling_agent
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

from tools_3_completo import DataPathTools


class DataPath:
    def __init__(self):
        self.llm = ChatOpenAI(model= 'gpt-4o-mini') # no olvides adicionar tu api_key en el .env
        self.tool = DataPathTools()
        self.history_messages = None  # Almacenamos el historial aquí


    def crear_agente(self):

        # Creamos un closure para pasar el historial de mensajes a la herramienta consultar_DataPath
        def consultar_DataPath_with_history(query: str) -> str:
            """Usa el sistema RAG para responder consultas sobre DataPath."""
            return DataPathTools.consultar_DataPath(query, self.history_messages)

        tools = [
            Tool(name="bajar_video_youtube", func=DataPathTools.bajar_video_de_youtube, description="Descarga un video de YouTube y devuelve la ruta del archivo descargado para que sea usado por otra tool."),
            Tool(name="extraer_audio_video", func=DataPathTools.extraer_audio, description="Extrae el audio de un archivo MP4 ubicado en una carpeta local, que podría haber sido extraído del youtube por otra Tool."),
            #Tool(name="describir_imagen", func=DataPathTools.describe_imagen, description="Analiza la imagen en image_path y devuelve una descripción detallada. Si la imagen contiene texto, será transcrito."),
            Tool(name="transcribir_audio", func=DataPathTools.transcribir_audio, description="Transcribe un archivo de audio guardado en audio_path a texto utilizando reconocimiento de voz."),
            Tool(name="guardar_nota", func=DataPathTools.guardar_nota, description="Guarda el texto en un archivo de texto dentro del directorio de notas y además devuelve ese texto o nota resumen al usuario."),
            DataPathTools.enviar_correo,  # Ya es un StructuredTool
            DataPathTools.registrar_google_sheet,  # Ya es un StructuredTool
            Tool(name="consultar_DataPath",func=consultar_DataPath_with_history, description="Usa el sistema RAG para responder consultas sobre DataPath.")
        ]

        llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Eres un asistente amigable llamado DataBot.

                    Tienes tres roles principales:

                    1️⃣ **Responder preguntas sobre DataPath:**  
                        - Usa la herramienta 'consultar_DataPath' para responder cualquier consulta sobre cursos, programas y servicios.  
                        - Continúa respondiendo preguntas mientras el usuario siga consultando sobre DataPath.  

                    2️⃣ **Conectar al usuario con un asesor:**  
                        - Si el usuario expresa interés en comunicarse con un asesor o recibir información personalizada, solicita su **nombre completo**, **correo electrónico** y el **programa de interés**.  
                        - Solo después de obtener estos datos, usa 'registrar_google_sheet' pasando los argumentos como campos separados (**nombre**, **correo**, **programa**) para registrar la información. 
                        - Es decir asegúrate de darle el formato correspondiente para la entrada de la función "registrar_google_sheet".
                        - Luego, usa 'a_enviar_correo' para notificar al usuario que un asesor se pondrá en contacto.  

                    3️⃣ **Procesar contenido multimedia:**  
                        - Si el usuario proporciona un enlace de YouTube o un archivo multimedia (MP4, MP3, OGG):  
                            - Usa 'bajar_video_youtube' para descargar el video (si es un enlace de YouTube).  
                            - Usa 'extraer_audio_video' para extraer el audio de videos MP4/YouTube.  
                            - Usa 'transcribir_audio' para convertir el audio en texto.  
                            - Finalmente, usa 'guardar_nota' para crear y guardar una nota basada en la transcripción.  
                            - Entrega el resultado de la nota creada y guardada en español, aquella que obtuviste basada en la transcripción.  

                    ⚡️ **Reglas Importantes:**  
                    - Antes de comenzar a interactuar, revisa el historial de la conversación (variable {chat_history}).  
                        - Si existe historial, utiliza esa información para no saludar de nuevo o volver a solicitar datos personales que ya se han proporcionado.
                        - Si no hay historial, da la bienvenida y solicita los datos necesarios.
                    - Responde siempre en español y usa emojis para mantener un tono amigable.
                    - Si el mensaje del usuario es ambiguo, pide más detalles de forma precisa.
                    - Detecta claramente la intención del usuario: si está preguntando, quiere contacto o procesar contenido. 
                    - Responde siempre en español y usa emojis para mantener un tono amigable. 😊 

                    Recuerda:  
                    - Si ya has saludado o el usuario ya se identificó en una conversación anterior, no repitas el saludo ni los pedidos de datos.
                    - Tu respuesta debe integrarse con la información presente en el historial para continuar la conversación de forma coherente.
                    """
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )


        agent = create_tool_calling_agent(
            llm=llm,
            tools=tools,
            prompt=prompt,
        )
        return agent, tools

    def procesar_mensaje(self, msg, agente, tools, history_messages=None):
        
        # Guardamos el historial de mensajes en la instancia
        self.history_messages = history_messages

        # Verificamos si history_messages llega correctamente
        if history_messages is None:
            print("No se recibió historial (history_messages es None)")
            messages = []
        else:
            print("Historial recibido desde app.py:")
            for idx, message in enumerate(history_messages):
                # Imprime índice, cuerpo y la clave que usamos (asegúrate de que los mensajes tienen la clave 'isUser')
                print(f"{idx}: {message}")

        # Se reconstruye la lista de mensajes para el prompt. 
        # Nota: Si antes usabas 'fromMe' y ahora usas 'isUser', asegúrate de que todos tus mensajes tengan la clave correcta.
        messages = []
        for message in history_messages:
            # Usamos 'isUser' para determinar el tipo de mensaje
            message_class = HumanMessage if message.get('isUser') else AIMessage
            messages.append(message_class(content=message.get('body')))
        messages.append(HumanMessage(content=msg))

        """Procesa el mensaje recibido vía WhatsApp y llama a la herramienta correcta."""
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agente,
            tools=tools,
            verbose=True,
        )


        # Prompt mejorado
        executor_prompt = {
            "input": (
                f"Analiza el siguiente mensaje y decide qué acción tomar:\n\n"

                f"1️⃣ **Si es una consulta general sobre DataPath:**\n"
                f"- Usa 'consultar_DataPath' para responder.\n"
                f"- Continúa respondiendo mientras el usuario siga haciendo preguntas.\n\n"

                f"2️⃣ **Si el usuario quiere hablar con un asesor o recibir información personalizada:**\n"
                f"- Solicita el **nombre completo**, **correo electrónico** y **programa de interés**.\n"
                f"- Solo después de obtener estos datos completos:\n"
                f"  - Usa 'registrar_google_sheet' para registrar al usuario.\n"
                f"  - Usa 'a_enviar_correo' para enviar una notificación.\n\n"

                f"3️⃣ **Si el mensaje contiene contenido multimedia (YouTube, MP4, MP3, OGG):**\n"
                f"- Si es un enlace de YouTube:\n"
                f"  - Usa 'bajar_video_youtube' para descargar el video.\n"
                f"  - Luego, usa 'extraer_audio_video' para extraer el audio.\n"
                f"  - Usa 'transcribir_audio' para transcribir el audio.\n"
                f"  - Finalmente, usa 'guardar_nota' para crear una nota.\n\n"

                f"- Si el usuario envía un archivo MP4:\n"
                f"  - Extrae el audio usando 'extraer_audio_video'.\n"
                f"  - Transcribe el audio con 'transcribir_audio'.\n"
                f"  - Guarda la nota usando 'guardar_nota'.\n\n"

                f"- Si el usuario envía un archivo MP3/OGG:\n"
                f"  - Transcribe directamente el audio usando 'transcribir_audio'.\n"
                f"  - Guarda la nota usando 'guardar_nota'.\n\n"

                f"4️⃣ **Reglas Generales:**\n"
                f"- No pidas datos personales a menos que el usuario exprese interés en comunicarse con un asesor.\n"
                f"- Si el mensaje es ambiguo, pide más detalles.\n"
                f"- Responde siempre en español, manteniendo un tono amigable y usando emojis. 😊\n\n"

                f"💬 **Mensaje del usuario:**\n{msg}"
            )
        }

        
        resultado = agent_executor.invoke(executor_prompt)


        return resultado