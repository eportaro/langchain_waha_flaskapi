from langchain.tools import tool
from langchain.tools import StructuredTool

from utils.download_youtube_yt_dlp import YoutubeDownloader
from utils.audio import Audio
from utils.crea_partes_notas import Notes

from utils.envio_correo import EnvioCorreo
from utils.registro_google_sheet import RegistroGoogleSheet

from bot.ai_bot import AIBot

import os

class DataPathTools:
    #============================================================================
    @tool
    def bajar_video_de_youtube(link: str) -> str:
        """Descarga un video desde un enlace de YouTube y devuelve la ruta del archivo descargado."""
        video_path = YoutubeDownloader().bajar_video(link)
        return video_path
    
    @tool
    def extraer_audio(video_path):
        """Extrae el audio de un video y lo guarda en formato WAV."""
        audio_path = Audio.extraer(video_path)
        return audio_path
    
    @tool
    def transcribir_audio(audio_path: str) -> str:
        """Transcribe un archivo de audio guardado en audio_path a texto."""
        transcripcion_path = Audio.transcribir(audio_path)
        if not os.path.exists(transcripcion_path):
            raise FileNotFoundError(f"El archivo de transcripción no existe: {transcripcion_path}")
        return transcripcion_path
    
    @tool
    def guardar_nota(transcripcion_path):
        """Guarda el texto final en un archivo de texto dentro del directorio _notas y además devuelve los resumenes al usuario."""
        notas = Notes.guardar_nota(transcripcion_path)
        return notas
    #==========================================================================

    @staticmethod
    def enviar_correo_func(nombre_lead: str, correo_lead: str, mensaje_para_lead: str):
        """Envía un correo necesitando solo el nombre del interesado, el correo del interesado y un mensaje para el interesado que va a depender del programa en el cuál él tenga el interés."""
        envio = EnvioCorreo()
        envio.enviar_correo(nombre_lead, correo_lead, mensaje_para_lead)
    
    enviar_correo = StructuredTool.from_function(
        enviar_correo_func,
        name="enviar_correo",
        description="Envía un correo necesitando solo el nombre del interesado, el correo del interesado y un mensaje para el interesado que va a depender del programa en el cuál él tenga el interés."
    )

    @staticmethod
    def registrar_google_sheet_func(nombre: str, correo: str, programa: str):
        """Registra los datos del interesado pidiendo nombre, correo y programa, estos 3 datos son los que registra en su hoja."""
        registro = RegistroGoogleSheet()
        registro.registrar_google_sheets(nombre,correo,programa)
    
    # Crear la herramienta estructurada
    registrar_google_sheet = StructuredTool.from_function(
        registrar_google_sheet_func,
        name="registrar_google_sheet",
        description="Registra los datos del interesado pidiendo nombre, correo y programa en Google Sheets."
    )
    
    # Agregar RAG como Tool
    @staticmethod
    def consultar_DataPath(query: str, history_messages=None) -> str:
        """Usa el sistema RAG para buscar información sobre DataPath y devuelve la respuesta."""
        # Imprime lo que recibe la tool en el parámetro history_messages
        print("En la tool 'consultar_DataPath', history_messages recibido:")
        print(history_messages)
        
        rag_instance = AIBot()  # Inicializar el sistema RAG
        
        # Usa historial de mensajes si está disponible
        if history_messages:
            try:
                # No necesitamos convertir el formato, ya que AIBot.__build_messages ya lo hace
                # Simplemente pasamos el history_messages tal como viene
                response = rag_instance.invoke(history_messages, query)
            except Exception as e:
                print(f"Error al procesar el historial en consultar_DataPath: {e}")
                # Fallback a invocación sin historial
                response = rag_instance.invoke([], query)
        else:
            response = rag_instance.invoke([], query)
            
        return response

    