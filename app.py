from flask import Flask, request, jsonify

from bot.ai_bot import AIBot
from services.waha import Waha

import time

import random

from agent_3_completo import DataPath

#Histórico del Chat
from utils.db_utils import store_chat_history, get_chat_history

app = Flask(__name__)


@app.route('/chatbot/webhook/', methods=['POST']) #Endpoint
def webhook():
    data = request.json

    #Para no enviar mensaje de respuesta a los Grupos de Whatsapp que tengo -------------------------
    chat_id = data['payload']['from']
    received_message = data['payload']['body']

    is_group = '@g.us' in chat_id
    if is_group:
        return jsonify({'status': 'success', 'message': 'Mensaje de grupo/status ignorada'}), 200
    #------------------------------------------------------------------------------------------------

    print(f'EVENTO RECIBIDO: {data}')

    waha = Waha()
    #ai_bot = AIBot() #Este es solo para probar el RAG, sólo el RAG
    bot = DataPath()
    agente, tools = bot.crear_agente()

    # Indica "escribiendo" en WhatsApp
    waha.start_typing(chat_id=chat_id)

    # 1) Guardar mensaje de usuario en Supabase
    store_chat_history(chat_id, "user", received_message)

    # 2) Obtener historial desde Supabase (últimos 10 mensajes)
    historial = get_chat_history(chat_id=chat_id, limit=10)
    print("Historial recuperado:", historial)

    #--------------------------------- 3) Test del RAG como Tool del Agente------------------------------
    try:
        resultado = bot.procesar_mensaje(received_message, agente, tools, history_messages=historial)
        response_message = resultado.get("output", "No se pudo procesar el mensaje correctamente.")
    except Exception as e:
        print(f"Error al procesar el mensaje: {e}")
        response_message = f"Ocurrió un error al procesar tu mensaje: {str(e)}"
    #--------------------------------------------------------------------------------------------------

    # 4) Guardar mensaje del bot en Supabase
    store_chat_history(chat_id, "bot", response_message)

    # 5) Enviar respuesta al usuario por WhatsApp
    waha.send_message(
        chat_id=chat_id,
        message=response_message,
    )

    waha.stop_typing(chat_id=chat_id)

    return jsonify({'status': 'success'}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)