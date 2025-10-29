# db_utils.py

import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

def get_supabase_client() -> Client:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    return create_client(supabase_url, supabase_key)

def store_chat_history(chat_id: str, sender: str, message: str) -> None:
    client = get_supabase_client()
    data = {
        "chat_id": chat_id,
        "sender": sender,
        "message": message
    }
    response = client.table("chat_history").insert(data).execute()
    
    # Comprobar si el objeto response tiene el atributo 'error'
    if hasattr(response, 'error') and response.error is not None:
        print("Error al almacenar el mensaje:", response.error)
    else:
        print("Mensaje almacenado correctamente:", response.data)


def get_chat_history(chat_id: str, limit: int = 10) -> list:
    client = get_supabase_client()
    response = (
        client.table("chat_history")
        .select("*")
        .eq("chat_id", chat_id)
        .order("created_at", desc=True)  # Orden cronológico descendente (los más nuevos primero)
        .limit(limit)
        .execute()
    )

    
    # Si el objeto response tiene un atributo 'error' y no es None,
    # consideramos que hubo un error.
    if hasattr(response, "error") and response.error is not None:
        print("Error al obtener el histórico:", response.error)
        return []
    
    # Si no hay error, extraemos los datos
    data = response.data or []
    messages = []
    for row in data:
        # Si 'sender' es "user", consideramos que es un mensaje del usuario.
        is_user = (row["sender"] == "user")
        messages.append({
            "body": row["message"],
            "isUser": is_user
        })
    return messages
