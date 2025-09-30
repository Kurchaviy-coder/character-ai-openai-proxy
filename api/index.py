from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PyCharacterAI import get_client
import asyncio
from typing import List, Optional, AsyncGenerator
import json
from datetime import datetime
import uuid

app = FastAPI()

# --- Глобальное хранилище для ID чатов ---
# Это простое решение для хранения сессий в памяти сервера.
# При перезапуске сервера "память" персонажей будет сброшена.
chat_sessions = {}

# --- Модели данных Pydantic ---

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str  # character_id, например "YntB_3_f-Yv2h29d3M_e-1aW3P_G1b4zAn2M2y3kG4"
    messages: List[Message]
    stream: Optional[bool] = False
    # chat_id больше не используется клиентом, сервер управляет им сам
    # chat_id: Optional[str] = None

# --- Зависимости FastAPI ---

async def get_token(authorization: Optional[str] = Header(None)) -> str:
    """Извлекает токен аутентификации из заголовка."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Неверный заголовок Authorization. Ожидается 'Bearer <token>'")
    return authorization.split(" ")[1]

# --- Эндпоинты API ---

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, token: str = Depends(get_token)):
    """Основной эндпоинт для обработки чатов, совместимый с OpenAI."""
    if not request.messages or request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Последнее сообщение должно быть от пользователя ('user').")

    try:
        client = await get_client(token=token)
        char_id = request.model
        
        # --- ИСПРАВЛЕННАЯ ЛОГИКА УПРАВЛЕНИЯ ПАМЯТЬЮ ---
        # Мы используем словарь в памяти сервера для отслеживания chat_id для каждого персонажа.
        # Это исправляет ошибку 'chat2' и решает проблему с памятью.
        
        chat_id = chat_sessions.get(char_id)

        if not chat_id:
            # Если для этого персонажа еще нет сессии, создаем новую.
            # Используем старый, надежный метод v1 API.
            print(f"Активная сессия для {char_id} не найдена. Создание нового чата.")
            chat, _ = await client.chat.create_chat(char_id)
            chat_id = chat.chat_id
            chat_sessions[char_id] = chat_id # Сохраняем ID для будущих запросов
            print(f"Новая сессия создана: {char_id} -> {chat_id}")

        user_msg = request.messages[-1].content

        # --- Логика потокового ответа (Streaming) ---
        if request.stream:
            stream_id = f"chatcmpl-{uuid.uuid4()}"
            created_ts = int(datetime.now().timestamp())

            async def stream_generator() -> AsyncGenerator[str, None]:
                try:
                    # Используем потоковый метод клиента из v1 API
                    async for chunk in client.chat.send_message_stream(char_id, chat_id, user_msg):
                        if chunk.text:
                            response_chunk = {
                                "id": stream_id,
                                "object": "chat.completion.chunk",
                                "created": created_ts,
                                "model": request.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": chunk.text},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(response_chunk)}\n\n"
                    
                    final_chunk = {
                        "id": stream_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    print(f"Ошибка во время потоковой передачи: {e}")

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        # --- Логика обычного (непотокового) ответа ---
        else:
            # Используем обычный метод клиента из v1 API
            answer = await client.chat.send_message(char_id, chat_id, user_msg)
            response_text = answer.get_primary_candidate().text if answer.get_primary_candidate() else ""

            return {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """Возвращает список доступных моделей (персонажей)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "YntB_3_f-Yv2h29d3M_e-1aW3P_G1b4zAn2M2y3kG4", # <-- ВАЖНО: Замените на ID вашего персонажа
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "user"
            }
        ]
    }

@app.get("/")
async def root():
    return {"message": "Character AI to OpenAI proxy is ready. Use POST /v1/chat/completions"}

