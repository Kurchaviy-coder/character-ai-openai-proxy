from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PyCharacterAI import get_client
import asyncio
from typing import List, Optional, AsyncGenerator
import json
from datetime import datetime
import uuid # Добавлен импорт для уникальных ID

app = FastAPI()

# --- Модели данных Pydantic ---

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str  # character_id, например "YntB_3_f-Yv2h29d3M_e-1aW3P_G1b4zAn2M2y3kG4"
    messages: List[Message]
    stream: Optional[bool] = False
    chat_id: Optional[str] = None  # Для продолжения существующего чата

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
        chat_id = request.chat_id

        # --- Логика обработки истории ---
        # Проблема: API OpenAI без состояний (передает всю историю), а Character.AI - с состояниями (использует chat_id).
        # Решение: Лучший способ - это когда клиент (Risu.ai) сохраняет chat_id из ответа и присылает его обратно.
        # Этот код будет работать наиболее эффективно, если клиент поддерживает такое поведение.
        
        if not chat_id:
            # Если chat_id не предоставлен, создаем новый чат.
            # В этом случае ИИ не будет иметь контекста предыдущих сообщений.
            chat, _ = await client.chat.create_chat(char_id)
            chat_id = chat.chat_id

        user_msg = request.messages[-1].content

        # --- Логика потокового ответа (Streaming) ---
        if request.stream:
            stream_id = f"chatcmpl-{uuid.uuid4()}"
            created_ts = int(datetime.now().timestamp())

            async def stream_generator() -> AsyncGenerator[str, None]:
                try:
                    # Используем потоковый метод клиента
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
                                }],
                                "chat_id": chat_id # <-- ИЗМЕНЕНИЕ: Добавляем chat_id, чтобы клиент его получил
                            }
                            yield f"data: {json.dumps(response_chunk)}\n\n"
                    
                    # Отправляем финальный блок с причиной завершения
                    final_chunk = {
                        "id": stream_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "chat_id": chat_id # <-- ИЗМЕНЕНИЕ: Добавляем chat_id также и в финальный чанк
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    print(f"Ошибка во время потоковой передачи: {e}")

            # Важно: media_type должен быть 'text/event-stream' для SSE
            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        # --- Логика обычного (непотокового) ответа ---
        else:
            answer = await client.chat.send_message(char_id, chat_id, user_msg)
            response_text = answer.get_primary_candidate().text if answer.get_primary_candidate() else ""

            # Возвращаем chat_id в ответе. Клиент должен его сохранить и использовать для следующего запроса.
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
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "chat_id": chat_id  # Пользовательское поле для сохранения состояния
            }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """Возвращает список доступных моделей (персонажей)."""
    # Это заглушка. Вам нужно заменить "YOUR_CHARACTER_ID" на реальный ID вашего персонажа.
    # Клиент выберет эту "модель" для начала чата.
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

