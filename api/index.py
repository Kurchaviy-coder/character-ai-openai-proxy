from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from PyCharacterAI import get_client
import asyncio
from typing import List, Optional, AsyncGenerator
import json
from datetime import datetime

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str  # character_id
    messages: List[Message]
    stream: Optional[bool] = False
    chat_id: Optional[str] = None  # persistent chat id (важно хранить на клиенте)

async def get_token(authorization: Optional[str] = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization")
    return authorization.split(" ")[1]

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, token: str = Depends(get_token)):
    # валидация
    if not request.messages or request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Last message must be role:user")

    try:
        client = await get_client(token=token)  # get_client из PyCharacterAI
        char_id = request.model
        # последний пользовательский месседж
        user_msg = request.messages[-1].content

        # Если chat_id нет — создаём, и если в messages есть предыдущие user-ходы, пытаемся
        # "ре-гидратировать" чат последовательной отправкой предыдущих user->(модель ответит)
        if not request.chat_id:
            chat, greeting = await client.chat.create_chat(char_id)
            chat_id = chat.chat_id

            # Если есть пред. сообщения кроме последнего, воспроизводим их как user-ходы
            # ВАЖНО: мы не можем вставить "assistant" сообщения, которые были где-то ещё;
            # CharacterAI хранит assistant-ответы только если они были сгенерированы самим сервисом.
            # Поэтому при ре-гидратации мы отправляем только user-ходы (модель сгенерирует ответы,
            # и исторя в чат-объекте будет последовательной). Это не даёт точную копию чужих assistant-ответов.
            if len(request.messages) > 1:
                # Отправляем все user-сообщения кроме последнего, чтобы создать историю.
                # Это может породить ответы от модели, которые будут частью истории.
                for msg in request.messages[:-1]:
                    if msg.role == "user":
                        # не стримим здесь — просто создаём历史
                        await client.chat.send_message(char_id, chat_id, msg.content)
                    else:
                        # если у тебя есть assistant-сообщения, их нельзя "вставить" в Character.ai.
                        # Пропускаем — можно позже добавить логику summary/injection, но это не то же самое.
                        continue
        else:
            chat_id = request.chat_id

        # STREAMING
        if request.stream:
            # Используем send_message(..., streaming=True) — это async iterable (как в README PyCharacterAI).
            async def stream_gen() -> AsyncGenerator[str, None]:
                # Получаем async iterator из PyCharacterAI
                answer_iter = await client.chat.send_message(char_id, chat_id, user_msg, streaming=True)
                # answer_iter — async iterable, где каждое сообщение — частично обновляемый объект
                # Мы будем пробегать по нему и отдавать текущий текст как delta.
                try:
                    async for partial in answer_iter:
                        # partial.get_primary_candidate().text — полный текст на данный момент
                        text = partial.get_primary_candidate().text
                        delta = {"content": text}
                        chunk_data = json.dumps({
                            "id": f"chatcmpl-{datetime.now().timestamp()}",
                            "object": "chat.completion.chunk",
                            "created": int(datetime.now().timestamp()),
                            "model": request.model,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": None}]
                        })
                        # SSE формат: каждое событие начинается с "data: "
                        yield f"data: {chunk_data}\n\n"
                    # После окончания
                    done_payload = json.dumps({"id": "chatcmpl-done", "object": "chat.completion.done"})
                    yield f"data: {done_payload}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    # в случае ошибки отправим её как событие и завершим
                    err = {"error": str(e)}
                    yield f"data: {json.dumps(err)}\n\n"
                finally:
                    # закрываем сессию клиента аккуратно
                    try:
                        await client.close_session()
                    except Exception:
                        pass

            # SSE media type — text/event-stream
            return StreamingResponse(stream_gen(), media_type="text/event-stream")

        # NON-STREAM (обычный режим)
        else:
            answer = await client.chat.send_message(char_id, chat_id, user_msg, streaming=False)
            response_text = answer.get_primary_candidate().text

            # Возвращаем chat_id, чтобы клиент мог сохранить его и присылать дальше
            return JSONResponse({
                "id": f"chatcmpl-{datetime.now().timestamp()}",
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": request.model,
                "chat_id": chat_id,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            })

    except Exception as e:
        # Логируем реальную ошибку в detail
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models(token: str = Depends(get_token)):
    # Заглушка — безопасно возвращаем пустую модель-список
    return {
        "object": "list",
        "data": [{"id": "default-char", "object": "model"}]
    }

@app.get("/")
async def root():
    return {"message": "Character AI OpenAI Proxy ready. POST /v1/chat/completions"}
