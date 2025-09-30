from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
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
    model: str  # character_id, e.g. "abc123"
    messages: List[Message]
    stream: Optional[bool] = False
    chat_id: Optional[str] = None  # Для persistent чата, если есть

async def get_token(authorization: Optional[str] = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization")
    return authorization.split(" ")[1]

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, token: str = Depends(get_token)):
    if not request.messages or request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Last message must be user")

    try:
        client = await get_client(token=token)
        char_id = request.model
        user_msg = request.messages[-1].content

        # Создаём чат если нет chat_id
        if not request.chat_id:
            chat, _ = await client.chat.create_chat(char_id)
            chat_id = chat.chat_id
        else:
            chat_id = request.chat_id

        if request.stream:
            async def stream_gen() -> AsyncGenerator[str, None]:
                answer = await client.chat.send_message(char_id, chat_id, user_msg, streaming=True)
                printed_length = 0  # Для симуляции, но в API просто yield chunks
                async for message_part in answer:
                    text = message_part.get_primary_candidate().text
                    delta = {"content": text[printed_length:]}
                    printed_length = len(text)
                    chunk_data = json.dumps({
                        "id": f"chatcmpl-{datetime.now().timestamp()}",
                        "object": "chat.completion.chunk",
                        "created": int(datetime.now().timestamp()),
                        "model": request.model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}]
                    })
                    yield f"data: {chunk_data}\n\n"
                # Финальный chunk с finish и chat_id
                final_delta = {"content": "", "chat_id": chat_id}
                final_data = json.dumps({
                    "id": f"chatcmpl-{datetime.now().timestamp()}",
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now().timestamp()),
                    "model": request.model,
                    "choices": [{"index": 0, "delta": final_delta, "finish_reason": "stop"}]
                })
                yield f"data: {final_data}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_gen(), media_type="text/plain")

        else:
            answer = await client.chat.send_message(char_id, chat_id, user_msg)
            response_text = answer.get_primary_candidate().text

            return {
                "id": f"chatcmpl-{datetime.now().timestamp()}",
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "chat_id": chat_id  # Добавлено для persistence
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models(token: str = Depends(get_token)):
    # Заглушка: вернёт твой character_id как модель
    return {
        "object": "list",
        "data": [{"id": "default-char", "object": "model"}]
    }

@app.get("/")
async def root():
    return {"message": "Character AI OpenAI Proxy ready. POST /v1/chat/completions"}
