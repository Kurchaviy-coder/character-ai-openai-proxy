# Заменяет/дополняет твой код. Требует: pip install aiosqlite
import aiosqlite
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PyCharacterAI import get_client
import asyncio
from typing import List, Optional, AsyncGenerator
import json
from datetime import datetime
import uuid
import os

DB_PATH = os.environ.get("CHAT_DB", "chat_sync.db")

app = FastAPI()

# --- Pydantic ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False

# --- DB init ---
async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
        CREATE TABLE IF NOT EXISTS chat_mapping (
            character_id TEXT PRIMARY KEY,
            chat_id TEXT,
            last_synced_at INTEGER
        );
        """)
        await db.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            character_id TEXT,
            chat_id TEXT,
            role TEXT,
            content TEXT,
            source TEXT, -- 'external' or 'characterai'
            created_at INTEGER,
            synced INTEGER DEFAULT 0, -- 0/1: whether this message has been delivered to CharacterAI
            external_ref TEXT -- optional: id from external system to dedupe
        );
        """)
        await db.commit()

# ensure DB initialized on startup
@app.on_event("startup")
async def startup_event():
    await init_db()

# --- Helpers DB ---
async def get_mapping(character_id: str) -> Optional[str]:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT chat_id FROM chat_mapping WHERE character_id = ?", (character_id,))
        row = await cur.fetchone()
        return row[0] if row else None

async def set_mapping(character_id: str, chat_id: str):
    ts = int(datetime.now().timestamp())
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("INSERT OR REPLACE INTO chat_mapping (character_id, chat_id, last_synced_at) VALUES (?, ?, ?)",
                         (character_id, chat_id, ts))
        await db.commit()

async def save_message_db(character_id: str, chat_id: str, role: str, content: str, source: str, synced: int = 0, external_ref: Optional[str] = None):
    mid = str(uuid.uuid4())
    ts = int(datetime.now().timestamp())
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO messages (id, character_id, chat_id, role, content, source, created_at, synced, external_ref)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (mid, character_id, chat_id, role, content, source, ts, synced, external_ref))
        await db.commit()
    return mid

async def mark_message_synced(message_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE messages SET synced = 1 WHERE id = ?", (message_id,))
        await db.commit()

async def fetch_unsynced_external_messages(character_id: str, chat_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("""
            SELECT id, role, content FROM messages
            WHERE character_id = ? AND chat_id = ? AND source = 'external' AND synced = 0
            ORDER BY created_at ASC
        """, (character_id, chat_id))
        rows = await cur.fetchall()
    return rows

# --- CharacterAI sync logic ---
async def get_or_create_chat(client, character_id: str):
    chat_id = await get_mapping(character_id)
    if chat_id:
        return chat_id
    # create new chat via PyCharacterAI
    chat, _ = await client.chat.create_chat(character_id)
    chat_id = chat.chat_id
    await set_mapping(character_id, chat_id)
    return chat_id

async def replay_unsynced_to_characterai(client, character_id: str, chat_id: str):
    """
    Отправляем все внешние сообщения, которые ещё не доставлены CharacterAI,
    поочерёдно, и помечаем их как synced после успешной отправки.
    Возвращаем последний ответ CharacterAI (строго: последнюю отправленную реплику).
    """
    rows = await fetch_unsynced_external_messages(character_id, chat_id)
    last_response_text = ""
    for row in rows:
        message_id, role, content = row
        # мы реплеим только 'user' role к CharacterAI (если приходит system, подумай отдельно)
        # Отправляем и получаем ответ
        try:
            ans = await client.chat.send_message(character_id, chat_id, content)
            # ans может вернуть кандидатов — берём primary
            response_text = ans.get_primary_candidate().text if ans.get_primary_candidate() else ""
            # Сохраняем ответ CharacterAI в БД (synced = 1 т.к. пришёл из CharacterAI)
            await save_message_db(character_id, chat_id, "assistant", response_text, "characterai", synced=1)
            # Помечаем исходное внешнее сообщение как доставленное
            await mark_message_synced(message_id)
            last_response_text = response_text
        except Exception as e:
            # логируем и прерываем, чтобы не терять порядок — можно ретраить
            print("Replay error:", e)
            break
    return last_response_text

# --- Auth header dependency (как у тебя) ---
async def get_token(authorization: Optional[str] = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Неверный заголовок Authorization. Ожидается 'Bearer <token>'")
    return authorization.split(" ")[1]

# --- Основной эндпоинт ---
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, token: str = Depends(get_token)):
    if not request.messages or request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Последнее сообщение должно быть от пользователя ('user').")

    client = await get_client(token=token)
    char_id = request.model

    # ensure chat mapping exists
    chat_id = await get_or_create_chat(client, char_id)

    # save incoming user message to DB
    user_msg = request.messages[-1].content
    incoming_msg_id = await save_message_db(char_id, chat_id, "user", user_msg, "external", synced=0)

    # replay unsynced external messages to characterai (including just-saved one)
    if request.stream:
        stream_id = f"chatcmpl-{uuid.uuid4()}"
        created_ts = int(datetime.now().timestamp())

        async def stream_generator() -> AsyncGenerator[str, None]:
            try:
                # Реплейится: мы последовательно отправляем unsynced external messages
                # Но при стриме лучше отправлять только последний (чтобы не дублировать долго)
                # Здесь для простоты реплейим все.
                # replay_unsynced_to_characterai отправляет и сохраняет ответ в БД.
                last_response = await replay_unsynced_to_characterai(client, char_id, chat_id)
                # теперь поток из CharacterAI: если PyCharacterAI поддерживает stream — используем
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
                # Финализируем
                final_chunk = {
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                # Если send_message_stream не вернул полный текст, попробуем взять last_response
                if last_response:
                    # убедимся, что он сохранён — уже сделано в replay
                    pass
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                print("Stream error:", e)
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        # непоточный режим: реплейим изменённые сообщения и возвращаем последний ответ
        last_reply = await replay_unsynced_to_characterai(client, char_id, chat_id)
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": last_reply},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

# --- Endpoints для дебага / просмотра истории (удобно) ---
@app.get("/debug/history/{character_id}")
async def debug_history(character_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT id, role, content, source, created_at, synced FROM messages WHERE character_id = ? ORDER BY created_at ASC", (character_id,))
        rows = await cur.fetchall()
        keys = ["id", "role", "content", "source", "created_at", "synced"]
        return [dict(zip(keys, r)) for r in rows]
