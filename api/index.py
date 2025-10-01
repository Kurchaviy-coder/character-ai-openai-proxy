# app.py
import os
import aiosqlite
import asyncio
import uuid
import json
from datetime import datetime
from typing import List, Optional, AsyncGenerator
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential
from PyCharacterAI import get_client

DB_PATH = os.environ.get("CHAT_DB", "chat_sync.db")

app = FastAPI()

# -- In-memory locks to avoid concurrent replay to same character/chat --
_locks: dict[str, asyncio.Lock] = {}

def get_lock_for(character_id: str) -> asyncio.Lock:
    lock = _locks.get(character_id)
    if lock is None:
        lock = asyncio.Lock()
        _locks[character_id] = lock
    return lock

# --- Pydantic models ---
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
            synced INTEGER DEFAULT 0, -- 0/1
            external_ref TEXT
        );
        """)
        await db.commit()

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

async def find_message_by_external_ref(external_ref: str) -> Optional[str]:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT id FROM messages WHERE external_ref = ? LIMIT 1", (external_ref,))
        row = await cur.fetchone()
        return row[0] if row else None

async def mark_message_synced(message_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE messages SET synced = 1 WHERE id = ?", (message_id,))
        await db.commit()

async def fetch_unsynced_external_messages(character_id: str, chat_id: str, exclude_id: Optional[str] = None, limit: int = 50):
    query = """
        SELECT id, role, content FROM messages
        WHERE character_id = ? AND chat_id = ? AND source = 'external' AND synced = 0
    """
    params = [character_id, chat_id]
    if exclude_id:
        query += " AND id != ?"
        params.append(exclude_id)
    query += " ORDER BY created_at ASC LIMIT ?"
    params.append(limit)
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(query, tuple(params))
        rows = await cur.fetchall()
    return rows  # list of tuples (id, role, content)

# --- CharacterAI helpers ---
async def get_or_create_chat(client, character_id: str):
    chat_id = await get_mapping(character_id)
    if chat_id:
        return chat_id
    chat, _ = await client.chat.create_chat(character_id)
    chat_id = chat.chat_id
    await set_mapping(character_id, chat_id)
    return chat_id

# retry helper for unstable network calls
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=20))
async def send_message_with_retry(client, char_id: str, chat_id: str, content: str):
    return await client.chat.send_message(char_id, chat_id, content)

async def replay_unsynced_to_characterai(client, character_id: str, chat_id: str, exclude_id: Optional[str] = None, limit: int = 50):
    """
    Реплеим все внешние сообщения (unsynced) в порядке, кроме exclude_id (если передан).
    Возвращаем последний ответ (текст) если был.
    """
    rows = await fetch_unsynced_external_messages(character_id, chat_id, exclude_id, limit)
    last_response_text = ""
    for row in rows:
        message_id, role, content = row
        # поддерживаем простую логику: реплейим user-сообщения в CharacterAI
        if role != "user":
            # простая политика: помечаем как synced, но не реплеим
            await mark_message_synced(message_id)
            continue
        try:
            ans = await send_message_with_retry(client, character_id, chat_id, content)
            response_text = ""
            try:
                response_text = ans.get_primary_candidate().text if ans.get_primary_candidate() else ""
            except Exception:
                # если структура другая — попробуем взять text напрямую (без падений)
                response_text = getattr(ans, "text", "") or ""
            if response_text:
                await save_message_db(character_id, chat_id, "assistant", response_text, "characterai", synced=1)
            # помечаем исходное как synced
            await mark_message_synced(message_id)
            last_response_text = response_text
        except Exception as e:
            # Логируем и прерываем, чтобы не нарушить порядок
            print(f"[replay] error sending message to CharacterAI: {e}")
            break
    return last_response_text

# --- Dependencies ---
async def get_token(authorization: Optional[str] = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Неверный заголовок Authorization. Ожидается 'Bearer <token>'")
    return authorization.split(" ")[1]

# идемпотентность ключ
async def get_idempotency_key(idempotency_key: Optional[str] = Header(None), external_ref: Optional[str] = Header(None)):
    # поддерживаем оба заголовка: Idempotency-Key и External-Ref
    return idempotency_key or external_ref

# --- Main endpoint ---
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, token: str = Depends(get_token), idempotency: Optional[str] = Depends(get_idempotency_key)):
    # валидация
    if not request.messages or request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Последнее сообщение должно быть от пользователя ('user').")
    client = await get_client(token=token)
    char_id = request.model
    # гарантируем chat_id
    chat_id = await get_or_create_chat(client, char_id)

    # идемпотентность / дедуп через external_ref
    external_ref = idempotency
    if external_ref:
        existing = await find_message_by_external_ref(external_ref)
        if existing:
            # Если уже есть — возвращаем последнюю assistant-реплику, если есть
            async with aiosqlite.connect(DB_PATH) as db:
                cur = await db.execute("""
                    SELECT content FROM messages
                    WHERE character_id = ? AND chat_id = ? AND role = 'assistant'
                    ORDER BY created_at DESC LIMIT 1
                """, (char_id, chat_id))
                row = await cur.fetchone()
                last = row[0] if row else ""
            return {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": request.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": last}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }

    # сохраняем входящее сообщение как external unsynced
    user_msg = request.messages[-1].content
    incoming_msg_id = await save_message_db(char_id, chat_id, "user", user_msg, "external", synced=0, external_ref=external_ref)

    lock = get_lock_for(char_id)
    async with lock:
        # Для стриминга: сначала реплейим все предыдущие unsynced (кроме текущего), затем стримим текущий
        if request.stream:
            stream_id = f"chatcmpl-{uuid.uuid4()}"
            created_ts = int(datetime.now().timestamp())

            # Реплей предыдущих unsynced (чтобы CharacterAI "видел" всю историю)
            await replay_unsynced_to_characterai(client, char_id, chat_id, exclude_id=incoming_msg_id, limit=200)

            # Теперь стримим только текущий user_msg
            async def stream_generator() -> AsyncGenerator[str, None]:
                buffer = ""
                try:
                    async for chunk in client.chat.send_message_stream(char_id, chat_id, user_msg):
                        if chunk.text:
                            buffer += chunk.text
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
                    # По окончании стрима — сохраняем финальный ассистент-ответ в БД и помечаем user как synced
                    if buffer:
                        await save_message_db(char_id, chat_id, "assistant", buffer, "characterai", synced=1)
                    await mark_message_synced(incoming_msg_id)
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
                    print(f"Stream error: {e}")
                    # корректный финал в случае ошибки
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                return

            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            # непоточный: реплейим все unsynced (включая текущий) и возвращаем последний ответ
            last = await replay_unsynced_to_characterai(client, char_id, chat_id, exclude_id=None, limit=200)
            return {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": request.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": last}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }

# --- Debug endpoints ---
@app.get("/debug/history/{character_id}")
async def debug_history(character_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT id, role, content, source, created_at, synced, external_ref FROM messages WHERE character_id = ? ORDER BY created_at ASC", (character_id,))
        rows = await cur.fetchall()
        keys = ["id", "role", "content", "source", "created_at", "synced", "external_ref"]
        return [dict(zip(keys, r)) for r in rows]

@app.get("/")
async def root():
    return {"message": "Character AI to OpenAI-like proxy (SQLite local). Use POST /v1/chat/completions"}
