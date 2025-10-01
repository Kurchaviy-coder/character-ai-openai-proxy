# api/index.py
import os
import sys
import traceback
import uuid
import json
import aiosqlite
import asyncio
from datetime import datetime
from typing import List, Optional, AsyncGenerator
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

# ВАЖНО: PyCharacterAI импортуем, но не инициализируем (get_client вызывается поздно)
from PyCharacterAI import get_client

APP = FastAPI()

# Path для SQLite — /tmp безопасен в серверлес
TMP_DIR = os.environ.get("TMPDIR") or os.environ.get("TMP") or "/tmp"
DB_PATH = os.environ.get("CHAT_DB", os.path.join(TMP_DIR, "chat_sync.db"))
ERROR_LOG_DIR = os.path.join(TMP_DIR, "app_logs")
os.makedirs(ERROR_LOG_DIR, exist_ok=True)

# Локальная инициализация БД (lazy)
_db_init_lock = asyncio.Lock()
_db_initialized = False

async def ensure_db():
    global _db_initialized
    if _db_initialized:
        return
    async with _db_init_lock:
        if _db_initialized:
            return
        try:
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
                    source TEXT,
                    created_at INTEGER,
                    synced INTEGER DEFAULT 0,
                    external_ref TEXT
                );
                """)
                await db.commit()
            _db_initialized = True
        except Exception as e:
            tb = traceback.format_exc()
            log_path = os.path.join(ERROR_LOG_DIR, f"db_init_{uuid.uuid4().hex}.log")
            try:
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(tb)
            except Exception:
                pass
            raise RuntimeError(f"DB init failed, see {log_path}") from e

# Простые helper'ы для БД
async def save_message_db(character_id: str, chat_id: str, role: str, content: str, source: str, synced: int = 0, external_ref: Optional[str] = None):
    await ensure_db()
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
    await ensure_db()
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT id FROM messages WHERE external_ref = ? LIMIT 1", (external_ref,))
        row = await cur.fetchone()
        return row[0] if row else None

async def mark_message_synced(message_id: str):
    await ensure_db()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE messages SET synced = 1 WHERE id = ?", (message_id,))
        await db.commit()

async def fetch_unsynced_external_messages(character_id: str, chat_id: str, exclude_id: Optional[str] = None, limit: int = 50):
    await ensure_db()
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
    return rows

# Retry-секция
@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, max=10))
async def send_message_with_retry(client, char_id: str, chat_id: str, content: str):
    return await client.chat.send_message(char_id, chat_id, content)

# Idempotency header dep
async def get_idempotency_key(idempotency_key: Optional[str] = Header(None), external_ref: Optional[str] = Header(None)):
    return idempotency_key or external_ref

# Simple in-memory locks to avoid гонки per character
_locks = {}
def get_lock_for(character_id: str):
    lock = _locks.get(character_id)
    if lock is None:
        lock = asyncio.Lock()
        _locks[character_id] = lock
    return lock

# Pydantic models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False

# Helper: save error logs
def write_error_log(prefix: str, exc: Exception) -> str:
    tb = traceback.format_exc()
    path = os.path.join(ERROR_LOG_DIR, f"{prefix}_{uuid.uuid4().hex}.log")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(tb)
    except Exception:
        pass
    return path

# Маршрут
@APP.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, authorization: Optional[str] = Header(None), idempotency: Optional[str] = Depends(get_idempotency_key)):
    # Быстрая валидация
    if not request.messages or request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Последнее сообщение должно быть от пользователя ('user').")
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
    else:
        raise HTTPException(status_code=401, detail="Неверный Authorization header. Ожидается 'Bearer <token>'")

    # Если мы на Vercel, лучше по умолчанию отключить стриминг (serverless не гарантирует долгие соединения)
    running_on_vercel = os.environ.get("VERCEL") == "1"
    if running_on_vercel and request.stream:
        # принудительно выключаем стрим и продолжим в обычном режиме
        request.stream = False

    # ensure db is ready
    try:
        await ensure_db()
    except Exception as e:
        logp = write_error_log("ensure_db", e)
        return JSONResponse(status_code=500, content={"error": "DB init failed", "log": logp})

    char_id = request.model
    # lazy get client (только здесь)
    try:
        client = await get_client(token=token)
    except Exception as e:
        logp = write_error_log("get_client", e)
        return JSONResponse(status_code=500, content={"error": "get_client failed", "log": logp})

    # ensure mapping / chat creation -- вызываем create_chat только при необходимости
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            cur = await db.execute("SELECT chat_id FROM chat_mapping WHERE character_id = ? LIMIT 1", (char_id,))
            row = await cur.fetchone()
            chat_id = row[0] if row else None
            if not chat_id:
                # создаём чат и сохраняем
                chat, _ = await client.chat.create_chat(char_id)
                chat_id = chat.chat_id
                await db.execute("INSERT OR REPLACE INTO chat_mapping (character_id, chat_id, last_synced_at) VALUES (?, ?, ?)",
                                 (char_id, chat_id, int(datetime.now().timestamp())))
                await db.commit()
    except Exception as e:
        logp = write_error_log("get_or_create_chat", e)
        return JSONResponse(status_code=500, content={"error": "chat creation failed", "log": logp})

    # idempotency/dedup
    external_ref = idempotency
    if external_ref:
        try:
            existing = await find_message_by_external_ref(external_ref)
            if existing:
                # возвращаем последний assistant ответ если есть
                async with aiosqlite.connect(DB_PATH) as db:
                    cur = await db.execute("""
                        SELECT content FROM messages
                        WHERE character_id = ? AND chat_id = ? AND role = 'assistant'
                        ORDER BY created_at DESC LIMIT 1
                    """, (char_id, chat_id))
                    row = await cur.fetchone()
                    last = row[0] if row else ""
                return {"id": f"chatcmpl-{uuid.uuid4()}", "object": "chat.completion", "created": int(datetime.now().timestamp()),
                        "model": request.model, "choices": [{"index": 0, "message": {"role": "assistant", "content": last}, "finish_reason": "stop"}]}
        except Exception as e:
            # логируем, но не прерываем
            write_error_log("idemp_check", e)

    # сохраняем incoming
    user_msg = request.messages[-1].content
    try:
        incoming_msg_id = await save_message_db(char_id, chat_id, "user", user_msg, "external", synced=0, external_ref=external_ref)
    except Exception as e:
        logp = write_error_log("save_incoming", e)
        return JSONResponse(status_code=500, content={"error": "save incoming failed", "log": logp})

    lock = get_lock_for(char_id)
    async with lock:
        try:
            # реплейим unsynced (limit небольшй для скорости)
            unsynced = await fetch_unsynced_external_messages(char_id, chat_id, exclude_id=incoming_msg_id, limit=100)
            for mid, role, content in unsynced:
                if role != "user":
                    await mark_message_synced(mid)
                    continue
                try:
                    ans = await send_message_with_retry(client, char_id, chat_id, content)
                    resp_text = ""
                    try:
                        resp_text = ans.get_primary_candidate().text if ans.get_primary_candidate() else ""
                    except Exception:
                        resp_text = getattr(ans, "text", "") or ""
                    if resp_text:
                        await save_message_db(char_id, chat_id, "assistant", resp_text, "characterai", synced=1)
                    await mark_message_synced(mid)
                except Exception as e:
                    write_error_log("replay_send", e)
                    break

            # non-stream path (recommended on Vercel)
            if not request.stream:
                # send current message and return final text
                try:
                    ans = await send_message_with_retry(client, char_id, chat_id, user_msg)
                    response_text = ""
                    try:
                        response_text = ans.get_primary_candidate().text if ans.get_primary_candidate() else ""
                    except Exception:
                        response_text = getattr(ans, "text", "") or ""
                    if response_text:
                        await save_message_db(char_id, chat_id, "assistant", response_text, "characterai", synced=1)
                    await mark_message_synced(incoming_msg_id)
                    return {"id": f"chatcmpl-{uuid.uuid4()}", "object": "chat.completion", "created": int(datetime.now().timestamp()),
                            "model": request.model, "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
                except Exception as e:
                    logp = write_error_log("send_current", e)
                    return JSONResponse(status_code=500, content={"error": "send current failed", "log": logp})

            # stream path (if not on Vercel) — fallback, may still not work on serverless
            else:
                stream_id = f"chatcmpl-{uuid.uuid4()}"
                created_ts = int(datetime.now().timestamp())
                buffer = ""
                async def stream_gen() -> AsyncGenerator[str, None]:
                    nonlocal buffer
                    try:
                        async for chunk in client.chat.send_message_stream(char_id, chat_id, user_msg):
                            if getattr(chunk, "text", None):
                                buffer += chunk.text
                                response_chunk = {"id": stream_id, "object": "chat.completion.chunk", "created": created_ts,
                                                  "model": request.model, "choices":[{"index":0, "delta":{"content": chunk.text}, "finish_reason": None}]}
                                yield f"data: {json.dumps(response_chunk)}\n\n"
                        # finalize
                        if buffer:
                            await save_message_db(char_id, chat_id, "assistant", buffer, "characterai", synced=1)
                        await mark_message_synced(incoming_msg_id)
                        final = {"id": stream_id, "object":"chat.completion.chunk", "created": created_ts, "model": request.model,
                                 "choices":[{"index":0, "delta":{}, "finish_reason":"stop"}]}
                        yield f"data: {json.dumps(final)}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception as e:
                        logp = write_error_log("stream_error", e)
                        yield f"data: {json.dumps({'error':'stream failed', 'log': logp})}\n\n"
                return StreamingResponse(stream_gen(), media_type="text/event-stream")
        except Exception as e:
            logp = write_error_log("main_handler", e)
            return JSONResponse(status_code=500, content={"error": "internal", "log": logp})

# Debug endpoints
@APP.get("/debug/logs")
async def debug_logs():
    files = sorted([f for f in os.listdir(ERROR_LOG_DIR) if f.endswith(".log")], reverse=True)
    latest = files[:10]
    return {"logs": latest, "log_dir": ERROR_LOG_DIR}

@APP.get("/")
async def root():
    return {"message": "ok"}
