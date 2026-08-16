import aiosqlite
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

DB_PATH = Path("data/jarvis_conversations.db")

async def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        await db.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                role TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
        ''')
        await db.commit()

async def create_conversation(title: str = "New Conversation") -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute('INSERT INTO conversations (title) VALUES (?)', (title,))
        await db.commit()
        return cursor.lastrowid

async def add_message(conversation_id: int, role: str, content: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            'INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)',
            (conversation_id, role, content)
        )
        await db.execute(
            'UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?',
            (conversation_id,)
        )
        await db.commit()

async def get_messages(conversation_id: int) -> List[Dict[str, str]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            'SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at ASC',
            (conversation_id,)
        )
        rows = await cursor.fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

async def get_all_conversations() -> List[Dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute('SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC')
        rows = await cursor.fetchall()
        return [{"id": row["id"], "title": row["title"], "updated_at": row["updated_at"]} for row in rows]

async def delete_conversation(conversation_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
        await db.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
        await db.commit()
