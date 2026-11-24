import os
import sqlite3
import random
import libsql
from pathlib import Path
from datetime import datetime, timezone
from werkzeug.security import check_password_hash

DATABASE_PATH = "conversation_history.sqlite3"

TURSO_URL = os.environ.get("TURSO_DATABASE_URL")
TURSO_TOKEN = os.environ.get("TURSO_AUTH_TOKEN")

def get_connection():
    if TURSO_URL and TURSO_TOKEN:
        return libsql.connect(database=TURSO_URL, auth_token=TURSO_TOKEN)
    return sqlite3.connect(DATABASE_PATH)


def create_database(database_name=DATABASE_PATH):
    """Creates a database with two tables (users, conversation) if it doesn't already exist"""
    print("connecting to database...")
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            user_name TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id INTEGER PRIMARY KEY,
            start_time TEXT,
            end_time TEXT,
            user_id INTEGER,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            message_id INTEGER PRIMARY KEY,
            role TEXT,
            content TEXT,
            conversation_id INTEGER,
            FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
        )
    ''')

    connection.commit()
    connection.close()


def create_user(user_name, password):
    """Creates a user, taking in their chosen user_name and assigning them an ID"""
    
    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute('''
            INSERT INTO users (user_name, password)
            VALUES (?, ?)
        ''', (user_name, password))

        user_id = cursor.lastrowid

        connection.commit()
        return user_id

    finally:
        connection.close()


def retrieve_user(user_name, password):
    """Based on a user_name and password, a user is retrieved from the database"""

    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute('''
            SELECT * FROM users WHERE user_name = ?
        ''', (user_name,))

        user = cursor.fetchone()

        if user:
            user = dict(zip([col[0] for col in cursor.description], user))
            hashed_password = user['password']
            if check_password_hash(hashed_password, password):
                return user
        return None

    finally:
        connection.close()


def create_conversation(user_id):
    """Creates a new conversation row and returns its conversation_id."""
    now = datetime.now(timezone.utc).isoformat()
    connection = get_connection()
    try:
        cursor = connection.cursor()
        cursor.execute('''
            INSERT INTO conversations (start_time, end_time, user_id)
            VALUES (?, ?, ?)
        ''', (now, now, user_id))
        conversation_id = cursor.lastrowid
        connection.commit()
        return conversation_id
    finally:
        connection.close()


def add_message(conversation_id, role, content):
    """Saves a single message and updates the conversation's end_time."""
    now = datetime.now(timezone.utc).isoformat()
    connection = get_connection()
    try:
        cursor = connection.cursor()
        cursor.execute('''
            INSERT INTO messages (conversation_id, role, content)
            VALUES (?, ?, ?)
        ''', (conversation_id, role, content))
        
        cursor.execute('''
            UPDATE conversations
            SET end_time = ?
            WHERE conversation_id = ?
        ''', (now, conversation_id))
        
        connection.commit()
    finally:
        connection.close()


def get_user_conversations(user_id):
    """Gets all conversations for a user (newest first), with the first message as a preview title."""
    connection = get_connection()
    try:
        cursor = connection.cursor()
        cursor.execute('''
            SELECT 
                c.conversation_id,
                c.start_time,
                c.end_time,
                (
                    SELECT m.content 
                    FROM messages m 
                    WHERE m.conversation_id = c.conversation_id AND m.role = 'user'
                    ORDER BY m.message_id DESC 
                    LIMIT 1
                ) AS preview
            FROM conversations c
            WHERE c.user_id = ?
            ORDER BY c.end_time DESC
        ''', (user_id,))
        rows = cursor.fetchall()
        cols = [col[0] for col in cursor.description]
        return [dict(zip(cols, row)) for row in rows]
    finally:
        connection.close()


def get_conversation_messages(conversation_id, user_id):
    """Gets all messages for a conversation, verifying it belongs to the user."""
    connection = get_connection()
    try:
        cursor = connection.cursor()
        cursor.execute('''
            SELECT m.role, m.content
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.conversation_id
            WHERE m.conversation_id = ? AND c.user_id = ?
            ORDER BY m.message_id ASC
        ''', (conversation_id, user_id))
        rows = cursor.fetchall()
        cols = [col[0] for col in cursor.description]
        return [dict(zip(cols, row)) for row in rows]
    finally:
        connection.close()


# def save_conversation(conversation_information, conversation_history):
#     """Saves a user's conversation"""

#     connection = get_connection()
#     cursor = connection.cursor()

#     cursor.execute('''
#         INSERT INTO conversations (start_time, end_time, user_id)
#         VALUES (:start_time, :end_time, :user_id)
#     ''', conversation_information)

#     conversation_id = cursor.lastrowid

#     message_data = [(conversation_id, message["role"], message["content"]) for message in conversation_history]

#     cursor.executemany('''
#         INSERT INTO messages (conversation_id, role, content)
#         VALUES (?, ?, ?)
#     ''', message_data)

#     connection.commit()
#     connection.close()


# add when/if needed
# def read_database():
#     pass