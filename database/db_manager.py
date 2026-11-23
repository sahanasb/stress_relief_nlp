import sqlite3
import random
from pathlib import Path
from werkzeug.security import check_password_hash

DATABASE_PATH = "conversation_history.sqlite3"


def create_database(database_name=DATABASE_PATH):
    """Creates a database with two tables (users, conversation) if it doesn't already exist"""
    print("connecting to database...")
    connection = sqlite3.connect(database_name)
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
        connection = sqlite3.connect(DATABASE_PATH)
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
        connection = sqlite3.connect(DATABASE_PATH)
        connection.row_factory = sqlite3.Row
        cursor = connection.cursor()

        cursor.execute('''
            SELECT * FROM users WHERE user_name = ?
        ''', (user_name,))

        user = cursor.fetchone()

        if user:
            hashed_password = user['password']
            if check_password_hash(hashed_password, password):
                return dict(user)
        return None

    finally:
        connection.close()


def save_conversation(conversation_information, conversation_history):
    """Saves a user's conversation"""

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute('''
        INSERT INTO conversations (start_time, end_time, user_id)
        VALUES (:start_time, :end_time, :user_id)
    ''', conversation_information)

    conversation_id = cursor.lastrowid

    message_data = [(conversation_id, message["role"], message["content"]) for message in conversation_history]

    cursor.executemany('''
        INSERT INTO messages (conversation_id, role, content)
        VALUES (?, ?, ?)
    ''', message_data)

    connection.commit()
    connection.close()


# add when/if needed
# def read_database():
#     pass