import sqlite3
import random

DATABASE_PATH = "conversation_history.sqlite3"


def create_database():
    """Creates a database with two tables (users, conversation) if it doesn't already exist"""
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            user_name TEXT NOT NULL
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


def create_user(user_name):
    """Creates a user, taking in their chosen user_name and assigning them an ID"""

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute('''
        INSERT INTO users (user_name)
        VALUES (?)
    ''', (user_name,))

    user_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return user_id


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