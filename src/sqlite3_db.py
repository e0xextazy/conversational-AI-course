import os
import sqlite3
from pathlib import Path


class MessageLogger():
    def __init__(self, folder_path: str) -> None:
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(os.path.join(folder_path, "chat.db")):
            self.con = sqlite3.connect(os.path.join(folder_path, "chat.db"))
            self.cur = self.con.cursor()
            self.cur.execute(
                "CREATE TABLE messages(user_id INT, message VARCHAR(255), response VARCHAR(255), tox_score FLOAT, datetime DATETIME DEFAULT CURRENT_TIMESTAMP)")
        else:
            self.con = sqlite3.connect(os.path.join(folder_path, "chat.db"))
            self.cur = self.con.cursor()

    def close(self) -> None:
        self.con.close()

    def write_message(self, message) -> None:
        self.cur.execute(
            f"INSERT INTO messages(user_id, message, response, tox_score) VALUES ({message['user_id']}, '{message['message']}', '{message['response']}', {message['tox_score']})")
        self.con.commit()

    def get_last_messages(self, user_id: int, n: int = 5):
        res = self.cur.execute(
            f"SELECT message, response FROM messages where user_id={user_id} Order By datetime DESC LIMIT {n}")

        return res.fetchall()
