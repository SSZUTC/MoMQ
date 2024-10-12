import sqlite3

from database.db import DB


class SQLite(DB):
    def __init__(self, db_path):
        # 定义连接参数

        super().__init__()
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        # 创建光标对象
        self.cursor = self.conn.cursor()
    def reset_connection(self):
        self.close()
        self.__init__(self.db_path)
