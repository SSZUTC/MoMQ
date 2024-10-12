# -*- coding: utf-8 -*-
# @Time     :  15:29
# @File     : base_db.py
# @Author   : Bruce
# @Team     : XGeneration
import mysql.connector

from database.db import DB


class MySQL(DB):
    def __init__(self, db_name="postgres", user="root", password='mysql123456', host="localhost", port='3306'):
        # 定义连接参数

        super().__init__()
        self.conn = mysql.connector.connect(
            database=db_name,
            user=user,
            password=password,
            host=host,
            port=port
        )
        self.db_name = db_name
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        # 创建光标对象
        self.cursor = self.conn.cursor()
        self.cursor.execute("SET SESSION max_execution_time = 30000;")

        #self.cursor.execute("SET SESSION sql_mode=(SELECT REPLACE(@@sql_mode,'ONLY_FULL_GROUP_BY',''));")
    def reset_connection(self):
        self.close()
        self.__init__(self.db_name, self.user, self.password, self.host, self.port)
