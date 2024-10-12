# -*- coding: utf-8 -*-
# @Time     :  15:13
# @File     : PG.py
# @Author   : Bruce
# @Team     : XGeneration

import psycopg2
from database.db import DB


class PG(DB):
    def __init__(self, db_name='postgres', user='postgres', password="postgres", host="localhost", port='5432'):

        # 定义连接参数
        conn = psycopg2.connect(
            dbname=db_name,
            user=user,
            password=password,
            host=host,
            port=port
        )
        self.db_name = db_name
        self.user = user
        self.password = password
        self.host = host
        self.conn = conn
        self.port = port
        # 创建光标对象
        cursor = conn.cursor()
        self.cursor= cursor
    def reset_connection(self):
        self.close()
        self.__init__(self.db_name, self.user, self.password, self.host, self.port)
