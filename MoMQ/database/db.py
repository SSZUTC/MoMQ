# -*- coding: utf-8 -*-
# @Time     :  15:29
# @File     : db.py
# @Author   : Bruce
# @Team     : XGeneration

# global names
DB_SQLite = 'SQLite'
DB_PostgreSQL = 'PostgreSQL'
DB_MySQL = 'MySQL'

import re


def add_limit_to_sql(sql_query, limit=100):
    """
    给SQL语句添加LIMIT限制，如果已存在LIMIT则保持不变，
    并确保SQL语句以分号结尾。

    :param sql_query: 输入的SQL查询语句
    :param limit: 默认的LIMIT值，如果SQL未指定LIMIT则使用此值
    :return: 增改后的SQL语句
    """
    # 去掉查询结尾可能的空白符和分号
    cleaned_query = sql_query.rstrip(" ;\t\n\r")

    # 使用正则表达式检查是否已包含LIMIT子句
    limit_pattern = r"(?i)\bLIMIT\b\s*\d+"
    if re.search(limit_pattern, cleaned_query):
        # 如果已经包含LIMIT子句，只需要确保SQL语句以分号结尾
        modified_sql_query = cleaned_query + ";"
    else:
        # 如果没有包含LIMIT子句，则添加LIMIT并确保以分号结尾
        modified_sql_query = cleaned_query + f" LIMIT {limit};"

    return modified_sql_query



class DB:
    def __init__(self):
        self.conn=None
        self.cursor = None
        self.max_limit = 10#
    def reset_connection(self):
        pass

    def fetch(self,sql):
        if not sql.strip().endswith(';'):
            sql += ';'
        try:
            # 执行SQL语句
            self.cursor.execute(sql)
            #print(sql)
            # 获取结果
            records = self.cursor.fetchall()
            err = None
        except Exception as e:
            err = str(e)
            print(err)
            records=None
            self.reset_connection()
        return records, err
    def fetch_with_exception_limit(self,sql):
        sql=add_limit_to_sql(sql,self.max_limit)
        #print(sql)
        try:
            # 执行SQL语句
            self.cursor.execute(sql)
            #print(sql)
            # 获取结果
            # 使用fetchone逐条读取，直到达到max_results
            # records=[]
            # for _ in range(self.max_limit):
            #
            #     row = self.cursor.fetchone()
            #
            #     if row is None:
            #         break
            #
            #     records.append(row)
            records = self.cursor.fetchall()
            error_info = ''
        except Exception as e:
            print("SQL error: ", e," SQL: %s"%sql)
            records=None
            self.reset_connection()
            error_info = e
        return records, error_info


    def fetch_with_exception(self,sql):
        if not sql.strip().endswith(';'):
            sql += ';'
        try:
            # 执行SQL语句
            self.cursor.execute(sql)
            #print(sql)
            # 获取结果
            records = self.cursor.fetchall()
            error_info = ''
        except Exception as e:
            #print("SQL error: ", e)
            records=None
            self.reset_connection()
            error_info = e
        return records, error_info
    def quick_update(self,sql):
        self.cursor.execute(sql)
    def quick_update_raw(self,sql,args):
        self.cursor.execute(sql,args)
    def commit(self):
        self.conn.commit()
    def update(self,sql):
        try:
            self.cursor.execute(sql)
        except Exception as e:
            print('error on %s'%sql)
            print(e)
        # 提交事务
        self.conn.commit()
    def create_table(self,sqls):
        sql_group = sqls.split(';\n')
        for sql in sql_group:
            self.update(sql)

    def close(self):
        # 关闭光标和连接
        self.cursor.close()
        self.conn.close()

    #def __del__(self):
        #self.close()