from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool


class NebulaGraphClient:
    def __init__(self, host='127.0.0.1', port=9669, username='root', password='nebula'):
        self.host = host
        self.port = port
        self.username = username
        self.password = password

        config = Config()
        config.timeout = 5000
        self.connection_pool = ConnectionPool()
        ok = self.connection_pool.init([(self.host, self.port)], config)

        if not ok:
            raise Exception("Failed to initialize the connection pool")
        
        # 获取Session
        self.session = self.connection_pool.get_session(self.username, self.password)
        if not self.session:
            raise Exception("Failed to create a session")
    
    
    def fetch(self, query_string, space):
        if not self.session:
            raise Exception("Session is not created. Call connect() first.")

        if space:
            query_string = f'USE {space}; {query_string}'

        # 执行查询
        try:
            result_set = self.session.execute(query_string)
        except:
            return None, 'timeout'
        if not result_set.is_succeeded():
            return None, 'err'
        
        # 处理查询结果
        # result_list = result_set.as_data_frame().values.tolist()
        # result_tuple = [tuple(row) for row in result_list]
        return result_set.as_data_frame().to_dict(orient='records'), None
    
    def close(self):
        if self.session:
            self.session.release()
        if self.connection_pool:
            self.connection_pool.close()
