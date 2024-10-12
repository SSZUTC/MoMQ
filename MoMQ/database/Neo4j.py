from neo4j import GraphDatabase
class Neo4jClient:
    def __init__(self, uri='bolt://127.0.0.1:7687', username='neo4j', password='neo4j123'):
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
    
    def fetch(self, query):
        try:
            with self.driver.session() as session:
                result = session.run(query)
                return [record.data() for record in result], None
        except Exception as e:
            return None, str(e)