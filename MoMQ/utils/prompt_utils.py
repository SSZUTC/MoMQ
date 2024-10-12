nl2sql_template = """你是一名数据库语言专家，现在需要阅读并理解下面的【数据库schema】描述，可能用到的【参考信息】，并运用数据库知识生成sql语句回答【用户问题】。
【用户问题】
{question}

【数据库schema】
{db_schema}

【参考信息】
{evidence}

【用户问题】
{question}

```sql"""

nl2mysql_template = """你是一名MySQL专家，现在需要阅读并理解下面的【数据库schema】描述，以及可能用到的【参考信息】，并运用MySQL知识生成sql语句回答【用户问题】。
【用户问题】
{question}

【数据库schema】
{db_schema}

【参考信息】
{evidence}

【用户问题】
{question}

```sql"""

nl2sqlite_template = """你是一名SQLite专家，现在需要阅读并理解下面的【数据库schema】描述，以及可能用到的【参考信息】，并运用SQLite知识生成sql语句回答【用户问题】。
【用户问题】
{question}

【数据库schema】
{db_schema}

【参考信息】
{evidence}

【用户问题】
{question}

```sql"""

nl2pgsql_template = """你是一名PostgreSQL专家，现在需要阅读并理解下面的【数据库schema】描述，以及可能用到的【参考信息】，并运用PostgreSQL知识生成sql语句回答【用户问题】。
【用户问题】
{question}

【数据库schema】
{db_schema}

【参考信息】
{evidence}

【用户问题】
{question}

```sql"""

nl2cypher_template = """你是一名Neo4j专家，现在需要阅读并理解下面的【图数据库schema】描述，以及可能用到的【参考信息】，并运用Cypher知识生成Cypher Query语句回答【用户问题】。
【用户问题】
{question}

【图数据库schema】
{db_schema}

【参考信息】
{evidence}

【用户问题】
{question}

```cypher"""

nl2ngql_template = """你是一名NebulaGraph专家，现在需要阅读并理解下面的【图数据库schema】描述，以及可能用到的【参考信息】，并运用nGQL知识生成Graph Query语句回答【用户问题】。
【用户问题】
{question}

【图数据库schema】
{db_schema}

【参考信息】
{evidence}

【用户问题】
{question}

```ngql"""

# nl2pgsql_template = """你是一名PostgreSQL专家，现在需要阅读并理解下面的【数据库schema】描述，以及可能用到的【参考信息】，并运用PostgreSQL知识生成【SQL】语句回答【用户问题】。
# 【用户问题】
# {question}

# 【数据库schema】
# {db_schema}

# 【参考信息】
# {evidence}

# 【用户问题】
# {question}

# 【SQL】"""

sqlcoder_template = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Generate a SQL query to answer this question: `{question}`
{instructions}

DDL statements:
{create_table_statements}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The following SQL query best answers the question `{question}`:
```sql

"""

pgsql2nl_template = """你是一名PostgreSQL专家，现在有一条【SQL】语句，是基于下面的【数据库schema】描述，以及可能用到的【参考信息】来生成回复【用户问题】的，现在请你根据【SQL】语句，推断出{qnum}条可能询问的【用户问题】。
【SQL】
{sql}

【数据库schema】
{db_schema}

【参考信息】
{evidence}

【SQL】
{sql}

【用户问题】"""

def gen_train_prompt(idx: int, data_item: dict, sql_type: str) -> dict:
    """

    """
    db_schema = data_item["db_schema"]
    question = data_item["rewrite"] if "rewrite" in data_item and data_item["rewrite"] else data_item["question"]
    evidence = data_item["evidence"]
    sql_type = sql_type.lower()

    if sql_type == "mysql":
        prompt = nl2mysql_template.format(db_schema=db_schema.strip(), question=question, evidence=evidence)
    elif sql_type == "postgresql":
        prompt = nl2pgsql_template.format(db_schema=db_schema.strip(), question=question, evidence=evidence)
    elif sql_type == "common":
        prompt = nl2sql_template.format(db_schema=db_schema.strip(), question=question, evidence=evidence)
    elif sql_type == "pg2nl":
        prompt = pgsql2nl_template.format(qnum=data_item["qnum"], sql=data_item["sql"], db_schema=db_schema.strip(), evidence=evidence)
    elif sql_type == "cypher":
        prompt = nl2cypher_template.format(db_schema=db_schema.strip(), question=question, evidence=evidence)
    elif sql_type == "ngql":
        prompt = nl2ngql_template.format(db_schema=db_schema.strip(), question=question, evidence=evidence)
    elif sql_type == "sqlite":
        prompt = nl2sqlite_template.format(db_schema=db_schema.strip(), question=question, evidence=evidence)
    else:
        prompt = sqlcoder_template.format(db_schema=db_schema.strip(), question=question)

    conversation = [
        {
            "role": "user",
            "content": prompt
        },
        {
            "role": "assistant",
            "content": data_item["sql"] if sql_type != "pg2nl" else data_item["question"]
        }
    ]
    train_item = {
        "id": idx,
        "conversations": conversation,
        "sql_type": sql_type
    }

    return train_item