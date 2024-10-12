import re
import sqlglot


def extract_sql_from_qwen(qwen_result) -> str:
    sql = qwen_result
    pattern = r"```sql(.*?)```"

    # 使用re.DOTALL标志来使得点号(.)可以匹配包括换行符在内的任意字符
    sql_code_snippets = re.findall(pattern, qwen_result, re.DOTALL)

    if len(sql_code_snippets) > 0:
        sql = sql_code_snippets[-1].strip()

    return sql

def has_sql_keywords(sql_query):
    # 定义匹配 JOIN、ORDER BY 和 GROUP BY 的正则表达式
    # pattern = r'\b(JOIN|ORDER BY|GROUP BY)\b'
    # pattern = r'\b(JOIN|GROUP BY)\b'
    pattern = r'\b(JOIN|GROUP BY|WHERE)\b'

    # 对sql_query进行搜索
    match = re.search(pattern, sql_query, re.IGNORECASE)

    # 如果找到了匹配项，返回 True，否则返回 False
    return match is not None


def extract_table_and_column_names(sql_expression):
    # 解析 SQL 表达式
    parsed = sqlglot.parse_one(sql_expression)

    # 提取表名和列名
    tables = set()
    columns = set()

    for exp in parsed.walk():
        # 对于找到的表引用，添加到集合中
        if exp.token_type == sqlglot.Tokens.TABLE:
            tables.add(exp.text())
        # 对于找到的列引用，添加到集合中
        elif exp.token_type == sqlglot.Tokens.IDENTIFIER and exp.parent.token_type in {sqlglot.Tokens.COLUMN,
                                                                                       sqlglot.Tokens.ALIAS}:
            columns.add(exp.text())

    return list(tables), list(columns)


# sql_expression = "SELECT a.name, a.age FROM users AS a WHERE a.id = 1"
# tables, columns = extract_table_and_column_names(sql_expression)
#
# print("Tables:", tables)
# print("Columns:", columns)