import re
import random
# random.seed(1314)
from typing import List
mac_sql_template = """【DB_ID】 {db_id}
【Schema】
{tables_info}
【Foreign keys】
{fks_info}"""

def scm_dict2text(db_id: str, db_dict: dict) -> str:
    """
    标准的dict转为标准的scm text
    args: db_id
    returns: db schema : mac sql  text
    """
    enums_pre_name = "Value examples:"  # enums_pre_name = "示例值: "
    table_info_list, fk_list = [], []
    for tab_name, tab_content in db_dict.items():
        if not tab_content:
            continue
        if tab_name == "foreign_key_map":
            for fks in tab_content:
                fk1, fk2 = fks
                fk1 = fk1.split(".")[0] + ".`" + fk1.split(".")[1] + "`"
                fk2 = fk2.split(".")[0] + ".`" + fk2.split(".")[1] + "`"
                fk_list.append(fk1 + " = " + fk2)
            # fk_list.append(tab_content)
            continue
        tab_columns = tab_content["header"]
        tab_col_type = tab_content.get("type", [])
        tab_col_enum = tab_content.get("enumerations", {})
        tab_col_desc = tab_content.get("column_description", [])

        tab_cols_list = []
        for idc, col in enumerate(tab_columns):
            col_strs = "  ("
            # +列名
            col_strs += col
            # +列类型
            if len(tab_col_type) > 0:
                col_strs += ":" + tab_col_type[idc] + ","
            else:
                col_strs += ","
            # +列描述
            if len(tab_col_desc) > 0:
                enum_desc = ""
                if col in tab_col_enum:
                    values = tab_col_enum[col]['values']
                    find = True
                    for value in values:
                        if tab_col_desc[idc].find(value) < 0:
                            find = False
                            break
                    if not find:
                        enum_desc = ", " + tab_col_enum[col]["desc"]

                    
                col_strs += " " + tab_col_desc[idc] + enum_desc + "."
                # col_strs += " " + tab_col_desc[idc] + "."
                
            # +枚举值
            col_enums = tab_col_enum.get(col, "")
            if isinstance(col_enums, list) and len(col_enums) > 0:
                col_enums = random.sample(col_enums, min(len(col_enums), random.randint(2, 8)))
                col_strs += " " + enums_pre_name + " " + str(col_enums) + "."
            elif isinstance(col_enums, dict) and len(col_enums['values']) > 0:
                col_enums = random.sample(col_enums['values'], min(len(col_enums['values']), random.randint(2, 8)))
                col_strs += " " + enums_pre_name + " " + str(col_enums) + "."

            col_strs += ")"
            tab_cols_list.append(col_strs)
        table_cols = ",\n".join(tab_cols_list)

        # +tab desc
        table_description = tab_content.get("table_description", "")
        if len(table_description) > 0:
            table_strs = f"# Table: {tab_name}, {table_description}\n[\n{table_cols}\n]"
        else:
            table_strs = f"# Table: {tab_name}\n[\n{table_cols}\n]"
        table_info_list.append(table_strs)
    return mac_sql_template.format(db_id=db_id, tables_info="\n".join(table_info_list), fks_info="\n".join(fk_list))

def scm_text2dict(schema_text:str):
    """
    标准的scm text转 dict
    args: schema_text: mac sql text
    returns:
    db_schema: {"tab_name":{"col_name":"col_line" }, ...}
    fk_list
    """
    schema_info = {}
    # tables_pattern = re.compile(r'# Table: (\w+)\n\[([\s\S]+?)\]\n', re.MULTILINE)
    tables_pattern = re.compile(r'# Table: (.+?)\n\[([\s\S]+?)\]\n', re.MULTILINE)
    matches = tables_pattern.findall(schema_text)
    # print(matches)
    # 提取表结构
    for match in matches:
        table_name, table_contents = match
        schema_info[table_name] = {}
        # 过滤Value examples行
        row_items = table_contents.strip().split(",\n")
        for item in row_items:
            left_s = item.find('(') if item.find('(') > -1 else 0
            right_s = item.rfind(')') if item.rfind(')') > -1 else len(item)
            item = item[left_s + 1:right_s]
            col_name_str = item.strip().split(",")[0]
            col_name = col_name_str.split(":")[0].strip()
            schema_info[table_name][col_name] = "(" + item + ")"

    # 提取外键候选主键
    fk_list = []
    if "【Foreign keys】" not in schema_text:
        return schema_info, fk_list
    fk_list_str = schema_text.strip().split("【Foreign keys】")[-1].strip().split("\n")
    for fks in fk_list_str:
        if len(fks) <= 0:
            continue
        fk1, fk2 = fks.strip().split("=")
        tab1, col1 = fk1.strip().split(".")
        tab2, col2 = fk2.strip().split(".")
        if "`" in col1 and "`" in col2:
            fk_list.append([tab1 + "." + col1[1:-1], tab2 + "." + col2[1:-1]])
        else:
            fk_list.append([tab1 + "." + col1, tab2 + "." + col2])
    # schema_info["foreign_key_map"] = fk_list

    return schema_info, fk_list

def scm_augdict2text(db_id: str, db_dict: dict, fk_item_list: List) -> str:
    """
    aug dict转为标准的scm text
    args: db_id, db_dict:{"tab_name":{"col_name":"col_line" }, ...}, fk:[[], []]
    returns: db schema : mac sql text
    """
    table_info_list = []
    for tab_name, tab_col_content in db_dict.items():
        if len(tab_col_content) <= 0:
            continue
        tab_cols_list = []
        for col, col_line in tab_col_content.items():
            tab_cols_list.append("  "+col_line)
        table_cols = ",\n".join(tab_cols_list)
        table_strs = f"# Table: {tab_name}\n[\n{table_cols}\n]"
        table_info_list.append(table_strs)

    fk_list = []
    for fks in fk_item_list:
        if len(fks) <= 0:
            continue
        fk1, fk2 = fks
        # fk1 = fk1.split(".")[0] + ".`" + fk1.split(".")[1] + "`"
        # fk2 = fk2.split(".")[0] + ".`" + fk2.split(".")[1] + "`"
        fk_list.append(fk1 + " = " + fk2)
    return mac_sql_template.format(db_id=db_id, tables_info="\n".join(table_info_list), fks_info="\n".join(fk_list))


def scm_fk_filter(db_dict: dict, fk_list: List):
    """
    过滤掉不属于db_dict的fk
    args: db_dict:{"tab_name":{"col_name":"col_line" }, ...},  fk:[[], []]
    returns: new fk_list
    """
    new_fk_list = []
    for fks in fk_list:
        if len(fks) <= 0:
            continue
        fk1, fk2 = fks
        fk1_temp = fk1.strip().split('.')
        t1, c1 = fk1_temp[0], fk1_temp[1]
        fk2_temp = fk2.strip().split('.')
        t2, c2 = fk2_temp[0], fk2_temp[1]
        if t1 in db_dict and t2 in db_dict:
            if c1 in db_dict[t1] and c2 in db_dict[t2]:
                new_fk_list.append([fk1, fk2])
    return new_fk_list



def chase_dict2mac_sql(db_id: str, db_content: dict):

    enums_name = "Value examples: "
    # enums_name = "示例值: "
    table_info_list, table_name_list, fk_list = [], [], []
    for tab_name, tab_dict in db_content.items():
        if tab_name == "foreign_key_map":
            for fks in db_content[tab_name]:
                fk1, fk2 = fks
                fk1 = fk1.split(".")[0] + ".`" + fk1.split(".")[1] + "`"
                fk2 = fk2.split(".")[0] + ".`" + fk2.split(".")[1] + "`"
                fk_list.append(fk1 + " = " + fk2)
            continue
        tab_cols = tab_dict["header"]
        tab_col_enums = tab_dict.get("enumerations", {})
        cols_list = []
        for col in tab_cols:
            col_strs = "  ("
            col_enums = tab_col_enums.get(col, "")
            # 无枚举值
            if len(col_enums) <= 0:
                col_strs += col
            else:
                col_enums = random.sample(col_enums, min(len(col_enums), random.randint(1, 8)))
                col_strs += col + ", "
                col_strs += enums_name
                col_strs += str(col_enums)
                col_strs += "."
            col_strs += ")"

            cols_list.append(col_strs)

        table_cols = ",\n".join(cols_list)
        table_strs = f"# Table: {tab_name}\n[\n{table_cols}\n]"
        table_name_list.append(tab_name)
        table_info_list.append(table_strs)

    return mac_sql_template.format(db_id=db_id, tables_info="\n".join(table_info_list), fks_info="\n".join(fk_list))

def chase_dict2mac_sql_desc(db_id: str, db_content: dict):

    enums_name = "Value examples: "
    # enums_name = "示例值: "
    table_info_list, table_name_list, fk_list = [], [], []
    for tab_name, tab_dict in db_content.items():
        if tab_name == "foreign_key_map":
            for fks in db_content[tab_name]:
                fk1, fk2 = fks
                fk1 = fk1.split(".")[0] + ".`" + fk1.split(".")[1] + "`"
                fk2 = fk2.split(".")[0] + ".`" + fk2.split(".")[1] + "`"
                fk_list.append(fk1 + " = " + fk2)
            continue
        tab_cols = tab_dict["header"]
        tab_col_enums = tab_dict.get("enumerations", {})
        cols_list = []
        for col in tab_cols:
            col_strs = "  ("
            col_enums = tab_col_enums.get(col, "")
            # 无枚举值
            if len(col_enums) <= 0:
                col_strs += col + ", "
                col_strs += col + "."
            else:
                col_enums = random.sample(col_enums, min(len(col_enums), random.randint(1, 8)))
                col_strs += col + ", "
                col_strs += col + ". "
                col_strs += enums_name
                col_strs += str(col_enums)
                col_strs += "."
            col_strs += ")"

            cols_list.append(col_strs)

        table_cols = ",\n".join(cols_list)
        table_strs = f"# Table: {tab_name}\n[\n{table_cols}\n]"
        table_name_list.append(tab_name)
        table_info_list.append(table_strs)

    return mac_sql_template.format(db_id=db_id, tables_info="\n".join(table_info_list), fks_info="\n".join(fk_list))

def update_fks(schema_text):

    if "【Foreign keys】" not in schema_text:
        return schema_text

    pre_scm, fks_text = schema_text.strip().split("【Foreign keys】")
    if len(fks_text) <= 0:
        return schema_text

    fks_list = fks_text.strip().split("\n")
    fk_temp_list = []
    for fk in fks_list:
        if len(fk) <= 0:
            continue
        f1, f2 = fk.split('=')
        f1 = f1.strip().split('.')
        if "`" in f1[1]:
            t1, c1 = f1[0], f1[1][1:-1]
        else:
            t1, c1 = f1[0], f1[1]
        f2 = f2.strip().split('.')
        if "`" in f2[1]:
            t2, c2 = f2[0], f2[1][1:-1]
        else:
            t2, c2 = f2[0], f2[1]
        fk_temp_list.append(t1+"."+c1 + " = " + t2+"."+c2)

    return pre_scm + "【Foreign keys】\n" + "\n".join(fk_temp_list)