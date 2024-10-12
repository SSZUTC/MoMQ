import json


def read_dict_list(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            data.append(json.loads(line))
    return data

def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def read_text(filename)->str:
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            line = line.strip()
            data.append(line)
    return data

def read_raw_text(filename)->str:

    # 使用with语句来确保文件正确关闭
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()  # 读取整个文件内容
    except:
        content = ''
    return content

def read_list(file):
    ls = []
    with open(file,'r', encoding='utf-8') as f:
        ls = f.readlines()

    ls  = [l.strip().replace('\n','') for l in ls]
    return ls

def save_list(file,ls):
    with open(file,'w', encoding='utf-8') as f:
        f.write('\n'.join(ls))


def read_map_file(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            data[line[0]] = line[1].split('、')
            data[line[0]].append(line[0])
    return data

def save_json(target_file,js):
    with open(target_file, 'w', encoding='utf-8') as f:
        json.dump(js, f, ensure_ascii=False, indent=4)
