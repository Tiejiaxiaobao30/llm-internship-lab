import json

def read_jsonl(file_path):
    record=[]
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_no ,data in enumerate(file, start=1):
            if not data.strip():  # 检查行是否为空
                continue
            try: #尝试执行可能出错的代码
                parsed = json.loads(data)
            except json.JSONDecodeError as e: 
                #如果出现指定的 JSON 解析异常，就捕获它，并命名为 e
                print(f"JSON解析失败：文件第{line_no}行，错误：{e}")
                raise 
                #raise 不是只能用来终止程序，它本质上是抛出异常，让上层代码有机会处理
            record.append({"line_no": line_no, "data": parsed})
            #json.loads(data) 解析 JSONL 后，得到的是字典
            #json.loads()解析jsonl的单独一行；json.load()解析整个json文件
    return record




