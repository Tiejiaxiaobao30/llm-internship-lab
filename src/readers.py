import json

def read_jsonl(file_path):
    record=[]
    required_fields = ["id", "text", "label"]
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_no ,data in enumerate(file, start=1):
            # Python 开始从文件中读取内容，并尝试按照 UTF-8 解码。如果文件中包含无效的 UTF-8 字节序列，Python 会抛出 UnicodeDecodeError 异常。
            if not data.strip():  # 检查行是否为空
                continue
            try: #尝试执行可能出错的代码
                parsed = json.loads(data)
            except json.JSONDecodeError as e: 
                #如果出现指定的 JSON 解析异常，就捕获它，并命名为 e
                print(f"JSON解析失败：文件第{line_no}行，错误：{e}")
                raise 
                #raise 重新抛出当前异常，并保留原始异常信息
            for field in required_fields:
                if field not in parsed:
                    raise ValueError(f"第{line_no}行缺少 '{field}' 字段")
            if not isinstance(parsed["text"], str):
                #isinstance(对象, 类型) 用来判断对象是否属于指定类型
                raise ValueError(f"第{line_no}行的 'text' 字段不是字符串类型")
            if parsed["text"].strip() == "":
                raise ValueError(f"第{line_no}行的 'text' 字段为空或仅包含空白字符")
            record.append({"line_no": line_no, "data": parsed})
            #json.loads(data) 解析 JSONL 后，得到的是字典
            #json.loads()解析jsonl的单独一行；json.load()解析整个json文件
    return record



