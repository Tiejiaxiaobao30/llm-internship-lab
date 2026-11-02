from src.readers import read_jsonl

def test_read_jsonl_basic():
    # 第一步：调用 read_jsonl 读取 sample.jsonl
    result = read_jsonl("data/sample.jsonl")
    # 第二步：断言读取到了三条有效数据
    assert len(result) == 3
    assert result[1]["line_no"] == 3
    assert result[2]["data"]["text"] == "深度学习"

import pytest
import json

def test_read_jsonl_invalid():
    with pytest.raises(json.JSONDecodeError):
        # 预期下面的代码会抛出 JSONDecodeError 异常
        read_jsonl("data/bad_sample.jsonl") #实际执行的操作

    # 如果确实抛出指定异常，测试通过。
    # 如果没有抛出异常，或者抛出了其他类型的异常，测试失败。