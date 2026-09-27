# -*- coding: utf-8 -*-
"""
仅命令行分群：读取学生 JSON，调用与主程序相同的 ``classify_student``。

用法（项目根目录）:
  python classify_student.py
  python classify_student.py path/to/student.json

与 ``run_agent.py`` 使用相同 ``models/`` 与环境变量（MODELS_DIR 等）。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# 同目录下导入 run_agent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_agent import STUDENT_JSON, classify_student  # noqa: E402
from utils import load_json  # noqa: E402


def main() -> int:
    p = (
        Path(os.environ.get("STUDENT_JSON", str(STUDENT_JSON))).resolve()
        if len(sys.argv) < 2
        else Path(sys.argv[1]).resolve()
    )
    if not p.is_file():
        print(f"未找到学生文件: {p}", file=sys.stderr)
        return 1
    try:
        data = load_json(p)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"读取失败: {e}", file=sys.stderr)
        return 1
    if not isinstance(data, dict):
        print("学生文件根节点须为 JSON 对象。", file=sys.stderr)
        return 1
    try:
        out = classify_student(data)
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as e:
        print(f"分群失败: {e}", file=sys.stderr)
        return 1
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
