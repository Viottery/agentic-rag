from pathlib import Path


PROMPT_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    """
    加载节点级 prompt 模板。

    约定：
    - name 为 prompts 目录下的文件名
    - 使用 UTF-8 编码
    """
    path = PROMPT_DIR / name
    return path.read_text(encoding="utf-8")