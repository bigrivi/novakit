# 安装

`novakit` 需要 **Python 3.12 或更高版本**（`pyproject.toml` 中的
`requires-python = ">=3.12"`，已在 3.12 和 3.13 上测试）。

## 从 PyPI 安装

```bash
pip install novakit
```

## 使用 uv 安装

```bash
uv add novakit
```

## 可选扩展

在 `pyproject.toml` 的 `[project.optional-dependencies]` 下定义：

| 扩展       | 安装内容                | 启用的功能 |
| -------- | -------------------- | --------------- |
| `images` | `pillow`             | `novakit.llm` 中的图片处理辅助函数 |
| `mcp`    | （无额外运行时依赖） | `novakit.mcp` 客户端、stdio/HTTP 传输、工具名分配 |

```bash
pip install "novakit[images]"
pip install "novakit[mcp]"
pip install "novakit[images,mcp]"
```

当前 `pyproject.toml` 中没有 `docs` 扩展；文档工具链（MkDocs、`mkdocstrings`）
属于文档基础设施，由基础设施通道处理，不是运行时依赖。

## 本地开发

```bash
git clone git@github.com:bigrivi/novakit.git
cd novakit
uv sync --extra dev
```

`dev` 扩展提供 `mypy`、`pytest`、`pytest-asyncio`、`pytest-cov` 和
`ruff`。完整预检流程为：

```bash
uv run pytest -q --cov=novakit --cov-report=term-missing
uv run ruff check novakit tests
uv run ruff format novakit tests
uv run mypy
```

测试套件要求 `novakit` 行覆盖率至少 99%，且离线运行；
提供商传输、文件系统访问和外部服务均由测试替身代替。

## 下一步

继续阅读可运行的[快速上手](quickstart.md)。
