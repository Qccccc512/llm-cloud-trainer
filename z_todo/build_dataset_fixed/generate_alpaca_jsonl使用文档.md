# `generate_alpaca_jsonl_v3.py` 使用文档

## 1) 核心功能

- **输入**：`.txt` / `.pdf` 文件，或目录（递归扫描目录内的 `.txt/.pdf`）

- **处理**：读取文本 → 清洗 → 切块 → 调用模型生成单轮问答

- **输出**：Alpaca `jsonl`，每行一条样本，固定为：

  ```json
  {"instruction":"问题","input":"","output":"回答"}
  ```

  - `input` 永远是空字符串 `""`
  - 输出数据中不会出现 `SOURCE_TEXT/TEXT` 之类字段，也不会把原文写入 `input`

- **风格可控**：可通过 `--style` / `--style-file` 指定“你想要的数据集长什么样”（例如法律问答）

------

## 2) 安装与配置

### 安装依赖

```bash
pip install openai pydantic PyPDF2 tqdm
```

### 配置 Key

建议用环境变量：

```bash
export OPENAI_API_KEY="你的key"
```

### 配置第三方 base_url

在脚本顶部修改：

```python
BASE_URL = "https://你的兼容接口域名/v1"
```

------

## 3) 参数说明（完整）

### 必填

- `--input`
  - 输入文件或目录（目录会递归扫描 `.txt/.pdf`）
  - 例：`--input law.txt` / `--input ./docs`
- `--output`
  - 输出 jsonl 路径
  - 例：`--output out.jsonl`

### 风格控制

- `--style`

  - 直接在命令行提供“数据集风格说明”（会作为 prompt 的一部分）

  - 适合短说明

  - 例：

    ```bash
    --style "生成法律问答：问题围绕要件/程序/例外/责任；回答严谨；不得引入文本外信息；不足说明文本未提供；不出现‘根据上文/本文’。"
    ```

- `--style-file`

  - 从文件读取风格说明（推荐长说明）
  - 例：`--style-file style_legal.txt`

- `--profile`

  - 内置预设风格（可选）
  - 可选值：
    - `concept_qa`：概念解释/定义/区别
    - `process_qa`：流程/步骤/清单
    - `pitfall_qa`：误区/纠偏建议
    - `application_qa`：场景应用/方案
    - `mixed`：混合（默认）

### 切块参数

- `--chunk-chars`
  - 每个 chunk 的最大字符数（默认 2500）
  - 影响：越大 chunk 越少；越小 chunk 越多
- `--overlap`
  - 相邻 chunk 重叠字符数（默认 200）
  - 约束：必须 `< chunk-chars`

### 生成规模参数

- `--pairs-per-chunk`
  - 每个 chunk 生成多少条问答（默认 3）
  - 越大输出越长，可能更容易被截断/解析失败
- `--max-examples`
  - 最多写入多少条样本（默认 200）
  - 达到后停止
- `--max-output-tokens`
  - 单次请求最大输出 tokens（默认 1200）
  - 如果输出经常被截断（JSON 不完整），可以调大

### 语言

- `--lang`
  - 输出语言提示（默认 `zh-CN`）
  - 例：`--lang en`

------

## 4) 使用示例

### 示例 A：法律问答（推荐 style-file）

1. 写一个 `style_legal.txt`，例如：

```text
请生成【法律问答】数据集：
instruction 必须是可独立理解的法律问题（要件/程序/例外/责任等），不要出现“根据上文/本文/材料”等措辞。
output 用严谨但通俗的语言回答，可分点说明；只依据提供文本，不引入文本外法条/案例；不足则说明“文本未提供/无法确定”。
只输出 JSON。
```

1. 运行：

```bash
python generate_alpaca_jsonl.py \
  --input law.txt \
  --output law_qa.jsonl \
  --pairs-per-chunk 4 \
  --max-examples 40 \
  --style-file style_legal.txt
```

### 示例 B：命令行直接写 style（短说明）

```bash
python generate_alpaca_jsonl.py \
  --input law.txt \
  --output law_qa.jsonl \
  --model "你第三方支持的模型名" \
  --pairs-per-chunk 3 \
  --max-examples 40 \
  --style "生成法律问答：问题围绕要件/程序/例外/责任；回答严谨；不得引入文本外信息；不足说明文本未提供；不出现‘根据上文/本文’。"
```

------

## 5) 常见现象

- 生成失败时会把模型原始输出写到：`alpaca.failures.txt`（用于查看为什么 JSON 解析失败或风格不符合）。