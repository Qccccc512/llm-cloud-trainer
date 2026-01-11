#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

from tqdm import tqdm
from PyPDF2 import PdfReader

from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI, BadRequestError


# =========================================================
# ✅ 不在代码里写死密钥：优先从命令行参数读取，其次环境变量
# =========================================================
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "").strip()
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()


# -----------------------------
# Output schema (model should NOT output input)
# We'll write input="" ourselves.
# -----------------------------
class QAExample(BaseModel):
    instruction: str = Field(..., min_length=1)
    output: str = Field(..., min_length=1)


class QABatch(BaseModel):
    examples: List[QAExample] = Field(..., min_length=1)


# -----------------------------
# Optional built-in profiles (quick presets)
# -----------------------------
PROFILE_SPECS_ZH = {
    "concept_qa": "以“概念解释/定义/区别辨析”为主。",
    "process_qa": "以“流程/步骤/清单/方法论”为主。",
    "pitfall_qa": "以“常见误区/纠偏建议”为主。",
    "application_qa": "以“场景应用/案例推演”为主。",
    "mixed": "混合：概念/流程/误区/应用均衡覆盖。",
}


# -----------------------------
# IO utils (txt/pdf)
# -----------------------------
def read_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n".join(parts)


def load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return read_txt(path)
    if suffix == ".pdf":
        return read_pdf(path)
    raise ValueError(f"Unsupported file type: {path}")


def iter_input_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return
    for p in input_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".pdf"}:
            yield p


# -----------------------------
# text cleaning + chunking
# -----------------------------
def normalize_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\r\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def iter_chunks(text: str, max_chars: int, overlap: int):
    """
    ✅ 保证 start 单调递增，避免 overlap 导致死循环
    """
    text = normalize_text(text)
    if not text:
        return

    n = len(text)
    if max_chars <= 0:
        raise ValueError("chunk-chars must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= max_chars:
        raise ValueError("overlap must be < chunk-chars")

    start = 0
    while start < n:
        end = min(n, start + max_chars)
        window = text[start:end]

        search_from = int(len(window) * 0.6)
        candidates = [
            window.rfind("。", search_from),
            window.rfind("？", search_from),
            window.rfind(". ", search_from),
            window.rfind("\n\n", search_from),
            window.rfind("\n", search_from),
        ]
        cut = max(candidates)
        if cut > 0:
            end = start + cut + 1

        chunk = text[start:end].strip()
        if chunk:
            yield chunk

        if end >= n:
            break

        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start


# -----------------------------
# Prompting (supports custom user style)
# -----------------------------
def build_prompts(
    chunk: str,
    n_pairs: int,
    lang: str,
    profile: str,
    style_text: str,
) -> Tuple[str, str]:
    """
    style_text：用户希望的数据集样子（例如“法律问答风格”），会拼进 system prompt。
    """
    style_text = (style_text or "").strip()
    profile_spec = PROFILE_SPECS_ZH.get(profile, PROFILE_SPECS_ZH["mixed"])

    if lang.lower().startswith("zh"):
        system = (
            "你是一个高精度数据集生成器。\n"
            "任务：根据给定文本，生成用于训练的“单轮问答”样本。\n\n"
            f"内置风格（profile，仅供参考）：{profile_spec}\n"
        )

        if style_text:
            system += (
                "\n【用户指定的数据集风格要求】\n"
                f"{style_text}\n"
            )

        system += (
            "\n【硬性规则】\n"
            f"- 必须生成且只生成 {n_pairs} 条样本。\n"
            "- 每条样本只包含两个字段：instruction（问题） 和 output（回答）。\n"
            "- 不要输出 input 字段；脚本会统一写入 input=\"\"。\n"
            "- instruction 必须自洽、可独立理解：不要出现“根据文本/材料/如上所述/本文/上文”等措辞。\n"
            "- output 必须直接回答 instruction，尽量用通用表达，不要大段照抄原文。\n"
            "- 不要编造文本中不存在的事实；信息不足时要明确说明“无法从文本确定/文本未提供”。\n"
            "- 禁止输出 Markdown、禁止输出多余解释；只输出严格 JSON。\n\n"
            "【输出格式（严格 JSON）】\n"
            "{\"examples\":[{\"instruction\":\"...\",\"output\":\"...\"}, ...]}\n"
        )
    else:
        system = (
            "You are a high-precision dataset generator.\n"
            "Generate single-turn QA samples grounded in the provided text.\n"
            f"Produce exactly {n_pairs} samples.\n"
        )
        if style_text:
            system += f"\nUSER STYLE REQUIREMENTS:\n{style_text}\n"
        system += (
            "\nHard rules:\n"
            "- Each sample must contain ONLY: instruction and output.\n"
            "- Do NOT include an input field.\n"
            "- Questions must be self-contained; do not refer to 'the passage' or 'the text above'.\n"
            "- Answers must be grounded; do not invent.\n"
            "- Return ONLY strict JSON: {\"examples\":[{\"instruction\":\"...\",\"output\":\"...\"}, ...]}\n"
        )

    user = f"TEXT:\n{chunk}\n"
    return system, user


# -----------------------------
# Model call (chat.completions JSON + retries)
# -----------------------------
def extract_json_object(s: str) -> str:
    s = s.strip()
    if "{" in s and "}" in s:
        return s[s.find("{") : s.rfind("}") + 1]
    return s


def generate_batch_via_chat_json(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    max_output_tokens: int,
    retries: int = 3,
    failure_log_path: str = "alpaca.failures.txt",
) -> QABatch:
    last_err: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_output_tokens,
            temperature=0,
        )

        text = (completion.choices[0].message.content or "").strip()

        # 防止服务端异常返回无限/超大
        if len(text) > 200_000:
            raise RuntimeError(f"Response too large: {len(text)} chars. Possible streaming/ignore max_tokens.")

        text2 = extract_json_object(text)

        try:
            data = json.loads(text2)
            return QABatch.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = e
            print(f"[WARN] JSON parse/validate failed (attempt {attempt}/{retries}): {e}")
            print("[WARN] output head:", text[:400].replace("\n", "\\n"))

            # 失败落盘，方便排查
            try:
                with open(failure_log_path, "a", encoding="utf-8") as ff:
                    ff.write("\n" + "=" * 80 + "\n")
                    ff.write(f"attempt={attempt}\n")
                    ff.write(text + "\n")
            except Exception:
                pass

            if attempt < retries:
                system = system + "\n上一次输出不是合法 JSON。请严格只输出 JSON 对象，不要任何额外文字。\n"

    raise RuntimeError(f"Failed to get valid JSON after retries. Last error: {last_err}")


# -----------------------------
# Filtering
# -----------------------------
def is_bad_example(instruction: str, output: str) -> bool:
    # 避免在最终数据中出现这些引用措辞
    bad_markers = ["SOURCE_TEXT", "TEXT:", "根据以上文本", "根据材料", "如上所述", "本文", "上文", "这段文字"]
    s = (instruction + " " + output).strip()
    if any(m in s for m in bad_markers):
        return True
    # 防止超长摘抄
    if len(output) > 2500:
        return True
    if len(instruction) > 500:
        return True
    return False


def read_style_from_args(style: str, style_file: str) -> str:
    style = (style or "").strip()
    style_file = (style_file or "").strip()

    if style_file:
        p = Path(style_file).expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"--style-file not found: {p}")
        return p.read_text(encoding="utf-8", errors="ignore").strip()

    return style


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="输入文件或目录（递归扫描 .txt/.pdf）")
    ap.add_argument("--output", required=True, help="输出 jsonl 路径")
    ap.add_argument("--api-key", default=DEFAULT_API_KEY, help="OpenAI 兼容接口的 API Key（也可用环境变量 OPENAI_API_KEY）")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI 兼容接口 base_url（也可用环境变量 OPENAI_BASE_URL，通常以 /v1 结尾）")
    ap.add_argument("--lang", default="zh-CN", help="样本语言，例如 zh-CN / en")

    # ✅ 新增：用户自定义“数据集长什么样”
    ap.add_argument("--style", default="", help="用户自定义的数据集风格说明（建议用引号包起来）")
    ap.add_argument("--style-file", default="", help="从文件读取风格说明（用于长 prompt）")

    # 预置 profile 仍保留（可选）
    ap.add_argument(
        "--profile",
        default="mixed",
        choices=list(PROFILE_SPECS_ZH.keys()),
        help="内置风格（可选）：concept_qa/process_qa/pitfall_qa/application_qa/mixed",
    )
    ap.add_argument("--model", default = "deepseek-chat", help="第三方服务支持的模型名")
    ap.add_argument("--chunk-chars", type=int, default=2500, help="每块最大字符数")
    ap.add_argument("--overlap", type=int, default=200, help="相邻块重叠字符数（必须小于 chunk-chars）")
    ap.add_argument("--pairs-per-chunk", type=int, default=3, help="每个 chunk 生成多少条问答")
    ap.add_argument("--max-examples", type=int, default=200, help="最多生成多少条样本（到达即停止）")
    ap.add_argument("--max-output-tokens", type=int, default=1200, help="单次请求最大输出 tokens")
    args = ap.parse_args()

    style_text = read_style_from_args(args.style, args.style_file)

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.api_key:
        raise SystemExit("缺少 --api-key（或环境变量 OPENAI_API_KEY）")
    if not args.base_url:
        raise SystemExit("缺少 --base-url（或环境变量 OPENAI_BASE_URL）")

    seen = set()
    total_written = 0

    with OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=60.0,
        max_retries=0,
        # 若第三方不是 Bearer：可以用下面两行替代
        # api_key="",
        # default_headers={"api-key": args.api_key},
    ) as client:

        with output_path.open("w", encoding="utf-8") as f:
            files = list(iter_input_files(input_path))
            if not files:
                raise SystemExit("没有找到 .txt / .pdf 文件")

            for file_path in tqdm(files, desc="Files"):
                raw = load_text(file_path)
                raw = normalize_text(raw)
                if len(raw) < 200:
                    continue

                for chunk in tqdm(
                    iter_chunks(raw, max_chars=args.chunk_chars, overlap=args.overlap),
                    desc=f"Chunks ({file_path.name})",
                    leave=False,
                ):
                    if total_written >= args.max_examples:
                        break
                    if len(chunk) < 200:
                        continue

                    system, user = build_prompts(
                        chunk=chunk,
                        n_pairs=args.pairs_per_chunk,
                        lang=args.lang,
                        profile=args.profile,
                        style_text=style_text,
                    )

                    try:
                        batch = generate_batch_via_chat_json(
                            client=client,
                            model=args.model,
                            system=system,
                            user=user,
                            max_output_tokens=args.max_output_tokens,
                            retries=3,
                        )
                    except BadRequestError as e:
                        raise SystemExit(
                            f"请求失败（可能模型名不对/参数不兼容）。\n"
                            f"model={args.model}\n"
                            f"原始错误：{e}"
                        )
                    except Exception as e:
                        raise RuntimeError(f"Generation failed: {e}")

                    for ex in batch.examples:
                        if total_written >= args.max_examples:
                            break

                        instruction = ex.instruction.strip()
                        output = ex.output.strip()
                        if not instruction or not output:
                            continue
                        if is_bad_example(instruction, output):
                            continue

                        key = (instruction, output)
                        if key in seen:
                            continue
                        seen.add(key)

                        json.dump(
                            {"instruction": instruction, "input": "", "output": output},
                            f,
                            ensure_ascii=False,
                        )
                        f.write("\n")
                        total_written += 1

                if total_written >= args.max_examples:
                    break

    print(f"Done. Wrote {total_written} examples to: {output_path}")


if __name__ == "__main__":
    main()
