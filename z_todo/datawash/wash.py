import argparse
import fitz  # PyMuPDF
import re
import unicodedata
import os
from pathlib import Path
from typing import List
from opencc import OpenCC
from tqdm import tqdm  # 建议安装: pip install tqdm，用于显示进度条


# ==========================================
# 核心逻辑层: DataCleaner (文本清洗)
# ==========================================
class DataCleaner:
    def __init__(self):
        self.cc = OpenCC('t2s')
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.html_pattern = re.compile(r'<[^>]+>')
        self.email_pattern = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
        self.phone_pattern = re.compile(r'(?<!\d)(1[3-9]\d{9})(?!\d)')
        self.valid_char_pattern = re.compile(r'[\u4e00-\u9fa5a-zA-Z0-9\s,.!?;:()，。！？；：、（）]')

    def process_line(self, line: str) -> str:
        line = line.strip()
        if not line: return None

        # 1. 标准化
        line = unicodedata.normalize('NFKC', line)
        line = self.cc.convert(line)

        # 2. 去噪
        line = self.url_pattern.sub('', line)
        line = self.html_pattern.sub('', line)
        line = "".join(ch for ch in line if unicodedata.category(ch)[0] != "C" or ch in ['\n', '\t', '\r'])

        # 3. 脱敏
        line = self.email_pattern.sub('[EMAIL]', line)
        line = self.phone_pattern.sub('[MOBILEPHONE]', line)

        # 4. 质量过滤 (阈值 0.3)
        if not self.quality_check(line):
            return None

        return line

    def quality_check(self, text, threshold=0.3):
        if not text.strip(): return False
        total_len = len(text)
        valid_len = len("".join(self.valid_char_pattern.findall(text)))
        special_ratio = 1 - (valid_len / total_len)
        return special_ratio <= threshold


# ==========================================
# 核心逻辑层: PDFProcessor (PDF 处理)
# ==========================================
class PDFProcessor:
    def __init__(self):
        self.header_height = 60
        self.footer_height = 50
        self.cleaner = DataCleaner()

    def extract_from_path(self, pdf_path: Path) -> List[str]:
        # PyMuPDF 支持直接传 Path 对象或字符串
        doc = fitz.open(str(pdf_path))
        full_text = []

        for page in doc:
            page_height = page.rect.height
            blocks = page.get_text("blocks")

            page_content = []
            for b in blocks:
                x0, y0, x1, y1, text, block_no, block_type = b
                if block_type == 1: continue  # 图片
                if y0 < self.header_height: continue  # 页眉
                if y1 > (page_height - self.footer_height): continue  # 页脚

                clean_text = text.strip()
                if clean_text:
                    page_content.append(clean_text)

            # 块合并并清洗
            raw_page_text = "\n".join(page_content)
            for line in raw_page_text.split('\n'):
                cleaned = self.cleaner.process_line(line)
                if cleaned:
                    full_text.append(cleaned)

        doc.close()
        return full_text


# ==========================================
# 核心逻辑层: TXTProcessor (TXT 处理)
# ==========================================
class TXTProcessor:
    def __init__(self):
        self.cleaner = DataCleaner()

    def extract_from_path(self, txt_path: Path) -> List[str]:
        cleaned_lines = []
        try:
            content = txt_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                content = txt_path.read_text(encoding='gbk')
            except Exception:
                print(f"⚠️ 警告: 无法解码文件 {txt_path.name}，已跳过。")
                return []

        for line in content.splitlines():
            res = self.cleaner.process_line(line)
            if res:
                cleaned_lines.append(res)
        return cleaned_lines


# ==========================================
# 主程序逻辑 (CLI)
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="DataLoom 本地清洗工具")
    parser.add_argument("--input", required=True, help="输入文件路径 (.pdf/.txt) 或 包含文件的文件夹")
    parser.add_argument("--output", required=True, help="清洗结果输出文件夹")

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集需要处理的文件
    files_to_process = []
    if input_path.is_file():
        files_to_process.append(input_path)
    elif input_path.is_dir():
        # 递归扫描所有 pdf 和 txt
        files_to_process.extend(input_path.rglob("*.pdf"))
        files_to_process.extend(input_path.rglob("*.txt"))
    else:
        print(f"❌ 错误: 输入路径不存在: {input_path}")
        return

    if not files_to_process:
        print("⚠️ 未找到任何 .pdf 或 .txt 文件。")
        return

    print(f"🧹 准备处理 {len(files_to_process)} 个文件...")

    # 初始化处理器
    pdf_proc = PDFProcessor()
    txt_proc = TXTProcessor()

    success_count = 0

    # 开始循环处理
    for file_p in tqdm(files_to_process, desc="Cleaning"):
        try:
            filename = file_p.name.lower()
            lines = []

            if filename.endswith(".pdf"):
                lines = pdf_proc.extract_from_path(file_p)
            elif filename.endswith(".txt"):
                lines = txt_proc.extract_from_path(file_p)

            if not lines:
                continue

            # 构造输出文件名: 统一改为 .txt 后缀
            # 例如: report.pdf -> cleaned_report.txt
            new_name = f"cleaned_{file_p.stem}.txt"
            out_file = output_dir / new_name

            # 写入结果
            out_file.write_text("\n".join(lines), encoding="utf-8")
            success_count += 1

        except Exception as e:
            print(f"\n❌ 处理文件 {file_p.name} 时出错: {e}")

    print(f"\n✅ 清洗完成! 成功: {success_count} / {len(files_to_process)}")
    print(f"📁 结果已保存至: {output_dir}")


if __name__ == "__main__":
    main()