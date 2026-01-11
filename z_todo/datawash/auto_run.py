import subprocess
import sys
import os

# ================= 配置区 =================
# 1. 锚点: 获取脚本所在绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 脚本路径
WASH_SCRIPT = os.path.join(BASE_DIR, "wash.py")
GENERATE_SCRIPT = os.path.join(BASE_DIR, "generate_alpaca_jsonl.py")

# 3. 数据流转路径
# 输入: 你的原始文件 (可以是单个文件，也可以是文件夹!)
RAW_INPUT = os.path.join(BASE_DIR, "test_data.txt")

# 中转: 清洗后的文件存放目录
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned_storage")

# 输出: 最终数据集
FINAL_OUTPUT = os.path.join(BASE_DIR, "final_dataset.jsonl")


# ==========================================

def run():
    # --- 第一步：清洗 (Wash) ---
    print(f"🚀 [1/2] 正在清洗数据...")
    print(f"    输入: {RAW_INPUT}")
    print(f"    输出目录: {CLEANED_DIR}")

    cmd_wash = [
        sys.executable,
        WASH_SCRIPT,
        "--input", RAW_INPUT,
        "--output", CLEANED_DIR
    ]

    try:
        subprocess.run(cmd_wash, check=True)
    except subprocess.CalledProcessError:
        print("❌ 清洗步骤失败，请检查错误日志。")
        return

    # --- 第二步：生成 (Generate) ---
    print(f"\n🚀 [2/2] 正在生成 Alpaca 数据集...")
    # 注意: generate 脚本支持接收一个目录作为 input，它会自动扫描里面的 txt

    cmd_gen = [
        sys.executable,
        GENERATE_SCRIPT,
        "--input", CLEANED_DIR,  # 直接把清洗结果目录传给它
        "--output", FINAL_OUTPUT,
        "--pairs-per-chunk", "3",
        "--max-examples", "50"
    ]

    try:
        subprocess.run(cmd_gen, check=True)
        print(f"\n🎉 流程结束！最终文件: {FINAL_OUTPUT}")
    except subprocess.CalledProcessError:
        print("❌ 生成步骤失败。")


if __name__ == "__main__":
    if not os.path.exists(RAW_INPUT):
        print(f"❌ 找不到原始输入: {RAW_INPUT}")
    else:
        run()