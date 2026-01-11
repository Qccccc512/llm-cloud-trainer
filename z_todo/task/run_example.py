"""
快速使用示例脚本
"""

from data_augmentation import AlpacaDataAugmentation

# 配置参数
API_KEY = ""  # 替换为你的API密钥
BASE_URL = "https://api.deepseek.com/v1"  # 或其他兼容的API地址
MODEL = "deepseek-chat"  # 或其他模型

# 输入输出文件
INPUT_FILE = "sample_data.json"
OUTPUT_FILE = "augmented_data.json"

# 增强参数（按照图片配置）
FEW_SHOT_NUM = 10  # 指令生成依赖样本数
SIMILARITY_THRESHOLD = 0.7  # 过滤相似度阈值
GENERATE_NUM = 10  # 生成样本数


def main():
    print("=" * 60)
    print("Alpaca数据增强工具")
    print("=" * 60)
    
    # 初始化增强器
    augmenter = AlpacaDataAugmentation(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
        few_shot_num=FEW_SHOT_NUM,
        similarity_threshold=SIMILARITY_THRESHOLD,
        generate_num=GENERATE_NUM
    )
    
    # 加载原始数据
    print(f"\n正在加载数据: {INPUT_FILE}")
    original_data = augmenter.load_data(INPUT_FILE)
    
    # 执行数据增强
    print(f"\n开始数据增强...")
    augmented_data = augmenter.augment_data(original_data)
    
    # 合并并保存
    all_data = original_data + augmented_data
    augmenter.save_data(all_data, OUTPUT_FILE)
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("数据增强完成！")
    print("=" * 60)
    print(f"原始数据: {len(original_data)} 条")
    print(f"新增数据: {len(augmented_data)} 条")
    print(f"总计数据: {len(all_data)} 条")
    print(f"输出文件: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
