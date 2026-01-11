"""
Alpaca格式数据增强脚本
利用大模型生成更多的训练数据
"""

import json
import random
from typing import List, Dict
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import argparse
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class AlpacaDataAugmentation:
    """Alpaca数据增强类"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-3.5-turbo",
        few_shot_num: int = 10,
        similarity_threshold: float = 0.7,
        generate_num: int = 10
    ):
        """
        初始化数据增强器
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            model: 使用的模型名称
            few_shot_num: 指令生成依赖样本数
            similarity_threshold: 过滤相似度阈值
            generate_num: 生成样本数
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.few_shot_num = few_shot_num
        self.similarity_threshold = similarity_threshold
        self.generate_num = generate_num
        
        # 加载句子嵌入模型用于相似度计算
        print("加载句子嵌入模型...")
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
    def load_data(self, file_path: str) -> List[Dict]:
        """加载Alpaca格式数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"加载了 {len(data)} 条原始数据")
        return data
    
    def format_few_shot_examples(self, examples: List[Dict]) -> str:
        """格式化few-shot示例"""
        formatted = []
        for i, example in enumerate(examples, 1):
            instruction = example.get('instruction', '')
            input_text = example.get('input', '')
            output = example.get('output', '')
            
            if input_text:
                formatted.append(f"示例{i}：\n【问题】{instruction}\n【输入】{input_text}\n【回答】{output}")
            else:
                formatted.append(f"示例{i}：\n【问题】{instruction}\n【回答】{output}")
        
        return "\n\n".join(formatted)
    
    def create_prompt(self, few_shot_examples: str) -> str:
        """创建生成prompt"""
        prompt = f"""请你仔细观察多个示例数据的输入和输出，按照你的理解，总结出相应规律，然后写出一个新的【问题】和【回答】。注意，新生成的【问题】和【回答】需要满足如下要求：

1. 生成的【问题】和【回答】不能与输入的【问题】和【回答】一致，但是需要保持格式相同。
2. 生成的【问题】不一定要局限于输入【问题】的话题或领域，生成的【回答】需要正确回答生成的【问题】。
3. 提供的【问题】和【回答】可能是多轮对话，生成的【问题】和【回答】也可以是多轮，但是需要保持格式相同。
4. 生成的【问题】和【回答】必须成对出现，而且【问题】需要在【回答】之前。

参考示例：
{few_shot_examples}

请生成一个新的【问题】和【回答】："""
        
        return prompt
    
    def generate_one_sample(self, few_shot_examples: str) -> Dict:
        """使用大模型生成一个新样本"""
        prompt = self.create_prompt(few_shot_examples)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的数据生成助手，擅长根据示例生成高质量的问答数据。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            return self.parse_generated_content(content)
            
        except Exception as e:
            print(f"生成数据时出错: {e}")
            return None
    
    def parse_generated_content(self, content: str) -> Dict:
        """解析生成的内容，提取问题和回答"""
        lines = content.strip().split('\n')
        instruction = ""
        input_text = ""
        output = ""
        
        current_field = None
        
        for line in lines:
            line = line.strip()
            if '【问题】' in line:
                instruction = line.split('【问题】')[-1].strip()
                current_field = 'instruction'
            elif '【输入】' in line:
                input_text = line.split('【输入】')[-1].strip()
                current_field = 'input'
            elif '【回答】' in line:
                output = line.split('【回答】')[-1].strip()
                current_field = 'output'
            elif current_field and line:
                # 继续添加到当前字段
                if current_field == 'instruction':
                    instruction += '\n' + line
                elif current_field == 'input':
                    input_text += '\n' + line
                elif current_field == 'output':
                    output += '\n' + line
        
        if instruction and output:
            return {
                'instruction': instruction,
                'input': input_text,
                'output': output
            }
        return None
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度"""
        embeddings = self.embedding_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity
    
    def is_similar_to_existing(self, new_sample: Dict, existing_data: List[Dict]) -> bool:
        """检查新样本是否与现有数据过于相似"""
        new_text = new_sample['instruction'] + ' ' + new_sample.get('input', '') + ' ' + new_sample['output']
        
        for existing in existing_data:
            existing_text = existing['instruction'] + ' ' + existing.get('input', '') + ' ' + existing['output']
            similarity = self.compute_similarity(new_text, existing_text)
            
            if similarity > self.similarity_threshold:
                return True
        
        return False
    
    def augment_data(self, original_data: List[Dict]) -> List[Dict]:
        """数据增强主函数"""
        augmented_data = []
        all_data = original_data.copy()
        
        attempts = 0
        max_attempts = self.generate_num * 3  # 最多尝试3倍的次数
        
        print(f"\n开始生成 {self.generate_num} 条新数据...")
        print(f"参数配置：few_shot_num={self.few_shot_num}, similarity_threshold={self.similarity_threshold}")
        
        while len(augmented_data) < self.generate_num and attempts < max_attempts:
            attempts += 1
            
            # 随机选择few-shot样本
            few_shot_samples = random.sample(original_data, min(self.few_shot_num, len(original_data)))
            few_shot_examples = self.format_few_shot_examples(few_shot_samples)
            
            # 生成新样本
            print(f"\n尝试 {attempts}: 生成新样本...")
            new_sample = self.generate_one_sample(few_shot_examples)
            
            if new_sample is None:
                print("生成失败，跳过")
                continue
            
            # 检查相似度
            if self.is_similar_to_existing(new_sample, all_data):
                print(f"生成的样本与现有数据相似度过高（>{self.similarity_threshold}），跳过")
                continue
            
            # 添加到结果
            augmented_data.append(new_sample)
            all_data.append(new_sample)
            print(f"✓ 成功生成第 {len(augmented_data)} 条数据")
            print(f"  问题: {new_sample['instruction'][:50]}...")
            
        print(f"\n数据增强完成！共生成 {len(augmented_data)} 条新数据（尝试了 {attempts} 次）")
        return augmented_data
    
    def save_data(self, data: List[Dict], output_path: str):
        """保存数据"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n数据已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Alpaca格式数据增强工具')
    parser.add_argument('--input', type=str, required=True, help='输入数据文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出数据文件路径')
    parser.add_argument('--api_key', type=str, required=True, help='API密钥')
    parser.add_argument('--base_url', type=str, default='https://api.openai.com/v1', help='API基础URL')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='模型名称')
    parser.add_argument('--few_shot_num', type=int, default=10, help='指令生成依赖样本数')
    parser.add_argument('--similarity_threshold', type=float, default=0.7, help='过滤相似度阈值')
    parser.add_argument('--generate_num', type=int, default=10, help='生成样本数')
    
    args = parser.parse_args()
    
    # 初始化数据增强器
    augmenter = AlpacaDataAugmentation(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        few_shot_num=args.few_shot_num,
        similarity_threshold=args.similarity_threshold,
        generate_num=args.generate_num
    )
    
    # 加载原始数据
    original_data = augmenter.load_data(args.input)
    
    # 执行数据增强
    augmented_data = augmenter.augment_data(original_data)
    
    # 合并并保存
    all_data = original_data + augmented_data
    augmenter.save_data(all_data, args.output)
    
    print(f"\n总结:")
    print(f"  原始数据: {len(original_data)} 条")
    print(f"  新增数据: {len(augmented_data)} 条")
    print(f"  总计数据: {len(all_data)} 条")


if __name__ == "__main__":
    main()
