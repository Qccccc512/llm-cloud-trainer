# Alpaca格式数据增强工具

使用大模型生成更多的Alpaca格式训练数据的Python脚本。

## 功能特点

✓ **智能生成**：基于few-shot learning，从示例中学习模式生成新数据
✓ **相似度过滤**：自动过滤与原数据相似度过高的样本
✓ **灵活配置**：支持自定义生成数量、相似度阈值等参数
✓ **多模型支持**：兼容OpenAI API及其他兼容接口

## 配置参数

根据图片中的配置要求：

- **指令生成依赖样本数**：10（每次生成时参考的示例数量）
- **过滤相似度阈值**：0.7（相似度超过此值的样本会被过滤）
- **生成样本数**：10（目标生成的新样本数量）

## 安装依赖

```bash
pip install openai sentence-transformers scikit-learn numpy
```

## 使用方法

### 基本用法

```bash
python data_augmentation.py \
  --input sample_data.json \
  --output augmented_data.json \
  --api_key YOUR_API_KEY \
  --few_shot_num 10 \
  --similarity_threshold 0.7 \
  --generate_num 10
```

### 参数说明

- `--input`: 输入的Alpaca格式数据文件路径（必需）
- `--output`: 输出的增强后数据文件路径（必需）
- `--api_key`: 大模型API密钥（必需）
- `--base_url`: API基础URL（默认：https://api.openai.com/v1）
- `--model`: 使用的模型名称（默认：gpt-3.5-turbo）
- `--few_shot_num`: 指令生成依赖样本数（默认：10）
- `--similarity_threshold`: 过滤相似度阈值（默认：0.7）
- `--generate_num`: 生成样本数（默认：10）

### 使用其他API（如国内大模型）

```bash
python data_augmentation.py \
  --input sample_data.json \
  --output augmented_data.json \
  --api_key YOUR_API_KEY \
  --base_url https://api.deepseek.com/v1 \
  --model deepseek-chat \
  --few_shot_num 10 \
  --similarity_threshold 0.7 \
  --generate_num 10
```

## Alpaca数据格式

输入数据应为JSON格式的列表，每个元素包含：

```json
{
  "instruction": "问题或指令",
  "input": "可选的输入内容",
  "output": "对应的回答或输出"
}
```

## 工作流程

1. **加载数据**：读取原始Alpaca格式数据
2. **随机采样**：随机选择指定数量的样本作为few-shot示例
3. **构建Prompt**：根据配置的prompt模板和示例构建生成提示
4. **调用大模型**：使用大模型API生成新的问答对
5. **相似度检测**：计算新样本与现有数据的相似度
6. **过滤保存**：过滤掉相似度过高的样本，保存合格数据

## Prompt模板

脚本使用的prompt模板遵循图片中的要求：

```
请你仔细观察多个示例数据的输入和输出，按照你的理解，总结出相应规律，
然后写出一个新的【问题】和【回答】。注意，新生成的【问题】和【回答】
需要满足如下要求：

1. 生成的【问题】和【回答】不能与输入的【问题】和【回答】一致，
   但是需要保持格式相同。
2. 生成的【问题】不一定要局限于输入【问题】的话题或领域，
   生成的【回答】需要正确回答生成的【问题】。
3. 提供的【问题】和【回答】可能是多轮对话，生成的【问题】和【回答】
   也可以是多轮，但是需要保持格式相同。
4. 生成的【问题】和【回答】必须成对出现，而且【问题】需要在【回答】之前。
```

## 示例

查看 `sample_data.json` 获取示例数据格式。

## 注意事项

1. 确保API密钥有效且有足够的调用额度
2. 生成数量较大时可能需要较长时间
3. 相似度阈值越低，过滤越严格，可能需要更多尝试次数
4. 建议先用小数据量测试配置是否合适

## 输出示例

```
加载了 5 条原始数据
开始生成 10 条新数据...
参数配置：few_shot_num=10, similarity_threshold=0.7

尝试 1: 生成新样本...
✓ 成功生成第 1 条数据
  问题: 什么是深度学习？...

...

数据增强完成！共生成 10 条新数据（尝试了 12 次）

总结:
  原始数据: 5 条
  新增数据: 10 条
  总计数据: 15 条
```

## 许可证

MIT License
