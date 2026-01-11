## 训练、微调:

llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

```
### examples/train_lora/llama3_lora_sft.yaml
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct

stage: sft
do_train: true
finetuning_type: lora
lora_target: all

dataset: identity,alpaca_en_demo
template: llama3
cutoff_len: 1024
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

output_dir: saves/llama3-8b/lora/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500
```


## 合并、量化

可以通过 llamafactory-cli export merge_config.yaml 指令来合并模型。其中 merge_config.yaml 需要您根据不同情况进行配置。

examples/merge_lora/llama3_lora_sft.yaml 提供了合并时的配置示例。

```
### examples/merge_lora/llama3_lora_sft.yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
adapter_name_or_path: saves/llama3-8b/lora/sft
template: llama3
finetuning_type: lora

### export
export_dir: models/llama3_lora_sft
export_size: 2
export_device: cpu
export_legacy_format: false
```

LLaMA-Factory 支持多种量化方法，包括: AQLM，AWQ，GPTQ，QLoRA…

GPTQ 等后训练量化方法(Post Training Quantization)是一种在训练后对预训练模型进行量化的方法。我们通过量化技术将高精度表示的预训练模型转换为低精度的模型，从而在避免过多损失模型性能的情况下减少显存占用并加速推理，我们希望低精度数据类型在有限的表示范围内尽可能地接近高精度数据类型的表示，因此我们需要指定量化位数 export_quantization_bit 以及校准数据集 export_quantization_dataset。

在进行模型合并时，请指定：

model_name_or_path: 预训练模型的名称或路径

template: 模型模板

export_dir: 导出路径

export_quantization_bit: 量化位数

export_quantization_dataset: 量化校准数据集

export_size: 最大导出模型文件大小

export_device: 导出设备

export_legacy_format: 是否使用旧格式导出

下面提供一个配置文件示例：

```
### examples/merge_lora/llama3_gptq.yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
template: llama3

### export
export_dir: models/llama3_gptq
export_quantization_bit: 4
export_quantization_dataset: data/c4_demo.json
export_size: 2
export_device: cpu
export_legacy_format: false
QLoRA 是一种在 4-bit 量化模型基础上使用 LoRA 方法进行训练的技术。它在极大地保持了模型性能的同时大幅减少了显存占用和推理时间。
```

警告

不要使用量化模型或设置量化位数 quantization_bit

下面提供一个配置文件示例：
```
### examples/merge_lora/llama3_q_lora.yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
adapter_name_or_path: saves/llama3-8b/lora/sft
template: llama3
finetuning_type: lora

### export
export_dir: models/llama3_lora_sft
export_size: 2
export_device: cpu
export_legacy_format: false
```


## 推理

LLaMA-Factory 支持多种推理方式。

可以使用 llamafactory-cli chat inference_config.yaml 或 llamafactory-cli webchat inference_config.yaml 进行推理与模型对话。对话时配置文件只需指定原始模型 model_name_or_path 和 template ，并根据是否是微调模型指定 adapter_name_or_path 和 finetuning_type。

如果您希望向模型输入大量数据集并保存推理结果，您可以启动 vllm 推理引擎对大量数据集进行快速的批量推理。您也可以通过 部署 api 服务的形式通过 api 调用来进行批量推理。

默认情况下，模型推理将使用 Huggingface 引擎。 您也可以指定 infer_backend: vllm 以使用 vllm 推理引擎以获得更快的推理速度。

备注: 使用任何方式推理时，模型 model_name_or_path 需要存在且与 template 相对应。

原始模型推理配置: 对于原始模型推理， inference_config.yaml 中 只需指定原始模型 model_name_or_path 和 template 即可。

'''
### examples/inference/llama3.yaml
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
template: llama3
infer_backend: huggingface #choices： [huggingface, vllm]
'''

微调模型推理配置
对于微调模型推理，除原始模型和模板外，还需要指定适配器路径 adapter_name_or_path 和微调类型 finetuning_type。

'''
### examples/inference/llama3_lora_sft.yaml
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
adapter_name_or_path: saves/llama3-8b/lora/sft
template: llama3
finetuning_type: lora
infer_backend: huggingface #choices： [huggingface, vllm]
'''

多模态模型
对于多模态模型，您可以运行以下指令进行推理。 
llamafactory-cli webchat examples/inference/llava1_5.yaml

examples/inference/llava1_5.yaml 的配置示例如下：

'''
model_name_or_path: llava-hf/llava-1.5-7b-hf
template: vicuna
infer_backend: huggingface #choices： [huggingface, vllm]
'''

批量推理
数据集
您可以通过以下指令启动 vllm 推理引擎并使用数据集进行批量推理：
python scripts/vllm_infer.py --model_name_or_path path_to_merged_model --dataset alpaca_en_demo

api
如果您需要使用 api 进行批量推理，您只需指定模型、适配器（可选）、模板、微调方式等信息。

下面是一个配置文件的示例：
'''
### examples/inference/llama3_lora_sft.yaml
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
adapter_name_or_path: saves/llama3-8b/lora/sft
template: llama3
finetuning_type: lora
'''

下面是一个启动并调用 api 服务的示例：

您可以使用 API_PORT=8000 CUDA_VISIBLE_DEVICES=0 llamafactory-cli api examples/inference/llama3_lora_sft.yaml 启动 api 服务并运行以下示例程序进行调用：

'''
# api_call_example.py
from openai import OpenAI
client = OpenAI(api_key="0",base_url="http://0.0.0.0:8000/v1")
messages = [{"role": "user", "content": "Who are you?"}]
result = client.chat.completions.create(messages=messages, model="meta-llama/Meta-Llama-3-8B-Instruct")
print(result.choices[0].message)
'''


## 评估

通用能力评估
在完成模型训练后，您可以通过 llamafactory-cli eval examples/train_lora/llama3_lora_eval.yaml 来评估模型效果。

配置示例文件 examples/train_lora/llama3_lora_eval.yaml 具体如下：

### examples/train_lora/llama3_lora_eval.yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
adapter_name_or_path: saves/llama3-8b/lora/sft # 可选项

### method
finetuning_type: lora

### dataset
task: mmlu_test # mmlu_test, ceval_validation, cmmlu_test
template: fewshot
lang: en
n_shot: 5

### output
save_dir: saves/llama3-8b/lora/eval

### eval
batch_size: 4
NLG 评估
此外，您还可以通过 llamafactory-cli train examples/extras/nlg_eval/llama3_lora_predict.yaml 来获得模型的 BLEU 和 ROUGE 分数以评价模型生成质量。

配置示例文件 examples/extras/nlg_eval/llama3_lora_predict.yaml 具体如下：

### examples/extras/nlg_eval/llama3_lora_predict.yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
adapter_name_or_path: saves/llama3-8b/lora/sft

### method
stage: sft
do_predict: true
finetuning_type: lora

### dataset
eval_dataset: identity,alpaca_en_demo
template: llama3
cutoff_len: 2048
max_samples: 50
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/llama3-8b/lora/predict
overwrite_output_dir: true

### eval
per_device_eval_batch_size: 1
predict_with_generate: true
ddp_timeout: 180000000
同样，您也通过在指令 python scripts/vllm_infer.py --model_name_or_path path_to_merged_model --dataset alpaca_en_demo 中指定模型、数据集以使用 vllm 推理框架以取得更快的推理速度。