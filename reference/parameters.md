# LLaMA Factory 全参数说明表格
## 一、微调参数（FinetuningArguments）
| 参数名称 | 类型 | 介绍 | 默认值 |
| --- | --- | --- | --- |
| pure_bf16 | bool | 是否以纯bf16精度训练模型(不使用AMP) | False |
| stage | Literal["pt", "sft", "rm", "ppo", "dpo", "kto"] | 训练阶段，rm(reward modeling), pt(pretrain), sft(Supervised Fine-Tuning), | sft |
| finetuning_type | Literal["lora", "freeze", "full"] | 微调方法 | lora |
| use_llama_pro | bool | 是否仅训练扩展块中的参数(LLaMA Pro模式) | False |
| use_adam_mini | bool | 是否使用Adam-mini优化器 | False |
| freeze_vision_tower | bool | MLLM训练时是否冻结视觉塔 | True |
| freeze_multi_modal_projector | bool | MLLM训练时是否冻结多模态投影器 | True |
| train_mm_proj_only | bool | 是否仅训练多模态投影器 | False |
| compute_accuracy | bool | 是否在评估时计算token级别的准确率 | False |
| disable_shuffling | bool | 是否禁用训练集的随机打乱 | False |
| plot_loss | bool | 是否保存训练过程中的损失曲线 | False |
| include_effective_tokens_per_second | bool | 是否计算有效的每秒token数 | False |

## 二、LoRA参数（LoraArguments）
| 参数名称 | 类型 | 介绍 | 默认值 |
| --- | --- | --- | --- |
| additional_target | Optional[str] | 除LoRA层之外设置为可训练并保存在最终检查点中的模块名称，使用逗号分隔多个模块 | None |
| lora_alpha | Optional[int] | LoRA缩放系数，一般情况下为lora_rank*2 | None |
| lora_dropout | float | LoRA微调中的dropout率 | 0 |
| lora_rank | int | LoRA微调的本征维数r，r越大可训练的参数越多 | 8 |
| lora_target | str | 应用LoRA方法的模块名称，使用逗号分隔多个模块，使用all指定所有模块 | all |
| loraplus_lr_ratio | Optional[float] | LoRA+学习率比例（λ=ηB/ηA），ηA、ηB分别是adapter matrices A与B的学习率 | None |
| loraplus_lr_embedding | Optional[float] | LoRA+嵌入层的学习率 | 1e-6 |
| use_rslora | bool | 是否使用秩稳定LoRA（Rank-Stabilized LoRA） | False |
| use_dora | bool | 是否使用权重分解LoRA（Weight-Decomposed LoRA） | False |
| pissa_init | bool | 是否初始化PiSSA适配器 | False |
| pissa_iter | Optional[int] | PiSSA中FSVD执行的迭代步数，使用-1将其禁用 | 16 |
| pissa_convert | bool | 是否将PiSSA适配器转换为正常的LoRA适配器 | False |
| create_new_adapter | bool | 是否创建一个具有随机初始化权重的新适配器 | False |

## 三、RLHF训练参数
| 参数名称 | 类型 | 介绍 | 默认值 |
| --- | --- | --- | --- |
| pref_beta | float | 偏好损失中的beta参数 | 0.1 |
| pref_ftx | float | DPO训练中的sft loss系数 | 0.0 |
| pref_loss | Literal["sigmoid", "hinge", "ipo", "kto_pair", "orpo", "simpo"] | DPO训练中使用的偏好损失类型 | sigmoid |
| dpo_label_smoothing | float | 标签平滑系数，取值范围为[0,0.5] | 0.0 |
| kto_chosen_weight | float | KTO训练中chosen标签loss的权重 | 1.0 |
| kto_rejected_weight | float | KTO训练中rejected标签loss的权重 | 1.0 |
| simpo_gamma | float | SimPO损失中的reward margin | 0.5 |
| ppo_buffer_size | int | PPO训练中的mini-batch大小 | 1 |
| ppo_epochs | int | PPO训练迭代次数 | 4 |
| ppo_score_norm | bool | 是否在PPO训练中使用归一化分数 | False |
| ppo_target | float | PPO训练中自适应KL控制的目标KL值 | 6.0 |
| ppo_whiten_rewards | bool | PPO训练中是否对奖励进行归一化 | False |
| ref_model | Optional[str] | PPO或DPO训练中使用的参考模型路径 | None |
| ref_model_adapters | Optional[str] | 参考模型的适配器路径 | None |
| ref_model_quantization_bit | Optional[int] | 参考模型的量化位数，支持4位或8位量化 | None |
| reward_model | Optional[str] | PPO训练中使用的奖励模型路径 | None |
| reward_model_adapters | Optional[str] | 奖励模型的适配器路径 | None |
| reward_model_quantization_bit | Optional[int] | 奖励模型的量化位数 | None |
| reward_model_type | Literal["lora", "full", "api"] | PPO训练中使用的奖励模型类型 | lora |

## 四、Freeze参数（FreezeArguments）
| 参数名称 | 类型 | 介绍 | 默认值 |
| --- | --- | --- | --- |
| freeze_trainable_layers | int | 可训练层的数量，正数表示最后n层被设置为可训练的，负数表示前n层被设置为可训练的 | 2 |
| freeze_trainable_modules | str | 可训练层的名称，使用all来指定所有模块 | all |
| freeze_extra_modules | Optional[str] | 除了隐藏层外可以被训练的模块名称，被指定的模块将会被设置为可训练的，使用逗号分隔多个模块 | None |

## 五、Apollo参数（ApolloArguments）
| 参数名称 | 类型 | 介绍 | 默认值 |
| --- | --- | --- | --- |
| use_apollo | bool | 是否使用APOLLO优化器 | False |
| apollo_target | str | 适用APOLLO的模块名称，使用逗号分隔多个模块，使用all指定所有线性模块 | all |
| apollo_rank | int | APOLLO梯度的秩 | 16 |
| apollo_update_interval | int | 更新APOLLO投影的步数间隔 | 200 |
| apollo_scale | float | APOLLO缩放系数 | 32.0 |
| apollo_proj | Literal["svd", "random"] | APOLLO低秩投影算法类型（svd或random） | random |
| apollo_proj_type | Literal["std", "right", "left"] | APOLLO投影类型 | std |
| apollo_scale_type | Literal["channel", "tensor"] | APOLLO缩放类型（channel或tensor） | channel |
| apollo_layerwise | bool | 是否启用层级更新以进一步节省内存 | False |
| apollo_scale_front | bool | 是否在梯度缩放前使用范数增长限制器 | False |

## 六、BAdam参数（BAdamArguments）
| 参数名称 | 类型 | 介绍 | 默认值 |
| --- | --- | --- | --- |
| use_badam | bool | 是否使用BAdam优化器 | False |
| badam_mode | Literal["layer", "ratio"] | BAdam的使用模式 | layer |
| badam_start_block | Optional[int] | layer-wise BAdam的起始块索引 | None |
| badam_switch_mode | Optional[Literal["ascending", "descending", "random", "fixed"]] | layer-wise BAdam中块更新策略 | ascending |
| badam_switch_interval | Optional[int] | layer-wise BAdam中块更新步数间隔，使用-1禁用块更新 | 50 |
| badam_update_ratio | float | ratio-wise BAdam中的更新比例 | 0.05 |
| badam_mask_mode | Literal["adjacent", "scatter"] | BAdam优化器的掩码模式 | adjacent |
| badam_verbose | int | BAdam优化器的详细输出级别，0表示无输出，1表示输出块前缀，2表示输出可训练参数 | 0 |

## 七、GaLore参数（GaLoreArguments）
| 参数名称 | 类型 | 介绍 | 默认值 |
| --- | --- | --- | --- |
| use_galore | bool | 是否使用GaLore算法 | False |
| galore_target | str | 应用GaLore的模块名称，使用逗号分隔多个模块，使用all指定所有线性模块 | all |
| galore_rank | int | GaLore梯度的秩 | 16 |
| galore_update_interval | int | 更新GaLore投影的步数间隔 | 200 |
| galore_scale | float | GaLore的缩放系数 | 0.25 |
| galore_proj_type | Literal["std", "reverse_std", "right", "left", "full"] | GaLore投影的类型 | std |
| galore_layerwise | bool | 是否启用逐层更新以进一步节省内存 | False |

## 八、数据参数（DataArguments）
| 参数名称 | 类型 | 介绍 | 默认值 |
| --- | --- | --- | --- |
| template | Optional[str] | 训练和推理时构造prompt的模板 | None |
| dataset | Optional[str] | 用于训练的数据集名称，使用逗号分隔多个数据集 | None |
| eval_dataset | Optional[str] | 用于评估的数据集名称，使用逗号分隔多个数据集 | None |
| eval_on_each_dataset | Optional[bool] | 是否在每个评估数据集上分开计算loss，默认concate后为整体计算 | False |
| dataset_dir | Optional[Union[str, Dict[str, Any]]] | 存储数据集的文件夹路径，可以是字符串或字典；当为字符串时，表示数据集目录的路径（例如data）；当为字典时，将覆盖默认从本地dataset_info.json加载的行为 | data |
| media_dir | Optional[str] | 存储图像、视频或音频的文件夹路径，如果未指定，默认为dataset_dir | None |
| data_shared_file_system | Optional[bool] | 多机多卡时，不同机器存放数据集的路径是否是共享文件系统；数据集处理在该值为true时只在第一个node发生，为false时在每个node都处理一次 | False |
| cutoff_len | int | 输入的最大token数，超过该长度会被截断 | 2048 |
| train_on_prompt | bool | 是否在输入prompt上进行训练 | False |
| mask_history | bool | 是否仅使用当前对话轮次进行训练 | False |
| streaming | bool | 是否启用数据流模式 | False |
| buffer_size | int | 启用streaming时用于随机选择样本的buffer大小 | 16384 |
| mix_strategy | Literal["concat", "interleave_under", "interleave_over"] | 数据集混合策略 | concat |
| interleave_probs | Optional[str] | 使用interleave策略时，指定从多个数据集中采样的概率，多个数据集的概率用逗号分隔 | None |
| overwrite_cache | bool | 是否覆盖缓存的训练和评估数据集 | False |
| preprocessing_batch_size | int | 预处理时每批次的示例数量 | 1000 |
| preprocessing_num_workers | Optional[int] | 预处理时使用的进程数量 | None |
| max_samples | Optional[int] | 每个数据集的最大样本数，设置后，每个数据集的样本数将被截断至指定的max_samples | None |
| eval_num_beams | Optional[int] | 模型评估时的num_beams参数 | None |
| ignore_pad_token_for_loss | bool | 计算loss时是否忽略pad token | True |
| val_size | float | 验证集相对所使用的训练数据集的大小，取值在[0,1)之间；启用streaming时val_size应是整数 | 0.0 |
| packing | Optional[bool] | 是否启用sequences packing，预训练时默认启用 | None |
| neat_packing | bool | 是否启用不使用cross-attention的sequences packing | False |
| tool_format | Optional[str] | 用于构造函数调用示例的格式 | None |
| tokenized_path | Optional[str] | Tokenized datasets的保存或加载路径；如果路径存在，会加载已有的tokenized datasets；如果路径不存在，则会在分词后将tokenized datasets保存在此路径中 | None |

## 九、模型参数（ModelArguments）
| 参数名称 | 类型 | 介绍 | 默认值 |
| --- | --- | --- | --- |
| model_name_or_path | Optional[str] | 模型路径（本地路径或Huggingface/ModelScope路径） | None |
| adapter_name_or_path | Optional[str] | 适配器路径（本地路径或Huggingface/ModelScope路径），使用逗号分隔多个适配器路径 | None |
| adapter_folder | Optional[str] | 包含适配器权重的文件夹路径 | None |
| cache_dir | Optional[str] | 保存从Hugging Face或ModelScope下载的模型的本地路径 | None |
| use_fast_tokenizer | bool | 是否使用fast_tokenizer | True |
| resize_vocab | bool | 是否调整词表和嵌入层的大小 | False |
| split_special_tokens | bool | 是否在分词时将special token分割 | False |
| new_special_tokens | Optional[str] | 要添加到tokenizer中的special token，多个special token用逗号分隔 | None |
| model_revision | str | 所使用的特定模型版本 | main |
| low_cpu_mem_usage | bool | 是否使用节省内存的模型加载方式 | True |
| rope_scaling | Optional[Literal["linear", "dynamic", "yarn", "llama3"]] | RoPE Embedding的缩放策略 | None |
| flash_attn | Literal["auto", "disabled", "sdpa", "fa2"] | 是否启用FlashAttention来加速训练和推理 | auto |
| shift_attn | bool | 是否启用Shift Short Attention (S²-Attn) | False |
| mixture_of_depths | Optional[Literal["convert", "load"]] | 需要将模型转换为mixture_of_depths (MoD)模型时指定convert；需要加载mixture_of_depths (MoD)模型时指定load | None |
| use_unsloth | bool | 是否使用unsloth优化LoRA微调 | False |
| use_unsloth_gc | bool | 是否使用UNloth的梯度检查点 | False |
| enable_liger_kernel | bool | 是否启用liger内核以加速训练 | False |
| moe_aux_loss_coef | Optional[float] | MoE架构中aux_loss系数，数值越大，各个专家负载越均衡 | None |
| disable_gradient_checkpointing | bool | 是否禁用梯度检查点 | False |
| use_reentrant_gc | bool | 是否启用可重入梯度检查点 | True |
| upcast_layernorm | bool | 是否将layernorm层权重精度提高至fp32 | False |
| upcast_lmhead_output | bool | 是否将lm_head输出精度提高至fp32 | False |
| train_from_scratch | bool | 是否随机初始化模型权重 | False |
| infer_backend | Literal["huggingface", "vllm"] | 推理时使用的后端引擎 | huggingface |
| offload_folder | str | 卸载模型权重的路径 | offload |
| use_cache | bool | 是否在生成时使用KV缓存 | True |
| infer_dtype | Literal["auto", "float16", "bfloat16", "float32"] | 推理时使用的模型权重和激活值的数据类型 | auto |
| hf_hub_token | Optional[str] | 用于登录HuggingFace的验证token | None |
| ms_hub_token | Optional[str] | 用于登录ModelScope Hub的验证token | None |
| om_hub_token | Optional[str] | 用于登录Modelers Hub的验证token | None |
| print_param_status | bool | 是否打印模型参数的状态 | False |
| trust_remote_code | bool | 是否信任来自Hub上数据集/模型的代码执行 | False |
| compute_dtype | Optional[torch.dtype] | 用于计算模型输出的数据类型，无需手动指定 | None |
| device_map | Optional[Union[str, Dict[str, Any]]] | 模型分配的设备映射，无需手动指定 | None |
| model_max_length | Optional[int] | 模型的最大输入长度，无需手动指定 | None |
| block_diag_attn | bool | 是否使用块对角注意力，无需手动指定 | False |

## 十、生成参数（GeneratingArguments）
| 参数名称 | 类型 | 介绍 | 默认值 |
| --- | --- | --- | --- |
| do_sample | bool | 是否使用采样策略生成文本，如果设置为False，将使用greedy decoding | True |
| temperature | float | 用于调整生成文本的随机性，temperature越高，生成的文本越随机；temperature越低，生成的文本越确定 | 0.95 |
| top_p | float | 用于控制生成时候选token集合大小的参数，例如：top_p=0.7意味着模型会先选择概率最高的若干个token直到其累积概率之和大于0.7，然后在这些token组成的集合中进行采样 | 0.7 |
| top_k | int | 用于控制生成时候选token集合大小的参数，例如：top_k=50意味着模型会在概率最高的50个token组成的集合中进行采样 | 50 |
| num_beams | int | 用于beam_search的束宽度，值为1表示不使用beam_search | 1 |
| max_length | int | 文本最大长度（包括输入文本和生成文本的长度） | 1024 |
| max_new_tokens | int | 生成文本的最大长度，设置max_new_tokens会覆盖max_length | 1024 |
| repetition_penalty | float | 对生成重复token的惩罚系数，对于已经生成过的token生成概率乘以1/repetition_penalty；值小于1.0会提高重复token的生成概率，大于1.0则会降低重复token的生成概率 | 1.0 |
| length_penalty | float | 在使用beam_search时对生成文本长度的惩罚系数，length_penalty>0鼓励模型生成更长的序列，length_penalty<0会鼓励模型生成更短的序列 | 1.0 |
| default_system | str | 默认的system_message，例如："You are a helpful assistant." | None |
| skip_special_tokens | bool | 解码时是否忽略特殊token | True |

## 十一、评估参数（EvalArguments）
| 参数名称 | 类型 | 介绍 | 默认值 |
| --- | --- | --- | --- |
| task | str | 评估任务的名称，可选项有mmlu_test、eval_validation、cmmlu_test | None |
| task_dir | str | 包含评估数据集的文件夹路径 | evaluation |
| batch_size | int | 每个GPU使用的批量大小 | 4 |
| seed | int | 用于数据加载器的随机种子 | 42 |
| lang | str | 评估使用的语言，可选值为en、zh | en |
| n_shot | int | few-shot的示例数量 | 5 |
| save_dir | str | 保存评估结果的路径，如果该路径已经存在则会抛出错误 | None |
| download_mode | str | 评估数据集的下载模式，如果数据集已经存在则重复使用，否则则下载 | DownloadMode.REUSE_DATASET_IF_EXISTS |

## 十二、模型量化参数（QuantizationArguments）
| 参数名称 | 类型 | 介绍 | 默认值 |
| --- | --- | --- | --- |
| quantization_method | Literal["bitsandbytes", "hqq", "eetq"] | 指定用于量化的算法 | bitsandbytes |
| quantization_bit | Optional[int] | 指定在量化过程中使用的位数，通常是4位、8位等 | None |
| quantization_type | Literal["fp4", "nf4"] | 量化时使用的数据类型 | nf4 |
| double_quantization | bool | 是否在量化过程中使用double quantization，通常用于"bitsandbytes" int4量化训练 | True |
| quantization_device_map | Optional[Literal["auto"]] | 用于推理4-bit量化模型的设备映射，需要"bitsandbytes >= 0.43.0" | None |

## 十三、模型导出参数（ExportArguments）
| 参数名称 | 类型 | 介绍 | 默认值 |
| --- | --- | --- | --- |
| export_dir | Optional[str] | 导出模型保存目录的路径 | None |
| export_size | int | 导出模型的文件分片大小（以GB为单位） | 5 |
| export_device | Literal["cpu", "auto"] | 导出模型时使用的设备，auto可自动加速导出 | cpu |
| export_quantization_bit | Optional[int] | 量化导出模型时使用的位数 | None |
| export_quantization_dataset | Optional[str] | 用于量化导出模型的数据集路径或数据集名称 | None |
| export_quantization_nsamples | int | 量化时使用的样本数量 | 128 |
| export_quantization_maxlen | int | 用于量化的模型输入的最大长度 | 1024 |
| export_legacy_format | bool | True: .bin格式保存；False: .safetensors格式保存 | False |
| export_hub_model_id | Optional[str] | 模型上传至Huggingface的仓库名称 | None |

## 十四、环境变量（Environment Variables）
| 名称 | 类型 | 介绍 |
| --- | --- | --- |
| API_HOST | API | API服务器监听的主机地址 |
| API_PORT | API | API服务器监听的端口号 |
| API_KEY | API | 访问API的密码 |
| API_MODEL_NAME | API | 指定API服务要加载和使用的模型名称 |
| API_VERBOSE | API | 控制API日志的详细程度 |
| FASTAPI_ROOT_PATH | API | 设置FastAPI应用的根路径 |
| MAX_CONCURRENT | API | API的最大并发请求数 |
| DISABLE_VERSION_CHECK | General | 是否禁用启动时的版本检查 |
| FORCE_CHECK_IMPORTS | General | 强制检查可选的导入 |
| ALLOW_EXTRA_ARGS | General | 允许在命令行中传递额外参数 |
| LLAMAFACTORY_VERBOSITY | General | 设置LLaMA-Factory的日志级别（"DEBUG","INFO","WARN"） |
| USE_MODELSCOPE_HUB | General | 优先使用ModelScope下载模型/数据集或使用缓存路径中的模型/数据集 |
| USE_OPENMIND_HUB | General | 优先使用Openmind下载模型/数据集或使用缓存路径中的模型/数据集 |
| USE_RAY | General | 是否使用Ray进行分布式执行或任务管理 |
| RECORD_VRAM | General | 是否记录VRAM使用情况 |
| OPTIM_TORCH | General | 是否表示启用特定的PyTorch优化 |
| NPU_JIT_COMPILE | General | 是否为NPU启用JIT编译 |
| CUDA_VISIBLE_DEVICES | General | GPU选择 |
| ASCEND_RT_VISIBLE_DEVICES | General | NPU选择 |
| FORCE_TORCHRUN | Torchrun | 是否强制使用torchrun启动脚本 |
| MASTER_ADDR | Torchrun | Torchrun部署中主节点（master node）的网络地址 |
| MASTER_PORT | Torchrun | Torchrun部署中主节点用于通信的端口号 |
| NNODES | Torchrun | 参与分布式部署的总节点数量 |
| NODE_RANK | Torchrun | 当前节点在所有节点中的rank，通常从0到NNODES-1 |
| NPROC_PER_NODE | Torchrun | 每个节点上的GPU数 |
| WANDB_DISABLED | Log | 是否禁用wandb |
| WANDB_PROJECT | Log | 设置wandb中的项目名称 |
| WANDB_API_KEY | Log | 访问wandb的api key |
| GRADIO_SHARE | Web UI | 是否创建一个可共享的webui链接 |
| GRADIO_SERVER_NAME | Web UI | 设置Gradio服务器IP地址（例如0.0.0.0） |
| GRADIO_SERVER_PORT | Web UI | 设置Gradio服务器的端口 |
| GRADIO_ROOT_PATH | Web UI | 设置Gradio应用的根路径 |
| GRADIO_IPV6 | Web UI | 启用Gradio服务器的IPv6支持 |
| ENABLE_SHORT_CONSOLE | Setting | 支持使用llmf表示llamafactory-cli |

## 十五、多模态模型参数（ProcessorArguments）
| 参数名称 | 类型 | 介绍 | 默认值 |
| --- | --- | --- | --- |
| image_max_pixels | int | 图像输入的最大像素数 | 768x768 |
| image_min_pixels | int | 图像输入的最小像素数 | 32x32 |
| video_max_pixels | int | 视频输入的最大像素数 | 256x256 |
| video_min_pixels | int | 视频输入的最小像素数 | 16x16 |
| video_fps | float | 视频输入的采样帧率（每秒采样帧数） | 2.0 |
| video_maxlen | int | 视频输入的最大采样帧数 | 128 |