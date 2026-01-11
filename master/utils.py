import os
import json
from copy import deepcopy
from typing import Any, Dict

import yaml
from sqlmodel import Session, select

from master.database import engine
from master.models import ModelRegistry


TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
TRAIN_TEMPLATE_PATH = os.path.join(TEMPLATE_DIR, "train_base.yaml")
EVAL_TEMPLATE_PATH = os.path.join(TEMPLATE_DIR, "eval_base.yaml")
EXPORT_TEMPLATE_PATH = os.path.join(TEMPLATE_DIR, "export_base.yaml")
INFER_TEMPLATE_PATH = os.path.join(TEMPLATE_DIR, "inference_base.yaml")

# 中文注释：宿主机数据根目录，用于将容器内 /app 路径映射到宿主机实际路径
HOST_BASE_PATH = os.getenv("HOST_BASE_PATH", "/home/ubuntu/CloudComputing/cloud-llm")


def _load_template(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_merge(deepcopy(base[k]), v)
        else:
            base[k] = v
    return base


def _dump_yaml(cfg: Dict[str, Any]) -> str:
    return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False)


def resolve_model_alias(model_name: str) -> str:
    """将展示名称映射到已配置的 HF 路径（若存在）。"""
    try:
        with Session(engine) as session:
            record = session.exec(select(ModelRegistry).where(ModelRegistry.name == model_name)).first()
            if record:
                return record.hf_path or record.path or model_name
    except Exception:
        pass
    return model_name

def _extract_overrides(params: Dict[str, Any]) -> Dict[str, Any]:
    overrides = params.pop("config_overrides", {}) or {}
    if not isinstance(overrides, dict):
        return {}
    return overrides


def generate_config(task_id: str, params_json: str) -> str:
    """生成训练用 YAML：加载全量模板，合并用户覆写。"""
    try:
        params = json.loads(params_json)
        overrides = _extract_overrides(params)
        base = _load_template(TRAIN_TEMPLATE_PATH)
        # 兼容旧字段
        simple_overrides = {
            "model_name_or_path": resolve_model_alias(params.get("model_name") or base.get("model_name_or_path")),
            "dataset": params.get("dataset", base.get("dataset")),
            "template": params.get("template", base.get("template")),
            "num_train_epochs": params.get("epochs", base.get("num_train_epochs")),
            "per_device_train_batch_size": params.get("batch_size", base.get("per_device_train_batch_size")),
            "learning_rate": params.get("learning_rate", base.get("learning_rate")),
        }
        base["output_dir"] = f"/app/output/{task_id}"
        merged = _deep_merge(base, simple_overrides)
        merged = _deep_merge(merged, overrides)
        return _dump_yaml(merged)
    except Exception as e:
        raise ValueError(f"Failed to generate config: {e}")


def generate_eval_config(eval_id: str, params_json: str) -> str:
    """生成评估用 YAML，支持 LoRA 适配器与用户覆写。"""
    try:
        params = json.loads(params_json)
        overrides = _extract_overrides(params)
        base = _load_template(EVAL_TEMPLATE_PATH)

        model_name = resolve_model_alias(params.get("model_name") or base.get("model_name_or_path"))
        base_updates = {
            "model_name_or_path": model_name,
            "dataset": params.get("dataset", base.get("dataset")),
            "eval_dataset": params.get("dataset", base.get("eval_dataset")),
            "template": params.get("template", base.get("template")),
            "per_device_eval_batch_size": params.get("batch_size", base.get("per_device_eval_batch_size")),
            "max_samples": params.get("max_samples", base.get("max_samples")),
            "seed": params.get("seed", base.get("seed")),
        }

        model_path = params.get("model_path")
        if model_path:
            base_updates["model_name_or_path"] = model_path
            host_path = model_path
            # 将容器内路径 /app/... 映射到宿主机实际目录，便于探测 LoRA 适配器
            if model_path.startswith("/app/"):
                host_path = os.path.join(HOST_BASE_PATH, model_path[len("/app/"):].lstrip("/"))

            adapter_cfg_path = os.path.join(host_path, "adapter_config.json")
            if os.path.isfile(adapter_cfg_path):
                base_updates["adapter_name_or_path"] = model_path
                base_updates.setdefault("finetuning_type", "lora")
                # 中文注释：若适配器内记录了底模，优先用底模路径，避免把适配器当成基础模型
                try:
                    with open(adapter_cfg_path, "r", encoding="utf-8") as f:
                        adapter_cfg = json.load(f)
                    base_model_from_adapter = adapter_cfg.get("base_model_name_or_path")
                    if base_model_from_adapter:
                        base_updates["model_name_or_path"] = resolve_model_alias(base_model_from_adapter)
                except Exception:
                    pass

        base["output_dir"] = f"/app/output/{eval_id}/eval"
        merged = _deep_merge(base, base_updates)
        merged = _deep_merge(merged, overrides)
        return _dump_yaml(merged)
    except Exception as e:
        raise ValueError(f"Failed to generate eval config: {e}")


def generate_export_config(export_id: str, params_json: str) -> str:
    """生成导出/合并/量化 YAML。"""
    try:
        params = json.loads(params_json)
        overrides = _extract_overrides(params)
        base = _load_template(EXPORT_TEMPLATE_PATH)

        base_updates = {
            "model_name_or_path": resolve_model_alias(params.get("model_name") or base.get("model_name_or_path")),
            "adapter_name_or_path": params.get("adapter_path") or base.get("adapter_name_or_path"),
            "template": params.get("template", base.get("template")),
            "quantization_method": params.get("quantization_method", base.get("quantization_method")),
            "quantization_type": params.get("quantization_type", base.get("quantization_type")),
            "double_quantization": params.get("double_quantization", base.get("double_quantization")),
            "quantization_device_map": params.get("quantization_device_map", base.get("quantization_device_map")),
            "export_quantization_bit": params.get("quantization_bit", base.get("export_quantization_bit")),
            "export_quantization_dataset": params.get("quantization_dataset", base.get("export_quantization_dataset")),
        }
        method = (base_updates.get("quantization_method") or "").lower()
        if method != "gptq":
            # 非 GPTQ 时移除 export 级量化字段，避免触发 HF GPTQ 路径
            base_updates["export_quantization_bit"] = None
            base_updates["export_quantization_dataset"] = None
        elif base_updates.get("export_quantization_bit") and not base_updates.get("export_quantization_dataset"):
            base_updates["export_quantization_dataset"] = "/app/data/c4_demo.jsonl"

        base["export_dir"] = f"/app/output/{export_id}/export"
        merged = _deep_merge(base, base_updates)
        merged = _deep_merge(merged, overrides)
        return _dump_yaml(merged)
    except Exception as e:
        raise ValueError(f"Failed to generate export config: {e}")


def generate_infer_config(deploy_id: str, params_json: str) -> str:
    """生成推理/部署 YAML，合并用户覆写。"""
    try:
        params = json.loads(params_json)
        overrides = _extract_overrides(params)
        base = _load_template(INFER_TEMPLATE_PATH)
        base_updates = {
            "model_name_or_path": resolve_model_alias(params.get("model_name") or base.get("model_name_or_path")),
            "adapter_name_or_path": params.get("adapter_path") or base.get("adapter_name_or_path"),
            "template": params.get("template", base.get("template")),
            "infer_backend": params.get("infer_backend", base.get("infer_backend")),
        }

        # 优先使用显式传入的模型路径（可指向训练/导出产物或原始基座）
        model_path = params.get("model_path")
        if model_path:
            base_updates["model_name_or_path"] = model_path
            host_path = model_path
            if model_path.startswith("/app/"):
                host_path = os.path.join(HOST_BASE_PATH, model_path[len("/app/"):].lstrip("/"))
            # 若模型目录里包含 adapter_config.json，则判定为 LoRA 适配器
            adapter_cfg_path = os.path.join(host_path, "adapter_config.json")
            if os.path.isfile(adapter_cfg_path) and not base_updates.get("adapter_name_or_path"):
                base_updates["adapter_name_or_path"] = model_path
                # 若适配器记录了底模，优先用底模路径，避免把 LoRA 当成基座
                try:
                    with open(adapter_cfg_path, "r", encoding="utf-8") as f:
                        adapter_cfg = json.load(f)
                    base_model_from_adapter = adapter_cfg.get("base_model_name_or_path")
                    if base_model_from_adapter:
                        base_updates["model_name_or_path"] = resolve_model_alias(base_model_from_adapter)
                except Exception:
                    pass

        # 根据是否存在适配器，修正 finetuning_type，避免基座模型被误当成 LoRA
        if base_updates.get("adapter_name_or_path"):
            base_updates.setdefault("finetuning_type", "lora")
        else:
            base_updates["adapter_name_or_path"] = None
            base_updates["finetuning_type"] = "full"

        # 生成参数（在线推理时可动态覆写，但先写入默认）
        gen_keys = [
            "do_sample", "temperature", "top_p", "top_k", "num_beams",
            "max_length", "max_new_tokens", "repetition_penalty", "length_penalty",
            "default_system", "skip_special_tokens",
        ]
        for k in gen_keys:
            if k in params:
                base_updates[k] = params[k]

        merged = _deep_merge(base, base_updates)
        merged = _deep_merge(merged, overrides)
        return _dump_yaml(merged)
    except Exception as e:
        raise ValueError(f"Failed to generate infer config: {e}")
