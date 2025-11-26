import torch
import numpy as np
from transformers import LlamaForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import datasets
from typing import Dict, List
import json


def prepare_calibration_data(
    dataset_name: str = "wikitext",
    tokenizer=None,
    nsamples: int = 128,
    seqlen: int = 2048,
    seed: int = 42
):
    """准备校准数据"""
    if dataset_name == "wikitext":
        data = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text_column = "text"
    elif dataset_name == "c4":
        data = datasets.load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train"
        )
        text_column = "text"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    data = data.filter(lambda x: len(x[text_column]) > 0)

    # 构建固定长度样本
    data_list = data[text_column]
    samples = []

    torch.manual_seed(seed)
    indices = list(range(len(data_list)))

    while len(samples) < nsamples and len(indices) > 0:
        start_idx = torch.randint(0, len(indices), (1,)).item()
        idx = start_idx
        tokens = []
        while len(tokens) < seqlen and idx < len(indices):
            item = data_list[indices[idx]]
            sep = "" if not tokens else "\n\n"
            tokens += tokenizer.tokenize(sep + item)
            idx += 1

        indices = indices[:start_idx] + indices[idx:]

        if len(tokens) >= seqlen:
            tokens = tokens[:seqlen]
            samples.append(tokenizer.convert_tokens_to_string(tokens))

    # Tokenize
    encodings = tokenizer(samples, padding=True, truncation=True, max_length=seqlen, return_tensors="pt")
    return encodings["input_ids"]


@torch.no_grad()
def compute_rmsnorm_outputs(
    model_path: str,
    nsamples: int = 128,
    seqlen: int = 2048,
    dataset_name: str = "wikitext",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, np.ndarray]:
    """
    计算Llama模型每层RMSnorm的输出矩阵（样本平均）

    Args:
        model_path: 模型路径
        nsamples: 样本数量
        seqlen: 序列长度
        dataset_name: 数据集名称
        device: 设备

    Returns:
        包含每层RMSnorm输出的字典
    """
    # 加载模型和tokenizer
    print(f"Loading model from {model_path}...")
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # 准备数据
    print(f"Preparing {nsamples} samples from {dataset_name}...")
    input_ids = prepare_calibration_data(dataset_name, tokenizer, nsamples, seqlen)

    # 存储每层的RMSnorm输出
    num_layers = len(model.model.layers)
    rmsnorm_outputs = {
        f"layer_{i}_input_layernorm": [] for i in range(num_layers)
    }
    rmsnorm_outputs.update({
        f"layer_{i}_post_attention_layernorm": [] for i in range(num_layers)
    })
    rmsnorm_outputs["final_layernorm"] = []

    # 注册hooks
    handles = []

    def create_hook(name):
        def hook(module, input, output):
            # output: [batch_size, seq_len, hidden_size]
            rmsnorm_outputs[name].append(output.detach().cpu().float())
        return hook

    # 注册每层的hooks
    for i, layer in enumerate(model.model.layers):
        h1 = layer.input_layernorm.register_forward_hook(
            create_hook(f"layer_{i}_input_layernorm")
        )
        h2 = layer.post_attention_layernorm.register_forward_hook(
            create_hook(f"layer_{i}_post_attention_layernorm")
        )
        handles.extend([h1, h2])

    # 最后的norm
    final_handle = model.model.norm.register_forward_hook(
        create_hook("final_layernorm")
    )
    handles.append(final_handle)

    # 前向传播
    print("Running forward pass...")
    batch_size = 1
    for i in range(0, len(input_ids), batch_size):
        batch = input_ids[i:i+batch_size].to(device)
        model(batch)
        if (i + batch_size) % 10 == 0:
            print(f"Processed {i + batch_size}/{len(input_ids)} samples")

    # 移除hooks
    for handle in handles:
        handle.remove()

    # 计算平均值
    print("Computing averages...")
    averaged_outputs = {}
    for name, outputs in rmsnorm_outputs.items():
        # outputs: list of [batch_size, seq_len, hidden_size]
        stacked = torch.cat(outputs, dim=0)  # [total_samples, seq_len, hidden_size]
        # 平均：先在seq_len维度平均，再在样本维度平均
        avg = stacked.mean(dim=(0, 1)).numpy()  # [hidden_size]
        averaged_outputs[name] = avg

    return averaged_outputs


def save_outputs(outputs: Dict[str, np.ndarray], save_path: str):
    """保存输出到文件"""
    # 保存为npz格式（二进制但可以用numpy读取）
    np.savez(save_path + ".npz", **outputs)
    print(f"Saved binary format to {save_path}.npz")

    # 保存为JSON格式（人类可读）
    json_outputs = {k: v.tolist() for k, v in outputs.items()}
    with open(save_path + ".json", "w") as f:
        json.dump(json_outputs, f, indent=2)
    print(f"Saved human-readable format to {save_path}.json")

    # 保存统计信息（更易读）
    stats = {}
    for name, values in outputs.items():
        stats[name] = {
            "shape": list(values.shape),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "first_10_values": values[:10].tolist()
        }

    with open(save_path + "_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {save_path}_stats.json")


if __name__ == "__main__":
    # 使用示例
    model_path = "/gpfs/volcano/models/meta-llama/Llama-2-13b-hf"  # 替换为你的模型路径

    outputs = compute_rmsnorm_outputs(
        model_path=model_path,
        nsamples=128,
        seqlen=2048,
        dataset_name="wikitext"
    )

    save_outputs(outputs, "rmsnorm_outputs")
