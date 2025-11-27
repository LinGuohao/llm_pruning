import torch
import numpy as np
from transformers import LlamaForCausalLM, AutoTokenizer
import datasets
from typing import Dict, List
import json
import gc


def cleanup_memory():
    """清理GPU内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    return encodings["input_ids"], encodings.get("attention_mask")


@torch.no_grad()
def pca_calc(
    X_list: List[torch.Tensor],
    ignore_masks: List[torch.Tensor] = None,
    device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    计算PCA获得旋转矩阵Q
    参考SliceGPT的pca_calc函数

    Args:
        X_list: 输入张量列表，每个是 [batch_size, seq_len, hidden_size]
        ignore_masks: attention mask列表，每个是 [batch_size, seq_len]
        device: 计算设备

    Returns:
        eigenvalues, eigenvectors (Q矩阵)
    """
    cleanup_memory()

    H = None
    for idx, X_batch in enumerate(X_list):
        # X_batch shape: [batch_size, seq_len, hidden_size]
        if ignore_masks is not None and len(ignore_masks) > idx:
            # 直接使用mask索引，PyTorch会自动广播
            # ignore_masks[idx] == 0: [batch_size, seq_len]
            # 这会将对应位置的整个hidden_size维度设为0
            X_batch = X_batch.clone()
            X_batch[ignore_masks[idx] == 0] = 0

        X_batch = X_batch.double().to(device=device)
        # X_batch.mT: [batch_size, hidden_size, seq_len]
        # X_batch.mT @ X_batch: [batch_size, hidden_size, hidden_size]
        # sum(dim=0): 对batch维度求和 -> [hidden_size, hidden_size]
        H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)
        H = H_batch if H is None else H + H_batch

    # 添加阻尼以提高数值稳定性
    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1]).to(device=device)
    H[diag, diag] = H[diag, diag] + damp

    # 特征值分解
    X_eig = torch.linalg.eigh(H)
    del H
    cleanup_memory()

    # 按特征值降序排序
    index = torch.argsort(X_eig[0], descending=True)
    eig_val = X_eig[0][index]
    eigen_vec = X_eig[1][:, index]

    return eig_val, eigen_vec


@torch.no_grad()
def get_layer0_inputs(model, tokenizer, input_ids, attention_mask=None):
    """
    获取第0层（第一个transformer层）的输入
    参考SliceGPT的get_layer0_inputs
    """
    # 临时替换第0层为捕获器
    class Catcher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input = None
            self.args = None
            self.kwargs = None

        def forward(self, hidden_states, *args, **kwargs):
            self.input = hidden_states
            self.args = args
            self.kwargs = kwargs
            raise ValueError  # 停止前向传播

    original_layer = model.model.layers[0]
    catcher = Catcher()
    model.model.layers[0] = catcher

    try:
        batch = {"input_ids": input_ids}
        if attention_mask is not None:
            batch["attention_mask"] = attention_mask
        model(**batch)
    except ValueError:
        pass

    # 获取捕获的输入
    layer0_input = catcher.input.cpu()

    # 恢复原始层
    model.model.layers[0] = original_layer

    return layer0_input


@torch.no_grad()
def get_layer_outputs(layer, hidden_states, attention_mask=None, position_ids=None):
    """
    获取单层的输出，包括中间的MLP输入
    参考SliceGPT的get_signals
    """
    mlp_ln_input = None

    def hook_fn(module, args, output):
        nonlocal mlp_ln_input
        mlp_ln_input = args[0].cpu()  # post_attention_layernorm的输入

    # 注册hook到post_attention_layernorm
    hook = layer.post_attention_layernorm.register_forward_hook(hook_fn)

    # 前向传播
    output = layer(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids
    )

    hook.remove()

    if isinstance(output, tuple):
        output = output[0]

    return mlp_ln_input, output


@torch.no_grad()
def compute_layer_pca_rotations(
    model_path: str,
    nsamples: int = 128,
    seqlen: int = 2048,
    dataset_name: str = "wikitext",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_memory: bool = True
) -> Dict[str, np.ndarray]:
    """
    逐层计算PCA旋转矩阵Q，节省显存
    参考SliceGPT的rotate_and_slice_sequential函数

    Args:
        model_path: 模型路径
        nsamples: 样本数量
        seqlen: 序列长度
        dataset_name: 数据集名称
        device: 设备
        save_memory: 是否使用节省显存模式（逐层处理）

    Returns:
        包含每层Q矩阵和特征值的字典
    """
    print(f"Loading model from {model_path}...")

    # 加载模型配置和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 准备数据
    print(f"Preparing {nsamples} samples from {dataset_name}...")
    input_ids, attention_mask = prepare_calibration_data(dataset_name, tokenizer, nsamples, seqlen)

    results = {}

    # 分批加载，只在需要时加载模型
    if save_memory:
        # 第一步：获取embedding输出（第0层输入）
        print("Computing embedding outputs (layer 0 inputs)...")
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            local_files_only=True
        )
        model.eval()

        # 收集所有样本的layer0输入（每个样本单独一个batch）
        layer0_inputs = []
        masks = []

        for i in range(len(input_ids)):
            batch_ids = input_ids[i:i+1].to(device)  # [1, seq_len]
            batch_mask = attention_mask[i:i+1] if attention_mask is not None else None

            # 移动embedding到GPU
            model.model.embed_tokens.to(device)
            layer0_input = get_layer0_inputs(model, tokenizer, batch_ids, batch_mask)
            layer0_inputs.append(layer0_input)  # [1, seq_len, hidden]
            if batch_mask is not None:
                masks.append(batch_mask.cpu())

            # 移回CPU
            model.model.embed_tokens.to('cpu')
            cleanup_memory()

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(input_ids)} samples")

        # 计算embedding输出的PCA
        print("Computing PCA for embedding outputs...")
        eig_val, Q = pca_calc(layer0_inputs, masks if masks else None, device)
        results["embedding_eigenvalues"] = eig_val.cpu().numpy()
        results["embedding_Q"] = Q.cpu().numpy()
        print(f"  Embedding Q shape: {Q.shape}")

        # 逐层处理
        num_layers = len(model.model.layers)

        for layer_idx in range(num_layers):
            print(f"\nProcessing layer {layer_idx}/{num_layers-1}...")

            # 将当前层移到GPU
            layer = model.model.layers[layer_idx].to(device)

            # 更新输入（应用上一层的Q矩阵旋转）
            rotated_inputs = []
            mlp_ln_inputs = []
            layer_outputs = []

            Q_device = Q.to(device).to(torch.float16)

            for i in range(len(layer0_inputs)):
                # 1. 准备输入数据
                inp = layer0_inputs[i].to(device)
                rotated_inp = torch.matmul(inp, Q_device)
                
                batch_size, seq_len, _ = rotated_inp.shape
                dtype = rotated_inp.dtype

                # =================== 修复代码开始 ===================
                
                # A. 构建 position_ids (解决上一个报错)
                position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

                # B. 手动构建 4D Causal Mask (解决当前报错)
                # 目标形状: [Batch, 1, Seq, Seq]
                # 逻辑: Causal(下三角) + Padding(填充部分)
                
                # B1. 创建因果遮罩 (下三角为0，其余为负无穷)
                # Llama 默认使用 float('-inf') 来代表 mask，但为了数值稳定建议用 min
                min_dtype = torch.finfo(dtype).min
                
                # 创建一个 [Seq, Seq] 的全负无穷矩阵
                causal_mask = torch.full((seq_len, seq_len), min_dtype, device=device, dtype=dtype)
                # 将对角线及以下（下三角）填为 0.0 (代表可见)
                causal_mask = torch.triu(causal_mask, diagonal=1)
                # 扩展维度 -> [1, 1, Seq, Seq]
                causal_mask = causal_mask[None, None, :, :]

                # B2. 结合 Padding Mask (如果有)
                batch_mask_raw = masks[i].to(device) if masks else None
                if batch_mask_raw is not None:
                    # batch_mask_raw 是 [Batch, Seq], 1=Keep, 0=Pad
                    # 我们需要把 0 变成负无穷
                    padding_mask = torch.zeros((batch_size, seq_len), device=device, dtype=dtype)
                    padding_mask = padding_mask.masked_fill(batch_mask_raw == 0, min_dtype)
                    # 扩展维度 -> [Batch, 1, 1, Seq]
                    padding_mask = padding_mask[:, None, None, :]
                    
                    # 广播相加: Causal + Padding
                    # 注意：负无穷 + 0 = 负无穷；0 + 0 = 0；负无穷 + 负无穷 = 负无穷
                    attention_mask = causal_mask + padding_mask
                else:
                    attention_mask = causal_mask.expand(batch_size, 1, -1, -1)

                # =================== 修复代码结束 ===================

                # 3. 调用层
                mlp_ln_inp, out = get_layer_outputs(
                    layer, 
                    rotated_inp, 
                    attention_mask=attention_mask, # 传入处理好的 4D Mask
                    position_ids=position_ids      # 传入 position_ids
                )

                mlp_ln_inputs.append(mlp_ln_inp)
                layer_outputs.append(out.cpu())

                cleanup_memory()

            # 计算MLP输入的PCA（attention输出）
            print(f"  Computing PCA for layer {layer_idx} MLP inputs...")
            eig_val_mlp, Q_mlp = pca_calc(mlp_ln_inputs, masks if masks else None, device)
            results[f"layer_{layer_idx}_mlp_input_eigenvalues"] = eig_val_mlp.cpu().numpy()
            results[f"layer_{layer_idx}_mlp_input_Q"] = Q_mlp.cpu().numpy()
            print(f"    MLP input Q shape: {Q_mlp.shape}")

            # 计算层输出的PCA
            print(f"  Computing PCA for layer {layer_idx} outputs...")
            eig_val_out, Q_out = pca_calc(layer_outputs, masks if masks else None, device)
            results[f"layer_{layer_idx}_output_eigenvalues"] = eig_val_out.cpu().numpy()
            results[f"layer_{layer_idx}_output_Q"] = Q_out.cpu().numpy()
            print(f"    Output Q shape: {Q_out.shape}")

            # 更新下一层的输入
            layer0_inputs = layer_outputs
            Q = Q_out

            # 将层移回CPU以节省显存
            layer.to('cpu')
            cleanup_memory()

        # 清理模型
        del model
        cleanup_memory()

    else:
        # 不节省显存的版本：一次性加载整个模型
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            local_files_only=True
        )
        model.eval()

        # 后续实现类似...
        raise NotImplementedError("Non-memory-saving mode not fully implemented")

    return results


def save_rotation_matrices(results: Dict[str, np.ndarray], save_path: str):
    """保存旋转矩阵"""
    # 保存为npz格式
    np.savez_compressed(save_path + ".npz", **results)
    print(f"Saved rotation matrices to {save_path}.npz")

    # 保存统计信息
    stats = {}
    for name, values in results.items():
        if "eigenvalues" in name:
            stats[name] = {
                "shape": list(values.shape),
                "top_10": values[:10].tolist(),
                "sum": float(values.sum()),
                "mean": float(values.mean())
            }
        elif "_Q" in name:
            stats[name] = {
                "shape": list(values.shape),
                "dtype": str(values.dtype)
            }

    with open(save_path + "_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {save_path}_stats.json")


if __name__ == "__main__":
    model_path = "/gpfs/volcano/models/meta-llama/Llama-2-13b-hf"

    results = compute_layer_pca_rotations(
        model_path=model_path,
        nsamples=128,
        seqlen=2048,
        dataset_name="wikitext",
        save_memory=True
    )

    save_rotation_matrices(results, "pca_rotation_matrices")
