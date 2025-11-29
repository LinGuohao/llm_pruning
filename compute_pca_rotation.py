import torch
import numpy as np
from transformers import LlamaForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
from typing import Dict, List, Tuple
import gc


def cleanup_memory():
    """清理显存"""
    gc.collect()
    torch.cuda.empty_cache()


def prepare_calibration_data(dataset_name: str, tokenizer, nsamples: int, seqlen: int):
    """准备校准数据"""
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(dataset["text"])
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")

    # Tokenize
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids

    # 切分成多个样本
    samples = []
    attention_masks = []
    for i in range(nsamples):
        start = i * seqlen
        end = start + seqlen
        if end > input_ids.shape[1]:
            break
        sample = input_ids[:, start:end]
        samples.append(sample)
        attention_masks.append(torch.ones_like(sample))

    input_ids = torch.cat(samples, dim=0)  # [nsamples, seqlen]
    attention_mask = torch.cat(attention_masks, dim=0)  # [nsamples, seqlen]

    return input_ids, attention_mask


def pca_calc(X_list: List[torch.Tensor], ignore_masks=None, device: str = "cuda"):
    """
    计算PCA，参考SliceGPT的实现
    X_list: 每个元素是 [batch_size, seq_len, hidden_size]
    """
    H = None
    nsamples = 0

    for idx, X_batch in enumerate(X_list):
        # X_batch: [batch_size, seq_len, hidden_size]
        if ignore_masks is not None and ignore_masks[idx] is not None:
            # 将padding位置置零
            X_batch = X_batch.clone()
            mask = ignore_masks[idx].unsqueeze(-1)  # [batch_size, seq_len, 1]
            X_batch = X_batch * mask.to(X_batch.device)

        X_batch = X_batch.double().to(device=device)

        # 计算协方差矩阵的累积
        # H = Σ(X.mT @ X) across all tokens
        H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)  # [hidden, hidden]

        if H is None:
            H = H_batch
        else:
            H = H + H_batch

        nsamples += X_batch.shape[0] * X_batch.shape[1]

    # 添加damping
    diag = torch.arange(H.shape[0], device=H.device)
    damp = 0.01 * torch.mean(torch.diag(H))
    H[diag, diag] = H[diag, diag] + damp

    # 特征分解
    X_eig = torch.linalg.eigh(H)
    eig_val = X_eig.eigenvalues
    eigen_vec = X_eig.eigenvectors

    # 按特征值从大到小排序
    indices = torch.argsort(eig_val, descending=True)
    eig_val = eig_val[indices]
    eigen_vec = eigen_vec[:, indices]

    return eig_val.cpu(), eigen_vec.cpu()


@torch.no_grad()
def get_layer0_inputs(model, tokenizer, input_ids, attention_mask=None):
    """
    获取第0层的输入（embedding输出）
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    # 只需要embedding的输出
    inputs_embeds = model.model.embed_tokens(input_ids)

    return inputs_embeds.cpu()


@torch.no_grad()
def get_layer_outputs(layer, hidden_states, attention_mask=None, position_ids=None):
    """
    获取单层的输出，包括中间的MLP输入（post_attention_layernorm的输出）
    参考SliceGPT的get_signals
    """
    mlp_ln_output = None

    def hook_fn(module, args, output):
        nonlocal mlp_ln_output
        # 使用 layernorm 的输出而不是输入
        mlp_ln_output = output.cpu()

    # 注册hook到post_attention_layernorm
    hook = layer.post_attention_layernorm.register_forward_hook(hook_fn)

    # 生成 position_ids
    if position_ids is None:
        batch_size, seq_len, _ = hidden_states.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

    # 生成 causal attention mask (4D)
    if attention_mask is not None:
        # attention_mask: [batch_size, seq_len], 1表示有效token，0表示padding
        batch_size, seq_len, _ = hidden_states.shape
        # 创建 causal mask: [seq_len, seq_len]
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=hidden_states.device))
        # 扩展为 [batch_size, 1, seq_len, seq_len]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        # 应用 padding mask
        padding_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(hidden_states.device).to(torch.bool)
        causal_mask = causal_mask & padding_mask
        # 转换为 attention mask 格式 (0表示attend, -inf表示mask)
        attention_mask_4d = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=hidden_states.dtype, device=hidden_states.device)
        attention_mask_4d.masked_fill_(~causal_mask, float('-inf'))
    else:
        attention_mask_4d = None

    # 前向传播
    output = layer(
        hidden_states,
        attention_mask=attention_mask_4d,
        position_ids=position_ids
    )

    hook.remove()

    if isinstance(output, tuple):
        layer_output = output[0]
    else:
        layer_output = output

    return mlp_ln_output, layer_output


def rotate_attention_inputs(layer, Q: torch.Tensor, device: str = "cuda"):
    """旋转attention的QKV权重"""
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(dtype=dtype)  # 保持在GPU上


def rotate_attention_output(layer, Q: torch.Tensor, device: str = "cuda"):
    """旋转attention的输出权重"""
    W = layer.self_attn.o_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)  # 保持在GPU上
    if W.bias is not None:
        b = W.bias.data.to(device=device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(dtype=dtype)  # 保持在GPU上


def rotate_mlp_input(layer, Q: torch.Tensor, device: str = "cuda"):
    """旋转MLP的输入权重（gate_proj和up_proj）"""
    for W in [layer.mlp.gate_proj, layer.mlp.up_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(dtype=dtype)  # 保持在GPU上


def rotate_mlp_output(layer, Q: torch.Tensor, device: str = "cuda"):
    """旋转MLP的输出权重（down_proj）"""
    W = layer.mlp.down_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)  # 保持在GPU上
    if W.bias is not None:
        b = W.bias.data.to(device=device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(dtype=dtype)  # 保持在GPU上


def rotate_embeddings(model, Q: torch.Tensor, device: str = "cuda"):
    """旋转embedding权重"""
    W = model.model.embed_tokens
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(dtype=dtype)  # 保持在原设备上


@torch.no_grad()
def compute_layer_pca_rotations(
    model_path: str,
    nsamples: int = 128,
    seqlen: int = 2048,
    dataset_name: str = "wikitext",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_memory: bool = True
):
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
        (results, model): 包含每层Q矩阵和特征值的字典，以及旋转后的模型
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

        # 旋转embedding权重
        print("Rotating embeddings...")
        Q_device = Q.to(device).to(torch.float64)
        rotate_embeddings(model, Q_device, device)

        # 逐层处理
        num_layers = len(model.model.layers)

        for layer_idx in range(num_layers):
            print(f"\nProcessing layer {layer_idx}/{num_layers-1}...")

            # 将当前层移到GPU
            layer = model.model.layers[layer_idx].to(device)

            # ===== 步骤1: 用上一层的Q旋转当前层的attention输入 =====
            print(f"  Step 1: Rotating attention inputs with Q from layer {layer_idx-1 if layer_idx > 0 else 'embedding'}...")
            Q_device = Q.to(device).to(torch.float64)
            rotate_attention_inputs(layer, Q_device, device)

            # ===== 步骤2: 第一次前向传播,获取mlp_ln_inputs(丢弃输出) =====
            print(f"  Step 2: First forward pass to get MLP inputs...")
            mlp_ln_inputs = []

            for i in range(len(layer0_inputs)):
                inp = layer0_inputs[i].to(device)
                mask = masks[i] if masks else None
                rotated_inp = torch.matmul(inp, Q.to(device=device, dtype=inp.dtype))

                # 前向传播
                mlp_ln_inp, _ = get_layer_outputs(layer, rotated_inp, attention_mask=mask, position_ids=None)
                mlp_ln_inputs.append(mlp_ln_inp)
                cleanup_memory()

            # ===== 步骤3: 计算Q_mlp =====
            print(f"  Step 3: Computing PCA for MLP inputs...")
            eig_val_mlp, Q_mlp = pca_calc(mlp_ln_inputs, masks if masks else None, device)
            results[f"layer_{layer_idx}_mlp_eigenvalues"] = eig_val_mlp.cpu().numpy()
            results[f"layer_{layer_idx}_mlp_Q"] = Q_mlp.cpu().numpy()
            print(f"    MLP Q shape: {Q_mlp.shape}")

            # ===== 步骤4: 用Q_mlp旋转attention输出 =====
            print(f"  Step 4: Rotating attention output with Q_mlp...")
            Q_mlp_device = Q_mlp.to(device).to(torch.float64)
            rotate_attention_output(layer, Q_mlp_device, device)

            # ===== 步骤5: 用Q_mlp旋转MLP输入 =====
            print(f"  Step 5: Rotating MLP inputs with Q_mlp...")
            rotate_mlp_input(layer, Q_mlp_device, device)

            # ===== 步骤6: 第二次前向传播,获取layer_outputs =====
            print(f"  Step 6: Second forward pass to get layer outputs...")
            layer_outputs = []

            for i in range(len(layer0_inputs)):
                inp = layer0_inputs[i].to(device)
                mask = masks[i] if masks else None
                rotated_inp = torch.matmul(inp, Q.to(device=device, dtype=inp.dtype))

                # 前向传播
                _, out = get_layer_outputs(layer, rotated_inp, attention_mask=mask, position_ids=None)
                layer_outputs.append(out.cpu())
                cleanup_memory()

            # 计算Q_out
            print(f"  Step 6.5: Computing PCA for layer outputs...")
            eig_val_out, Q_out = pca_calc(layer_outputs, masks if masks else None, device)
            results[f"layer_{layer_idx}_output_eigenvalues"] = eig_val_out.cpu().numpy()
            results[f"layer_{layer_idx}_output_Q"] = Q_out.cpu().numpy()
            print(f"    Output Q shape: {Q_out.shape}")

            # ===== 步骤7: 用Q_out旋转MLP输出 =====
            print(f"  Step 7: Rotating MLP output with Q_out...")
            Q_out_device = Q_out.to(device).to(torch.float64)
            rotate_mlp_output(layer, Q_out_device, device)

            # 更新下一层的输入和Q
            layer0_inputs = layer_outputs
            Q = Q_out.cpu()  # 将Q移到CPU，下一层用的时候再移到GPU

            # 将层的权重移回CPU，然后将层模块移回CPU
            layer.to('cpu')
            cleanup_memory()

        # 保存旋转后的模型
        print("\n" + "="*50)
        print("All layers processed! Model has been rotated.")
        print("="*50)

        # 返回旋转后的模型和结果
        return results, model

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

    return results, None


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

    results, rotated_model = compute_layer_pca_rotations(
        model_path=model_path,
        nsamples=128,
        seqlen=2048,
        dataset_name="wikitext",
        save_memory=True
    )

    # 保存旋转矩阵
    save_rotation_matrices(results, "pca_rotation_matrices")

    # 保存旋转后的模型
    if rotated_model is not None:
        save_path = "rotated_llama_model"
        print(f"\nSaving rotated model to {save_path}...")
        rotated_model.save_pretrained(save_path)

        # 同时保存tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
        tokenizer.save_pretrained(save_path)

        print(f"Rotated model saved successfully to {save_path}!")
        print(f"You can load it later with: LlamaForCausalLM.from_pretrained('{save_path}')")
