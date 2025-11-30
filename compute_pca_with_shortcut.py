"""
计算 PCA 旋转矩阵并保存 shortcut Q 矩阵
实现与 SliceGPT 完全相同的残差连接处理逻辑
"""
import torch
import numpy as np
from transformers import LlamaForCausalLM, AutoTokenizer
import json
from typing import Dict
import torch.nn as nn

# 导入已有的函数
from compute_pca_rotation import (
    prepare_calibration_data,
    pca_calc,
    get_layer0_inputs,
    get_layer_outputs,
    rotate_attention_inputs,
    rotate_attention_output,
    rotate_mlp_input,
    rotate_mlp_output,
    rotate_embeddings,
    cleanup_memory
)


def fuse_ln_linear(layernorm, linear_layers):
    """
    将 RMSNorm 的权重融合到后续的 Linear 层中
    参考 SliceGPT 的 fuse_ln_linear 实现

    专门针对 Llama 模型（使用 RMSNorm，无 bias）

    原理：
    原始: y = Linear(RMSNorm(x))
          = W @ (x * rms_weight)

    融合后: y = (W * rms_weight) @ x

    其中 rms_weight 是 RMSNorm 的缩放系数（逐元素）

    Args:
        layernorm: RMSNorm 层（Llama 的 LayerNorm，实际是 RMSNorm）
        linear_layers: 下游的 Linear 层列表（如 q_proj, k_proj, v_proj）
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # 将 Linear 权重的每一行乘以对应的 RMSNorm scale
        # W_new[i, :] = W[i, :] * rms_weight[i]
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

    print(f"  ✓ Fused RMSNorm scale into {len(linear_layers)} linear layers")


class CompressedLlamaDecoderLayer(nn.Module):
    """
    压缩的 Llama Decoder Layer，包含 shortcut Q 矩阵
    参考 SliceGPT 的 CompressedLlamaDecoderLayer 实现
    """
    def __init__(self, original_layer, attn_shortcut_Q=None, mlp_shortcut_Q=None):
        super().__init__()
        self.original_layer = original_layer

        # Shortcut Q 矩阵（用于残差连接）
        if attn_shortcut_Q is not None:
            self.attn_shortcut_Q = nn.Parameter(attn_shortcut_Q, requires_grad=False)
        else:
            self.attn_shortcut_Q = None

        if mlp_shortcut_Q is not None:
            self.mlp_shortcut_Q = nn.Parameter(mlp_shortcut_Q, requires_grad=False)
        else:
            self.mlp_shortcut_Q = None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        """
        前向传播，实现 SliceGPT 的残差连接逻辑

        根据图示：
        - Attention 残差: residual @ (Q1^T @ Q2) + attention_output
        - FFN 残差: residual @ (Q1^T @ Q3) + ffn_output
        """
        residual = hidden_states

        # ===== Attention 部分 =====
        # 1. Input LayerNorm
        hidden_states = self.original_layer.input_layernorm(hidden_states)

        # 2. Self Attention
        hidden_states, self_attn_weights, present_key_value = self.original_layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        # 3. Attention 残差连接 (使用 shortcut Q)
        if self.attn_shortcut_Q is not None:
            # residual @ (Q_prev^T @ Q_attn) + attention_output
            rotated_residual = torch.matmul(residual, self.attn_shortcut_Q)
            hidden_states = rotated_residual + hidden_states
        else:
            # 没有 shortcut Q 时，直接相加
            hidden_states = residual + hidden_states

        # ===== FFN 部分 =====
        residual = hidden_states

        # 4. Post Attention LayerNorm
        hidden_states = self.original_layer.post_attention_layernorm(hidden_states)

        # 5. MLP
        hidden_states = self.original_layer.mlp(hidden_states)

        # 6. FFN 残差连接 (使用 shortcut Q)
        if self.mlp_shortcut_Q is not None:
            # residual @ (Q_attn^T @ Q_mlp) + mlp_output
            rotated_residual = torch.matmul(residual, self.mlp_shortcut_Q)
            hidden_states = rotated_residual + hidden_states
        else:
            # 没有 shortcut Q 时，直接相加
            hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


@torch.no_grad()
def compute_layer_pca_rotations_with_shortcut(
    model_path: str,
    nsamples: int = 128,
    seqlen: int = 2048,
    dataset_name: str = "wikitext",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_memory: bool = True
):
    """
    逐层计算 PCA 旋转矩阵，并计算 shortcut Q 矩阵

    Shortcut Q 的计算逻辑（参考 SliceGPT）：
    - attn_shortcut_Q = Q_prev^T @ Q_mlp
    - mlp_shortcut_Q = Q_mlp^T @ Q_out

    其中：
    - Q_prev: 上一层的输出 Q（或 embedding Q）
    - Q_mlp: 当前层 attention 后的 Q
    - Q_out: 当前层 FFN 后的 Q
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
    shortcut_Qs = {}  # 保存所有的 shortcut Q 矩阵

    if save_memory:
        # 第一步：获取embedding输出（第0层输入）
        print("Computing embedding outputs (layer 0 inputs)...")
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            local_files_only=True
        )
        model.eval()

        # 收集所有样本的layer0输入
        layer0_inputs = []
        masks = []

        for i in range(len(input_ids)):
            batch_ids = input_ids[i:i+1].to(device)
            batch_mask = attention_mask[i:i+1] if attention_mask is not None else None

            model.model.embed_tokens.to(device)
            layer0_input = get_layer0_inputs(model, tokenizer, batch_ids, batch_mask)
            layer0_inputs.append(layer0_input)
            if batch_mask is not None:
                masks.append(batch_mask.cpu())

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

        # ===== 重要：解绑 lm_head 和 embed_tokens 的权重共享 =====
        # Llama 默认 lm_head.weight 和 embed_tokens.weight 是同一块内存（tie_word_embeddings=True）
        # 如果不解绑，旋转 embedding 时会把 lm_head 也旋转了，导致后续再旋转 lm_head 时出错
        print("Untying lm_head from embed_tokens...")
        model.lm_head.weight = nn.Parameter(model.lm_head.weight.clone())
        print("✓ lm_head.weight is now independent from embed_tokens.weight")

        # 旋转embedding权重
        print("Rotating embeddings...")
        Q_device = Q.to(device).to(torch.float64)
        rotate_embeddings(model, Q_device, device)

        # 逐层处理
        num_layers = len(model.model.layers)
        Q_prev = Q  # 保存上一层的 Q（初始为 embedding Q）

        for layer_idx in range(num_layers):
            print(f"\nProcessing layer {layer_idx}/{num_layers-1}...")

            # 将当前层移到GPU
            layer = model.model.layers[layer_idx].to(device)

            # ===== 步骤0: Fuse LayerNorm 到 Linear 层 =====
            print(f"  Step 0: Fusing LayerNorm into Linear layers...")

            # Fuse input_layernorm 到 attention 输入层 (q_proj, k_proj, v_proj)
            fuse_ln_linear(
                layer.input_layernorm,
                [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]
            )

            # Fuse post_attention_layernorm 到 MLP 输入层 (gate_proj, up_proj)
            fuse_ln_linear(
                layer.post_attention_layernorm,
                [layer.mlp.gate_proj, layer.mlp.up_proj]
            )

            # ===== 步骤0.5: 将 RMSNorm 权重设置为 1.0 =====
            # 重要！fuse 之后必须将权重设为 1.0，否则 gamma 会被应用两次
            # 不替换类，保持 state_dict 兼容性
            print(f"  Step 0.5: Setting RMSNorm weights to 1.0...")
            torch.nn.init.ones_(layer.input_layernorm.weight)
            torch.nn.init.ones_(layer.post_attention_layernorm.weight)
            print(f"  ✓ Set 2 RMSNorm weights to 1.0 (gamma disabled)")

            # ===== 步骤1: 用上一层的Q旋转当前层的attention输入 =====
            print(f"  Step 1: Rotating attention inputs with Q from layer {layer_idx-1 if layer_idx > 0 else 'embedding'}...")
            Q_device = Q_prev.to(device).to(torch.float64)
            rotate_attention_inputs(layer, Q_device, device)

            # ===== 步骤2: 第一次前向传播,获取mlp_ln_inputs =====
            print(f"  Step 2: First forward pass to get MLP inputs...")
            mlp_ln_inputs = []

            for i in range(len(layer0_inputs)):
                inp = layer0_inputs[i].to(device)
                mask = masks[i] if masks else None
                rotated_inp = torch.matmul(inp, Q_prev.to(device=device, dtype=inp.dtype))

                mlp_ln_inp, _ = get_layer_outputs(layer, rotated_inp, attention_mask=mask, position_ids=None)
                mlp_ln_inputs.append(mlp_ln_inp)
                cleanup_memory()

            # ===== 步骤3: 计算Q_mlp =====
            print(f"  Step 3: Computing PCA for MLP inputs...")
            eig_val_mlp, Q_mlp = pca_calc(mlp_ln_inputs, masks if masks else None, device)
            results[f"layer_{layer_idx}_mlp_eigenvalues"] = eig_val_mlp.cpu().numpy()
            results[f"layer_{layer_idx}_mlp_Q"] = Q_mlp.cpu().numpy()
            print(f"    MLP Q shape: {Q_mlp.shape}")

            # ===== 计算 attn_shortcut_Q = Q_prev^T @ Q_mlp =====
            print(f"  Computing attn_shortcut_Q...")
            dtype = torch.float32  # 使用 float32 存储 shortcut Q
            attn_shortcut_Q = torch.matmul(
                Q_prev.to(device).to(dtype).T,  # Q_prev^T
                Q_mlp.to(device).to(dtype)      # Q_mlp
            ).cpu()
            shortcut_Qs[f"layer_{layer_idx}_attn_shortcut_Q"] = attn_shortcut_Q.numpy()
            print(f"    attn_shortcut_Q shape: {attn_shortcut_Q.shape}")

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
                rotated_inp = torch.matmul(inp, Q_prev.to(device=device, dtype=inp.dtype))

                _, out = get_layer_outputs(layer, rotated_inp, attention_mask=mask, position_ids=None)
                layer_outputs.append(out.cpu())
                cleanup_memory()

            # 计算Q_out
            print(f"  Step 6.5: Computing PCA for layer outputs...")
            eig_val_out, Q_out = pca_calc(layer_outputs, masks if masks else None, device)
            results[f"layer_{layer_idx}_output_eigenvalues"] = eig_val_out.cpu().numpy()
            results[f"layer_{layer_idx}_output_Q"] = Q_out.cpu().numpy()
            print(f"    Output Q shape: {Q_out.shape}")

            # ===== 计算 mlp_shortcut_Q = Q_mlp^T @ Q_out =====
            print(f"  Computing mlp_shortcut_Q...")
            mlp_shortcut_Q = torch.matmul(
                Q_mlp.to(device).to(dtype).T,  # Q_mlp^T
                Q_out.to(device).to(dtype)     # Q_out
            ).cpu()
            shortcut_Qs[f"layer_{layer_idx}_mlp_shortcut_Q"] = mlp_shortcut_Q.numpy()
            print(f"    mlp_shortcut_Q shape: {mlp_shortcut_Q.shape}")

            # ===== 步骤7: 用Q_out旋转MLP输出 =====
            print(f"  Step 7: Rotating MLP output with Q_out...")
            Q_out_device = Q_out.to(device).to(torch.float64)
            rotate_mlp_output(layer, Q_out_device, device)

            # 更新下一层的输入和Q
            layer0_inputs = layer_outputs
            Q_prev = Q_out.cpu()  # 当前层的 Q_out 成为下一层的 Q_prev

            # 将层移回CPU
            layer.to('cpu')
            cleanup_memory()

        # ===== Fuse pre-head LayerNorm 到 lm_head =====
        print("\n" + "="*50)
        print("Fusing pre-head LayerNorm (model.norm) into lm_head...")
        print("="*50)

        # Llama 的最后一层 LayerNorm (model.norm)
        fuse_ln_linear(model.model.norm, [model.lm_head])

        # 将 model.norm 权重设置为 1.0（不替换类，保持兼容性）
        print("Setting model.norm weight to 1.0...")
        torch.nn.init.ones_(model.model.norm.weight)
        print("✓ Set model.norm weight to 1.0 (gamma disabled)")

        # ===== 旋转 lm_head (Language Model Head) =====
        print("\n" + "="*50)
        print("Rotating lm_head (final projection to vocabulary)...")
        print("="*50)
        print(f"Using Q_out from last layer (layer {num_layers-1})")

        # Q_prev 现在是最后一层的 Q_out
        lm_head = model.lm_head
        print(f"Original lm_head weight shape: {lm_head.weight.shape}")

        # 旋转 lm_head: W_new = W @ Q
        dtype = lm_head.weight.data.dtype
        W_ = lm_head.weight.data.to(device=device, dtype=torch.float64)
        Q_device = Q_prev.to(device=device, dtype=torch.float64)

        lm_head.weight.data = torch.matmul(W_, Q_device).to(device="cpu", dtype=dtype)
        print(f"Rotated lm_head weight shape: {lm_head.weight.shape}")
        print("✓ lm_head rotation complete")

        cleanup_memory()

        # 替换为 CompressedLlamaDecoderLayer
        print("\n" + "="*50)
        print("Replacing layers with CompressedLlamaDecoderLayer...")
        print("="*50)

        for layer_idx in range(num_layers):
            original_layer = model.model.layers[layer_idx]

            # 加载 shortcut Q 矩阵
            attn_shortcut_Q_np = shortcut_Qs.get(f"layer_{layer_idx}_attn_shortcut_Q")
            mlp_shortcut_Q_np = shortcut_Qs.get(f"layer_{layer_idx}_mlp_shortcut_Q")

            attn_shortcut_Q = torch.from_numpy(attn_shortcut_Q_np).to(dtype=torch.float16) if attn_shortcut_Q_np is not None else None
            mlp_shortcut_Q = torch.from_numpy(mlp_shortcut_Q_np).to(dtype=torch.float16) if mlp_shortcut_Q_np is not None else None

            # 创建压缩层
            compressed_layer = CompressedLlamaDecoderLayer(
                original_layer,
                attn_shortcut_Q=attn_shortcut_Q,
                mlp_shortcut_Q=mlp_shortcut_Q
            )

            # 替换层
            model.model.layers[layer_idx] = compressed_layer

            print(f"  Layer {layer_idx}: attn_shortcut_Q={attn_shortcut_Q.shape if attn_shortcut_Q is not None else None}, "
                  f"mlp_shortcut_Q={mlp_shortcut_Q.shape if mlp_shortcut_Q is not None else None}")

        print("\n" + "="*50)
        print("All layers processed! Model has been rotated with shortcut Q.")
        print("="*50)

        return results, shortcut_Qs, model

    else:
        raise NotImplementedError("Non-memory-saving mode not fully implemented")


def save_results(results: Dict[str, np.ndarray], shortcut_Qs: Dict[str, np.ndarray], save_path: str):
    """保存旋转矩阵和 shortcut Q 矩阵"""
    # 1. 保存旋转矩阵
    np.savez_compressed(save_path + "_rotation.npz", **results)
    print(f"Saved rotation matrices to {save_path}_rotation.npz")

    # 2. 保存 shortcut Q 矩阵
    np.savez_compressed(save_path + "_shortcut.npz", **shortcut_Qs)
    print(f"Saved shortcut Q matrices to {save_path}_shortcut.npz")

    # 3. 保存统计信息
    stats = {
        "rotation_matrices": {},
        "shortcut_matrices": {}
    }

    for name, values in results.items():
        if "eigenvalues" in name:
            stats["rotation_matrices"][name] = {
                "shape": list(values.shape),
                "top_10": values[:10].tolist(),
                "sum": float(values.sum()),
                "mean": float(values.mean())
            }
        elif "_Q" in name:
            stats["rotation_matrices"][name] = {
                "shape": list(values.shape),
                "dtype": str(values.dtype)
            }

    for name, values in shortcut_Qs.items():
        stats["shortcut_matrices"][name] = {
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "norm": float(np.linalg.norm(values))
        }

    with open(save_path + "_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {save_path}_stats.json")


if __name__ == "__main__":
    model_path = "/gpfs/volcano/models/meta-llama/Llama-2-13b-hf"

    # 计算旋转矩阵和 shortcut Q 矩阵
    results, shortcut_Qs, rotated_model = compute_layer_pca_rotations_with_shortcut(
        model_path=model_path,
        nsamples=128,
        seqlen=2048,
        dataset_name="wikitext",
        save_memory=True
    )

    # 保存结果
    save_results(results, shortcut_Qs, "pca_with_shortcut_lm_head")

    # 保存旋转后的模型（包含 shortcut Q）
    # 模仿 SliceGPT 的保存方式：只保存 state_dict
    save_path = "rotated_llama_model_with_shortcut_lm_head_ln_fusion"
    print(f"\nSaving rotated model with shortcut Q to {save_path}...")
    print("Using SliceGPT-style saving method (state_dict only)...")

    import os
    import pathlib
    save_dir = pathlib.Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. 保存 state_dict（包含所有权重，包括 shortcut_Q）
    state_dict_path = save_dir / "rotated_model.pt"
    torch.save(rotated_model.state_dict(), state_dict_path)
    print(f"✓ Saved model state_dict to {state_dict_path}")

    # 2. 保存 tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
    tokenizer.save_pretrained(save_dir)
    print(f"✓ Saved tokenizer to {save_dir}")

    # 3. 保存 config.json
    rotated_model.config.save_pretrained(save_dir)
    print(f"✓ Saved config.json to {save_dir}")

    # 4. 保存 CompressedLlamaDecoderLayer 类定义
    # 将当前文件中的 CompressedLlamaDecoderLayer 提取并保存
    modeling_file = save_dir / "modeling_compressed_llama.py"
    with open(__file__, 'r') as f:
        content = f.read()

    # 提取类定义
    class_start = content.find("class CompressedLlamaDecoderLayer")
    if class_start != -1:
        # 找到类定义结束
        class_end = content.find("\n\nclass ", class_start + 1)
        if class_end == -1:
            class_end = content.find("\n\n@", class_start + 1)
        if class_end == -1:
            class_end = content.find("\n\ndef ", class_start + 1)

        if class_end != -1:
            class_definition = content[class_start:class_end]

            with open(modeling_file, 'w') as f:
                f.write('"""\nCompressedLlamaDecoderLayer for rotated models with shortcut Q\n')
                f.write('This file is auto-generated by compute_pca_with_shortcut.py\n"""\n\n')
                f.write("import torch\n")
                f.write("from torch import nn\n")
                f.write("from transformers.models.llama.modeling_llama import LlamaDecoderLayer\n\n")
                f.write(class_definition)

            print(f"✓ Saved CompressedLlamaDecoderLayer definition to {modeling_file}")

    print(f"\n{'='*60}")
    print(f"Model saved successfully!")
    print(f"{'='*60}")
    print(f"Location: {save_dir}")
    print(f"Files saved:")
    print(f"  - rotated_model.pt (state_dict with shortcut_Q matrices)")
    print(f"  - config.json (model configuration)")
    print(f"  - tokenizer files")
    print(f"  - modeling_compressed_llama.py (CompressedLlamaDecoderLayer class)")
    print(f"\nTo evaluate, use:")
    print(f"  python eval_rotated_model_ppl.py --model-path {save_path}")
    print(f"{'='*60}")
