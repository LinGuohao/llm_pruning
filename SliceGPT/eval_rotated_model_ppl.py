"""
评估旋转模型（带 shortcut Q）的困惑度
使用与 eval_model_ppl.py 完全相同的数据集和评估方法
加载逻辑参考 SliceGPT 的 load_sliced_model()
"""
import torch
import numpy as np
from transformers import LlamaForCausalLM, AutoTokenizer, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import argparse
import gc
import sys
import pathlib


def cleanup_memory():
    """清理显存"""
    gc.collect()
    torch.cuda.empty_cache()


def get_wikitext2(tokenizer, seqlen: int = 2048):
    """
    加载 WikiText2 测试数据集
    与 eval_model_ppl.py 完全一致
    """
    print("Loading WikiText2 test dataset...")
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # 将整个测试集连接成一个长文本
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    return testenc


class WikiText2Dataset(Dataset):
    """
    WikiText2 数据集类
    与 eval_model_ppl.py 完全一致
    """
    def __init__(self, tokenized_data, seqlen=2048):
        # 计算样本数量
        nsamples = tokenized_data.input_ids.numel() // seqlen

        # 精确切割成固定长度的块
        input_ids = tokenized_data.input_ids[0, : nsamples * seqlen]
        input_ids = input_ids.reshape(nsamples, seqlen)

        attn_mask = tokenized_data.attention_mask[0, : nsamples * seqlen]
        attn_mask = attn_mask.reshape(nsamples, seqlen)

        self.input_ids = input_ids
        self.attn_mask = attn_mask

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attn_mask[idx]
        }

    def __len__(self):
        return len(self.input_ids)


def prepare_test_dataloader(tokenizer, seqlen: int = 2048, batch_size: int = 1):
    """
    准备测试数据加载器
    与 eval_model_ppl.py 完全一致
    """
    # 加载数据
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Tokenize
    tokenized_data = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # 创建数据集
    test_dataset = WikiText2Dataset(tokenized_data, seqlen)

    # 创建数据加载器
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Test dataset: {len(test_dataset)} samples")

    return test_loader


@torch.no_grad()
def evaluate_ppl(model, pad_token_id, testloader, device="cuda"):
    """
    评估困惑度
    与 eval_model_ppl.py 完全一致
    """
    print("\nEvaluating perplexity...")
    model.eval()

    # 设置损失函数
    if pad_token_id is not None:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    nlls = []

    # 批处理评估
    for i, batch in enumerate(testloader):
        # 将批次移到设备
        batch = {k: v.to(device) for k, v in batch.items()}

        # 前向传播
        logits = model(**batch).logits  # (batch_size, seq_len, vocab_size)

        # 自回归位移: 预测 token_{i+1} 从 token_i
        logits = logits[:, :-1, :]  # (batch_size, seq_len-1, vocab_size)
        shift_labels = batch["input_ids"][:, 1:]  # (batch_size, seq_len-1)

        # 计算 loss
        # CrossEntropyLoss 需要 logits 形状: (batch, vocab_size, seq_len)
        nll = loss_fn(
            logits.permute(0, 2, 1),  # (batch, vocab_size, seq_len-1)
            shift_labels
        ).float()  # (batch, seq_len-1)

        # Mask 处理: 忽略 padding tokens
        mask = shift_labels != loss_fn.ignore_index  # (batch, seq_len-1)
        nll_means = (nll * mask).sum(dim=1) / mask.sum(dim=1)  # (batch,)
        nlls.append(nll_means)

        # 打印进度
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(testloader)} batches")

        # 清理显存
        cleanup_memory()

    # 计算最终困惑度
    nlls_tensor = torch.cat(nlls)  # (total_samples,)
    ppl = torch.exp(nlls_tensor.mean())

    print(f"\nPerplexity: {ppl.item():.4f}")

    return ppl.item()


@torch.no_grad()
def evaluate_ppl_simple(model, tokenizer, seqlen: int = 2048, device="cuda"):
    """
    简单的困惑度评估（单个样本循环）
    与 eval_model_ppl.py 完全一致
    """
    print("\nEvaluating perplexity (simple method)...")
    model.eval()

    # 加载数据
    testenc = get_wikitext2(tokenizer, seqlen)

    # 计算样本数量
    nsamples = testenc.input_ids.numel() // seqlen
    input_tok = testenc.input_ids[0, :(seqlen * nsamples)].view(nsamples, seqlen)

    print(f"Number of samples: {nsamples}")

    # 损失函数
    loss_fct = torch.nn.CrossEntropyLoss().to(device)
    acc_loss = 0.0

    # 逐样本评估
    for ii in range(nsamples):
        input_ids = input_tok[ii, :].to(device).view(1, -1)  # (1, seqlen)

        # 前向传播
        output = model(input_ids, use_cache=False, output_hidden_states=False)[0]
        # output: (1, seqlen, vocab_size)

        # 自回归位移
        shift_logits = output[:, :-1, :].contiguous()  # (1, seqlen-1, vocab_size)
        shift_labels = input_ids[:, 1:]  # (1, seqlen-1)

        # 计算损失
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),  # (seqlen-1, vocab_size)
            shift_labels.view(-1)  # (seqlen-1,)
        )

        acc_loss += loss.item()

        # 打印进度
        if (ii + 1) % 10 == 0:
            print(f"  Processed {ii + 1}/{nsamples} samples")

        # 清理显存
        cleanup_memory()

    # 计算平均损失和困惑度
    avg_loss = acc_loss / nsamples
    ppl = np.exp(avg_loss)

    print(f"\nAverage loss: {avg_loss:.4f}")
    print(f"Perplexity: {ppl:.4f}")

    return ppl


def load_rotated_model(model_path: str, device="cuda"):
    """
    加载旋转模型（带 shortcut Q）
    参考 SliceGPT 的 load_sliced_model() 实现

    加载流程：
    1. 从 config.json 创建未初始化的模型
    2. 导入 CompressedLlamaDecoderLayer 类
    3. 替换所有层为 CompressedLlamaDecoderLayer
    4. 加载 state_dict（包含 shortcut_Q 权重）
    """
    model_dir = pathlib.Path(model_path)

    print("="*60)
    print("Loading rotated model with shortcut Q...")
    print("="*60)

    # 1. 加载 config（不加载权重）
    print("Step 1: Loading model config...")
    config = LlamaConfig.from_pretrained(model_dir)

    # 2. 创建未初始化的模型（只有结构，没有权重）
    print("Step 2: Creating uninitialized model...")

    class UninitializedLlamaForCausalLM(LlamaForCausalLM):
        def _init_weights(self, module):
            # 防止权重初始化
            pass

    model = UninitializedLlamaForCausalLM(config)
    model = model.to(dtype=torch.float16)

    # 3. 导入 CompressedLlamaDecoderLayer 类
    print("Step 3: Importing CompressedLlamaDecoderLayer...")

    # 动态导入保存的类定义
    modeling_file = model_dir / "modeling_compressed_llama.py"
    if not modeling_file.exists():
        raise FileNotFoundError(
            f"CompressedLlamaDecoderLayer definition not found at {modeling_file}\n"
            f"Make sure you saved the model using compute_pca_with_shortcut.py"
        )

    # 将模型目录添加到 Python 路径
    sys.path.insert(0, str(model_dir))

    try:
        from modeling_compressed_llama import CompressedLlamaDecoderLayer
    except ImportError as e:
        raise ImportError(
            f"Failed to import CompressedLlamaDecoderLayer from {modeling_file}\n"
            f"Error: {e}"
        )

    # 4. 替换所有层为 CompressedLlamaDecoderLayer
    print("Step 4: Replacing layers with CompressedLlamaDecoderLayer...")

    hidden_size = config.hidden_size
    num_layers = len(model.model.layers)

    for layer_idx in range(num_layers):
        original_layer = model.model.layers[layer_idx]

        # 创建压缩层（先不设置 shortcut_Q，从 state_dict 加载）
        compressed_layer = CompressedLlamaDecoderLayer(
            original_layer,
            attn_shortcut_Q=None,
            mlp_shortcut_Q=None
        )

        # 注册 shortcut_Q 参数（初始化为零，后续从 state_dict 加载实际值）
        compressed_layer.attn_shortcut_Q = torch.nn.Parameter(
            torch.zeros(hidden_size, hidden_size, dtype=torch.float16)
        )
        compressed_layer.mlp_shortcut_Q = torch.nn.Parameter(
            torch.zeros(hidden_size, hidden_size, dtype=torch.float16)
        )

        # 替换层
        model.model.layers[layer_idx] = compressed_layer

    print(f"✓ Replaced {num_layers} layers with CompressedLlamaDecoderLayer")

    # 5. 加载 state_dict（包含所有权重和 shortcut_Q 矩阵）
    print("Step 5: Loading state_dict (weights + shortcut_Q)...")

    state_dict_path = model_dir / "rotated_model.pt"
    if not state_dict_path.exists():
        raise FileNotFoundError(f"Model weights not found at {state_dict_path}")

    state_dict = torch.load(state_dict_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    print(f"✓ Loaded state_dict from {state_dict_path}")

    # 6. 设置模型为评估模式
    model.eval()

    print("="*60)
    print("✓ Model loaded successfully!")
    print(f"  Model type: {config.model_type}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num layers: {num_layers}")
    print(f"  All layers use CompressedLlamaDecoderLayer with shortcut Q")
    print("="*60)

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate rotated model (with shortcut Q) perplexity on WikiText2"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the rotated model directory (saved by compute_pca_with_shortcut.py)"
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=2048,
        help="Sequence length for evaluation (default: 2048)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation (default: 1)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simple evaluation method (no batching)"
    )

    args = parser.parse_args()

    print("="*60)
    print("Rotated Model Perplexity Evaluation")
    print("="*60)
    print(f"Model path: {args.model_path}")
    print(f"Sequence length: {args.seqlen}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Evaluation method: {'simple' if args.simple else 'batched'}")
    print("="*60)

    # 加载 tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded")

    # 加载旋转模型（使用 SliceGPT 风格的加载）
    model = load_rotated_model(args.model_path, args.device)

    # 移动到设备
    print(f"\nMoving model to {args.device}...")
    model.to(args.device)

    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")

    # 评估困惑度（使用与 eval_model_ppl.py 完全相同的方法）
    if args.simple:
        # 使用简单方法
        ppl = evaluate_ppl_simple(model, tokenizer, args.seqlen, args.device)
    else:
        # 使用批处理方法
        test_loader = prepare_test_dataloader(tokenizer, args.seqlen, args.batch_size)
        pad_token_id = model.config.pad_token_id
        ppl = evaluate_ppl(model, pad_token_id, test_loader, args.device)

    # 打印最终结果
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Perplexity: {ppl:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
