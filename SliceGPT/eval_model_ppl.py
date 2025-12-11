"""
模型困惑度评估脚本
使用与 SliceGPT 完全相同的数据集和评估方法
"""
import torch
import numpy as np
from transformers import LlamaForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import argparse
import gc


def cleanup_memory():
    """清理显存"""
    gc.collect()
    torch.cuda.empty_cache()


def get_wikitext2(tokenizer, seqlen: int = 2048):
    """
    加载 WikiText2 测试数据集
    与 SliceGPT 的 eval_slicegpt_ppl.py 中的实现完全一致
    """
    print("Loading WikiText2 test dataset...")
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # 将整个测试集连接成一个长文本
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    return testenc


class WikiText2Dataset(Dataset):
    """
    WikiText2 数据集类
    与 SliceGPT 的 prepare_test_dataloader 实现完全一致
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
    与 SliceGPT 的 data_utils.prepare_test_dataloader 完全一致
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
    与 SliceGPT 的 gpu_utils.evaluate_ppl 完全一致

    Args:
        model: 待评估的模型
        pad_token_id: padding token ID
        testloader: 测试数据加载器
        device: 设备

    Returns:
        perplexity: 困惑度
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
    与 SliceGPT 的 eval_slicegpt_ppl.py 中的实现一致

    这个方法用于快速验证，不使用批处理
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate model perplexity on WikiText2")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model directory"
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

    print("="*50)
    print("Model Perplexity Evaluation")
    print("="*50)
    print(f"Model path: {args.model_path}")
    print(f"Sequence length: {args.seqlen}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print("="*50)

    # 加载模型和 tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map=args.device
    )
    model.eval()

    print(f"Model loaded: {model.config.model_type}")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")

    # 评估困惑度
    if args.simple:
        # 使用简单方法
        ppl = evaluate_ppl_simple(model, tokenizer, args.seqlen, args.device)
    else:
        # 使用批处理方法
        test_loader = prepare_test_dataloader(tokenizer, args.seqlen, args.batch_size)
        pad_token_id = model.config.pad_token_id
        ppl = evaluate_ppl(model, pad_token_id, test_loader, args.device)

    # 打印最终结果
    print("\n" + "="*50)
    print(f"Final Perplexity: {ppl:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
