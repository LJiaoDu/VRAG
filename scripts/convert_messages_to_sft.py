#!/usr/bin/env python3
"""
将多轮messages格式转换为SFTDataset需要的prompt-response格式

使用方法：
python scripts/convert_messages_to_sft.py \
    --input examples/search_sft.json \
    --output data/search_sft_converted.parquet \
    --tokenizer Qwen/Qwen2-VL-7B-Instruct
"""

import json
import argparse
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm


def convert_sliding_window(messages, tokenizer):
    """
    滑动窗口方式：每个assistant回复都生成一个训练样本

    样本1: [user1] -> assistant1
    样本2: [user1, assistant1, user2] -> assistant2
    样本3: [user1, assistant1, user2, assistant2, user3] -> assistant3

    优点：数据利用率高
    """
    samples = []
    current_history = []

    for msg in messages:
        if msg['role'] == 'user':
            current_history.append(msg)
        elif msg['role'] == 'assistant':
            # 使用tokenizer的chat_template格式化历史
            prompt = tokenizer.apply_chat_template(
                current_history,
                add_generation_prompt=True,  # 添加assistant前缀
                tokenize=False
            )

            response = msg['content']

            samples.append({
                'prompt': prompt,
                'response': response
            })

            # 将assistant加入历史
            current_history.append(msg)

    return samples


def convert_full_conversation(messages, tokenizer):
    """
    完整对话方式：整个历史作为prompt，最后一个assistant作为response

    优点：简单
    缺点：只用最后一个回复训练，数据利用率低
    """
    # 找到最后一个assistant
    last_assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]['role'] == 'assistant':
            last_assistant_idx = i
            break

    if last_assistant_idx is None:
        return None

    # 前面所有内容作为prompt
    prompt_messages = messages[:last_assistant_idx]
    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=False
    )

    response = messages[last_assistant_idx]['content']

    return {
        'prompt': prompt,
        'response': response
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='examples/search_sft.json',
                       help='输入的messages格式JSON文件')
    parser.add_argument('--output', type=str, default='data/search_sft_converted.parquet',
                       help='输出的parquet文件')
    parser.add_argument('--tokenizer', type=str, default='Qwen/Qwen2-VL-7B-Instruct',
                       help='使用的tokenizer')
    parser.add_argument('--method', type=str, default='sliding', choices=['sliding', 'full'],
                       help='转换方法: sliding(滑动窗口) 或 full(完整对话)')
    args = parser.parse_args()

    # 加载tokenizer
    print(f"加载tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # 读取数据
    print(f"读取数据: {args.input}")
    with open(args.input, 'r') as f:
        data = json.load(f)

    # 转换数据
    print(f"转换数据（方法: {args.method}）...")
    all_samples = []

    for item in tqdm(data):
        messages = item['messages']
        images = item.get('images', [])

        if args.method == 'sliding':
            samples = convert_sliding_window(messages, tokenizer)
            # 为每个样本分配对应的图片
            for i, sample in enumerate(samples):
                # 第i个assistant对应到前i+1张图片
                sample['images'] = images[:i+1] if images else []
                all_samples.append(sample)
        else:  # full
            sample = convert_full_conversation(messages, tokenizer)
            if sample:
                sample['images'] = images
                all_samples.append(sample)

    # 保存为parquet
    print(f"保存到: {args.output}")
    df = pd.DataFrame(all_samples)
    df.to_parquet(args.output)

    # 统计信息
    print("\n转换完成！")
    print(f"原始对话数量: {len(data)}")
    print(f"生成训练样本数量: {len(all_samples)}")
    print(f"\n数据列: {list(df.columns)}")
    print(f"\n第一个样本:")
    print(f"  Prompt前100字符: {all_samples[0]['prompt'][:100]}...")
    print(f"  Response: {all_samples[0]['response'][:100]}...")
    print(f"  Images数量: {len(all_samples[0].get('images', []))}")


if __name__ == '__main__':
    main()
