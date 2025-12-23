#!/usr/bin/env python3
"""
准备SFT训练数据
将转换好的messages格式数据转为训练可用的格式
"""

import json
import pandas as pd
import argparse
from pathlib import Path


def convert_to_parquet(input_file, output_file, train_ratio=0.9):
    """
    转换为Parquet格式（用于FSDP Trainer）

    参数:
        input_file: 输入JSON文件（convert_messages_to_sft.py的输出）
        output_file: 输出parquet文件路径
        train_ratio: 训练集比例
    """
    print(f"读取数据: {input_file}")
    df = pd.read_parquet(input_file)

    print(f"数据条数: {len(df)}")
    print(f"数据列: {list(df.columns)}")

    # 分割训练集和验证集
    train_size = int(len(df) * train_ratio)
    train_df = df[:train_size]
    val_df = df[train_size:]

    # 保存
    train_output = output_file.replace('.parquet', '_train.parquet')
    val_output = output_file.replace('.parquet', '_val.parquet')

    train_df.to_parquet(train_output)
    val_df.to_parquet(val_output)

    print(f"\n✅ 转换完成！")
    print(f"训练集: {train_output} ({len(train_df)} 条)")
    print(f"验证集: {val_output} ({len(val_df)} 条)")

    # 显示第一个样本
    print(f"\n第一个样本预览:")
    sample = train_df.iloc[0]
    print(f"  Prompt前100字符: {sample['prompt'][:100]}...")
    print(f"  Response前100字符: {sample['response'][:100]}...")
    if 'images' in sample and len(sample['images']) > 0:
        print(f"  Images数量: {len(sample['images'])}")

    return train_output, val_output


def convert_to_llamafactory(input_file, output_file):
    """
    转换为LLaMA Factory格式

    参数:
        input_file: 输入parquet文件
        output_file: 输出JSON文件
    """
    print(f"读取数据: {input_file}")
    df = pd.read_parquet(input_file)

    # 重构为messages格式
    # 假设输入已经是prompt-response格式，需要重建messages
    llamafactory_data = []

    for idx, row in df.iterrows():
        # 将prompt和response合并为messages
        # 注意：这里假设prompt已经是chat_template处理后的
        # 需要根据实际情况调整

        messages = [
            {
                "role": "user",
                "content": row['prompt']
            },
            {
                "role": "assistant",
                "content": row['response']
            }
        ]

        item = {"messages": messages}

        if 'images' in row and row['images']:
            item['images'] = row['images']

        llamafactory_data.append(item)

    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(llamafactory_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 转换完成！")
    print(f"输出文件: {output_file}")
    print(f"样本数量: {len(llamafactory_data)}")

    # 显示第一个样本
    print(f"\n第一个样本预览:")
    print(json.dumps(llamafactory_data[0], indent=2, ensure_ascii=False))

    return output_file


def main():
    parser = argparse.ArgumentParser(description='准备SFT训练数据')
    parser.add_argument('--input', type=str, required=True,
                       help='输入parquet文件路径')
    parser.add_argument('--output-parquet', type=str, default='data/sft_data.parquet',
                       help='输出parquet文件路径（用于FSDP Trainer）')
    parser.add_argument('--output-json', type=str, default='data/llamafactory_sft.json',
                       help='输出JSON文件路径（用于LLaMA Factory）')
    parser.add_argument('--format', type=str, choices=['parquet', 'json', 'both'],
                       default='both',
                       help='输出格式：parquet（FSDP）, json（LLaMA Factory）, both（两者）')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                       help='训练集比例（仅用于parquet格式）')

    args = parser.parse_args()

    # 检查输入文件
    if not Path(args.input).exists():
        print(f"❌ 错误：输入文件不存在: {args.input}")
        return

    print("="*60)
    print("准备SFT训练数据")
    print("="*60)

    # 转换为parquet格式
    if args.format in ['parquet', 'both']:
        print("\n📦 转换为Parquet格式（FSDP Trainer）...")
        train_file, val_file = convert_to_parquet(
            args.input,
            args.output_parquet,
            args.train_ratio
        )

    # 转换为LLaMA Factory格式
    if args.format in ['json', 'both']:
        print("\n📦 转换为LLaMA Factory格式...")
        # 使用训练集转换
        if args.format == 'both':
            input_for_json = train_file
        else:
            input_for_json = args.input

        json_file = convert_to_llamafactory(
            input_for_json,
            args.output_json
        )

    print("\n" + "="*60)
    print("✅ 所有转换完成！")
    print("="*60)

    # 打印下一步说明
    print("\n📋 下一步操作：")

    if args.format in ['parquet', 'both']:
        print("\n使用FSDP Trainer训练:")
        print(f"  python -m verl.trainer.fsdp_sft_trainer \\")
        print(f"    data.train_files={train_file} \\")
        print(f"    data.val_files={val_file} \\")
        print(f"    data.prompt_key=prompt \\")
        print(f"    data.response_key=response \\")
        print(f"    model.partial_pretrain=Qwen/Qwen2.5-VL-7B-Instruct")

    if args.format in ['json', 'both']:
        print("\n使用LLaMA Factory训练:")
        print(f"  1. 将 {json_file} 复制到 LLaMA-Factory/data/")
        print(f"  2. 在 LLaMA-Factory/data/dataset_info.json 中注册数据集")
        print(f"  3. 运行: llamafactory-cli train examples/train_qlora/qwen2vl_sft.yaml")


if __name__ == '__main__':
    main()
