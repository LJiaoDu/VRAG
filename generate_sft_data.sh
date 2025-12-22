#!/bin/bash
set -e  # 遇到错误立即退出

echo "========================================"
echo "VRAG SFT数据集生成完整流程"
echo "========================================"

# ============================================
# 步骤1: 检查准备工作
# ============================================
echo ""
echo "步骤1: 检查准备工作..."

if [ ! -d "search_engine/corpus/img" ]; then
    echo "❌ 错误: 图像目录不存在，请先运行 pdf2images.py"
    exit 1
fi

if [ ! -f "search_engine/corpus/rag_dataset.json" ]; then
    echo "⚠️  警告: RAG数据集不存在，复制示例文件..."
    cp examples/rag_dataset.json search_engine/corpus/rag_dataset.json
fi

echo "✅ 准备工作完成"

# ============================================
# 步骤2: 建立向量索引
# ============================================
echo ""
echo "步骤2: 建立向量检索索引..."

if [ ! -d "search_engine/corpus/colqwen_ingestion" ]; then
    echo "正在生成向量索引（这可能需要几分钟）..."
    python search_engine/ingestion.py
    echo "✅ 向量索引生成完成"
else
    echo "✅ 向量索引已存在，跳过"
fi

# ============================================
# 步骤3: 启动搜索引擎
# ============================================
echo ""
echo "步骤3: 启动搜索引擎API..."

# 检查是否已启动
if curl -s http://localhost:8002/docs > /dev/null 2>&1; then
    echo "✅ 搜索引擎已在运行"
else
    echo "正在后台启动搜索引擎..."
    nohup python search_engine/search_engine_api.py > search_engine.log 2>&1 &

    # 等待API启动
    echo "等待API启动..."
    for i in {1..30}; do
        if curl -s http://localhost:8002/docs > /dev/null 2>&1; then
            echo "✅ 搜索引擎启动成功"
            break
        fi
        sleep 2
    done
fi

# ============================================
# 步骤4: 配置API密钥
# ============================================
echo ""
echo "步骤4: 检查API密钥..."

if [ -z "$DASH_SCOPE_KEY" ]; then
    echo "❌ 错误: 请设置DASH_SCOPE_KEY环境变量"
    echo "   export DASH_SCOPE_KEY='sk-xxxxxxxxxxxxx'"
    exit 1
else
    echo "✅ API密钥已配置"
fi

# ============================================
# 步骤5: 生成轨迹数据（核心步骤）
# ============================================
echo ""
echo "步骤5: 生成训练轨迹数据..."
echo "⚠️  这一步会调用阿里云API，将产生费用"
echo "按Enter继续，或Ctrl+C取消..."
read

mkdir -p search_engine/corpus/results

python scripts/data_construct_pipeline.py \
  --dataset=search_engine/corpus \
  --query_file=rag_dataset.json \
  --experiment_type=cot \
  --workers_num=1 \
  --topk=10

echo "✅ 轨迹数据生成完成: search_engine/corpus/results/cot_crop.jsonl"

# ============================================
# 步骤6: 转换为SFT格式
# ============================================
echo ""
echo "步骤6: 转换为SFT训练格式..."

# 修改转换脚本的路径
sed -i.bak "s|file_name_raw = 'Path to the raw data'|file_name_raw = 'search_engine/corpus/results/cot_crop.jsonl'|g" scripts/cot_convert_sft.py

mkdir -p data
python scripts/cot_convert_sft.py

echo "✅ SFT数据生成完成: data/search_sft_w_crop.json"

# ============================================
# 步骤7: 转换为Parquet（可选）
# ============================================
echo ""
echo "步骤7: 是否转换为Parquet格式？(y/n)"
read -r response

if [ "$response" = "y" ]; then
    echo "正在转换为Parquet格式..."
    python3 << 'EOF'
import sys
sys.path.append('.')
from scripts.hf_dataset_convert import convert_dataset

USER_PROMPT = '''Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and user will return the searched results. Every time you retrieve an image, you have the option to crop it to obtain a clearer view, the format for coordinates is <bbox>[x1, y1, x2, y2]</bbox>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}'''

convert_dataset(
    USER_PROMPT,
    ['./data/search_sft_w_crop.json'],
    ['search_sft'],
    'search_sft_final'
)
EOF
    echo "✅ Parquet数据生成完成: data/search_sft_final.parquet"
fi

# ============================================
# 完成
# ============================================
echo ""
echo "========================================"
echo "✅ SFT数据集生成完成！"
echo "========================================"
echo ""
echo "生成的文件："
echo "  - 轨迹数据: search_engine/corpus/results/cot_crop.jsonl"
echo "  - SFT格式: data/search_sft_w_crop.json"
if [ "$response" = "y" ]; then
    echo "  - Parquet格式: data/search_sft_final.parquet"
fi
echo ""
echo "下一步："
echo "  使用这些数据训练模型:"
echo "  bash train_grpo_qwen2_5_vl_7b.sh"
echo ""
