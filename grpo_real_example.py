"""
GRPO真实案例模拟
展示从问题到训练的完整流程
"""

from collections import defaultdict
import math

print("=" * 80)
print("GRPO实战案例：多步RAG推理")
print("=" * 80)

# ==================== 场景设置 ====================
print("\n【场景】")
print("问题: '2023年公司的营收是多少？'")
print("参考答案: '5.2亿美元'")
print("文档: company_report_2023.pdf")
print("\n同一个prompt生成5个不同的responses")

# ==================== Response生成 ====================
print("\n" + "=" * 80)
print("步骤1: 多步推理生成（5个responses）")
print("=" * 80)

responses_info = [
    {
        "id": 1,
        "turns": 3,
        "interactions": [
            ("search", "2023年公司年报"),
            ("bbox", "[100,200,300,400]"),
            ("answer", "5.2亿美元")
        ],
        "total_tokens": 1540,
        "final_answer": "5.2亿美元",
        "retrieved_pages": ["page_5"],
        "reference_pages": ["page_5"]
    },
    {
        "id": 2,
        "turns": 2,
        "interactions": [
            ("search", "财务数据2023"),
            ("answer", "5.2亿美元")
        ],
        "total_tokens": 1200,
        "final_answer": "5.2亿美元",
        "retrieved_pages": ["page_5"],
        "reference_pages": ["page_5"]
    },
    {
        "id": 3,
        "turns": 3,
        "interactions": [
            ("search", "营收"),
            ("bbox", "[50,100,200,300]"),
            ("answer", "5.5亿美元")  # 错误！
        ],
        "total_tokens": 1300,
        "final_answer": "5.5亿美元",
        "retrieved_pages": ["page_3", "page_5"],
        "reference_pages": ["page_5"]
    },
    {
        "id": 4,
        "turns": 2,
        "interactions": [
            ("search", "2023数据"),
            ("think", "我认为是...")  # 忘记用<answer>标签！
        ],
        "total_tokens": 900,
        "final_answer": None,  # 格式错误
        "retrieved_pages": ["page_2"],
        "reference_pages": ["page_5"]
    },
    {
        "id": 5,
        "turns": 4,
        "interactions": [
            ("search", "公司年报"),
            ("bbox", "[0,0,100,100]"),
            ("search", "营收数据"),
            ("answer", "5.2亿美元")
        ],
        "total_tokens": 1600,
        "final_answer": "5.2亿美元",
        "retrieved_pages": ["page_2", "page_3", "page_5"],
        "reference_pages": ["page_5"]
    }
]

for resp in responses_info:
    print(f"\nResponse {resp['id']}:")
    print(f"  交互轮数: {resp['turns']}")
    print(f"  操作序列: {' → '.join([op for op, _ in resp['interactions']])}")
    print(f"  总Token数: {resp['total_tokens']}")
    print(f"  最终答案: {resp['final_answer']}")

# ==================== 计算Rewards ====================
print("\n" + "=" * 80)
print("步骤2: 计算每个Response的Reward")
print("=" * 80)

def calculate_anls(pred, ref):
    """简化的ANLS计算"""
    if pred == ref:
        return 1.0
    return 0.0

def calculate_ndcg(retrieved, reference):
    """简化的NDCG计算"""
    relevance = [1 if page in reference else 0 for page in retrieved]
    if sum(relevance) == 0:
        return 0.0
    dcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevance))
    idcg = sum((2**1 - 1) / math.log2(i + 2) for i in range(len(reference)))
    return dcg / idcg if idcg > 0 else 0.0

rewards = []
print("\n公式: reward = 0.7 * answer_match + 0.1 * format + 0.2 * ndcg\n")

for resp in responses_info:
    # 格式检查
    has_answer = resp['final_answer'] is not None
    format_score = 1.0 if has_answer else 0.0

    # 答案正确性
    answer_match = 0.0
    if has_answer:
        answer_match = calculate_anls(resp['final_answer'], "5.2亿美元")

    # 检索质量
    ndcg = 0.0
    if has_answer:
        ndcg = calculate_ndcg(resp['retrieved_pages'], resp['reference_pages'])

    # 最终reward
    final_reward = 0.7 * answer_match + 0.1 * format_score + 0.2 * ndcg

    rewards.append(final_reward)

    print(f"Response {resp['id']}:")
    print(f"  格式正确: {format_score:.1f}")
    print(f"  答案匹配: {answer_match:.1f}")
    print(f"  检索质量(NDCG): {ndcg:.2f}")
    print(f"  ➜ 最终Reward: {final_reward:.2f}")

# ==================== Token-level Rewards ====================
print("\n" + "=" * 80)
print("步骤3: 构建Token-level Rewards")
print("=" * 80)

print("\n每个response的token_level_rewards是一个向量，只有最后一个位置有值：\n")

token_level_rewards = []
for i, resp in enumerate(responses_info):
    # 创建全0向量
    token_rewards = [0.0] * resp['total_tokens']
    # 最后一个token赋值
    token_rewards[-1] = rewards[i]
    token_level_rewards.append(token_rewards)

    print(f"Response {resp['id']}: [0, 0, 0, ..., 0, {rewards[i]:.2f}]  (长度: {resp['total_tokens']})")

# ==================== Sum操作 ====================
print("\n" + "=" * 80)
print("步骤4: Sum操作 - Token-level → Trajectory-level")
print("=" * 80)

scores = [sum(tr) for tr in token_level_rewards]

print("\n执行 sum(dim=-1) 操作：\n")
for i, (resp, score) in enumerate(zip(responses_info, scores)):
    print(f"Response {resp['id']}: sum([0, 0, ..., 0, {rewards[i]:.2f}]) = {score:.2f}")

print(f"\nTrajectory-level scores: {[f'{s:.2f}' for s in scores]}")

# ==================== GRPO归一化 ====================
print("\n" + "=" * 80)
print("步骤5: GRPO归一化（Group Relative）")
print("=" * 80)

# 统计
mean = sum(scores) / len(scores)
variance = sum((s - mean) ** 2 for s in scores) / len(scores)
std = math.sqrt(variance)

print(f"\n同一prompt的5个responses:")
print(f"  均值(μ): {mean:.3f}")
print(f"  标准差(σ): {std:.3f}")

# 归一化
normalized_scores = [(s - mean) / (std + 1e-6) for s in scores]

print("\n归一化计算 (score - μ) / σ:\n")
for i, (resp, score, norm) in enumerate(zip(responses_info, scores, normalized_scores)):
    symbol = "+" if norm >= 0 else ""
    quality = "✓ 好!" if norm > 0.5 else ("⚠ 一般" if norm > -0.5 else "✗ 差!")
    print(f"Response {resp['id']}: ({score:.2f} - {mean:.3f}) / {std:.3f} = {symbol}{norm:.3f}  {quality}")

print(f"\nNormalized advantages: {[f'{n:.3f}' for n in normalized_scores]}")

# ==================== 广播到Token ====================
print("\n" + "=" * 80)
print("步骤6: 广播到每个Token (Tile操作)")
print("=" * 80)

print("\n将标量advantage复制到每个token:\n")
advantages = []
for i, resp in enumerate(responses_info):
    adv = [normalized_scores[i]] * resp['total_tokens']
    advantages.append(adv)

    print(f"Response {resp['id']}: [{normalized_scores[i]:.3f}, {normalized_scores[i]:.3f}, ..., {normalized_scores[i]:.3f}]")
    print(f"             (长度: {resp['total_tokens']} tokens，每个值都是 {normalized_scores[i]:.3f})")

# ==================== PPO Loss示例 ====================
print("\n" + "=" * 80)
print("步骤7: PPO Loss计算示例")
print("=" * 80)

print("\n假设计算第1个response的前10个token的loss:\n")

# 模拟old_log_probs和new_log_probs
old_log_probs_sample = [-0.5, -0.3, -0.8, -0.4, -0.6, -0.7, -0.5, -0.9, -0.4, -0.6]
new_log_probs_sample = [-0.48, -0.29, -0.75, -0.38, -0.58, -0.68, -0.52, -0.88, -0.39, -0.62]

print("Token | old_log_p | new_log_p | ratio  | advantage | pg_loss")
print("-" * 70)

pg_losses = []
for t in range(10):
    ratio = math.exp(new_log_probs_sample[t] - old_log_probs_sample[t])
    adv = normalized_scores[0]  # Response 1的advantage

    # PPO clip
    clipped_ratio = max(0.9, min(1.1, ratio))
    pg_loss1 = -adv * ratio
    pg_loss2 = -adv * clipped_ratio
    pg_loss = max(pg_loss1, pg_loss2)
    pg_losses.append(pg_loss)

    print(f"  {t:2d}  | {old_log_probs_sample[t]:6.2f}   | {new_log_probs_sample[t]:6.2f}   | {ratio:.3f} | {adv:7.3f}  | {pg_loss:7.3f}")

avg_loss = sum(pg_losses) / len(pg_losses)
print(f"\n平均loss (前10个token): {avg_loss:.3f}")

# ==================== 总结 ====================
print("\n" + "=" * 80)
print("完整流程总结")
print("=" * 80)

print("""
1. 生成阶段 (vLLM推理引擎):
   ✓ 5个responses通过多轮交互生成
   ✓ 每个response包含多个turns (search/bbox/answer)
   ✓ Token数量: 900~1600不等

2. Log Prob计算 (Actor模型):
   ✓ 生成完成后，一次性计算所有token的log_prob
   ✓ 使用训练模型进行forward pass

3. Reward计算 (评估系统):
   ✓ 格式检查: 是否有<answer>标签
   ✓ 答案匹配: 调用LLM judge或ANLS
   ✓ 检索质量: NDCG评分
   ✓ 加权组合得到最终reward

4. GRPO Advantage (核心算法):
   ✓ Sum: token-level → trajectory-level
   ✓ 归一化: 基于同一prompt的多个responses
   ✓ Tile: 每个token获得相同的advantage

5. PPO Loss & 更新:
   ✓ 好的responses (高advantage): 增加生成概率
   ✓ 差的responses (低advantage): 降低生成概率
   ✓ Clip机制防止更新过大
""")

print("\n【关键洞察】")
print("-" * 80)
for i, (resp, score, norm) in enumerate(zip(responses_info, scores, normalized_scores)):
    if norm > 0:
        effect = "增加这种模式的概率 ↑"
    else:
        effect = "降低这种模式的概率 ↓"

    print(f"Response {resp['id']}: score={score:.2f}, adv={norm:+.3f} → {effect}")
    if resp['final_answer'] != "5.2亿美元":
        print(f"           原因: {resp['final_answer'] if resp['final_answer'] else '格式错误'}")

print("\n结果: 模型学会生成更准确、更高质量的多步推理！🎯")
print("=" * 80)
