"""
演示GRPO如何通过sum操作将token-level reward转为trajectory-level score
"""

import numpy as np
from collections import defaultdict

print("=" * 80)
print("GRPO的Sum操作详解")
print("=" * 80)

# ==================== 模拟数据 ====================
batch_size = 3
response_length = 5  # 为了演示简化为5个token

# 创建token_level_rewards张量
# Shape: (batch_size, response_length)
token_level_rewards = np.zeros((batch_size, response_length), dtype=np.float32)

# 模拟reward分配：只有最后一个有效token有reward
# Response 0: 长度5, reward = 0.8
token_level_rewards[0, 4] = 0.8

# Response 1: 长度3 (有2个padding), reward = 0.6
token_level_rewards[1, 2] = 0.6

# Response 2: 长度5, reward = 1.0
token_level_rewards[2, 4] = 1.0

print("\n【步骤1】Token-level rewards (batch_size=3, response_length=5)")
print("每行代表一个response，每列代表一个token位置")
print(token_level_rewards)
print(f"Shape: {token_level_rewards.shape}")

# ==================== Sum操作 ====================
print("\n【步骤2】执行 .sum(axis=-1) 操作")
print("axis=-1 表示沿着最后一个维度（response_length）求和")

scores = token_level_rewards.sum(axis=-1)

print(f"\nScores shape: {scores.shape}")  # (3,)
print(f"Scores值: {scores}")

print("\n计算过程展示：")
for i in range(batch_size):
    print(f"  Response {i}: sum({token_level_rewards[i].tolist()}) = {scores[i]:.1f}")

# ==================== 可视化 ====================
print("\n" + "=" * 80)
print("可视化对比")
print("=" * 80)

print("\nToken-level (输入):")
print("  Response 0: [0.0, 0.0, 0.0, 0.0, 0.8] ──sum──> 0.8")
print("  Response 1: [0.0, 0.0, 0.6, 0.0, 0.0] ──sum──> 0.6")
print("  Response 2: [0.0, 0.0, 0.0, 0.0, 1.0] ──sum──> 1.0")

print("\n                    ↓")
print("              sum(dim=-1)")
print("                    ↓")

print("\nTrajectory-level (输出):")
print("  [0.8, 0.6, 1.0]")

# ==================== 实际代码中的完整流程 ====================
print("\n" + "=" * 80)
print("完整的GRPO Advantage计算流程")
print("=" * 80)

# 假设这3个response来自同一个prompt
index = np.array([0, 0, 0])  # 相同的prompt_id
eos_mask = np.array([
    [1, 1, 1, 1, 1],  # Response 0: 全部有效
    [1, 1, 1, 0, 0],  # Response 1: 前3个有效
    [1, 1, 1, 1, 1],  # Response 2: 全部有效
], dtype=np.float32)

print("\n【步骤3】Scores归一化 (GRPO核心)")

id2score = defaultdict(list)
id2mean = {}
id2std = {}

# 收集同一prompt的所有scores
for i in range(batch_size):
    id2score[index[i]].append(scores[i])

print(f"\n同一prompt的所有scores: {id2score[0]}")

# 计算均值和标准差
for idx in id2score:
    id2mean[idx] = np.mean(id2score[idx])
    id2std[idx] = np.std(id2score[idx])

print(f"均值 (mean): {id2mean[0]:.4f}")
print(f"标准差 (std): {id2std[0]:.4f}")

# 归一化
normalized_scores = np.zeros(batch_size)
epsilon = 1e-6
for i in range(batch_size):
    normalized_scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

print(f"\n归一化后的scores: {normalized_scores}")
print("\n归一化计算过程：")
for i in range(batch_size):
    print(f"  Response {i}: ({scores[i]:.1f} - {id2mean[0]:.4f}) / {id2std[0]:.4f} = {normalized_scores[i]:.4f}")

print("\n【步骤4】广播到每个token")

# 将标量advantage扩展到每个token
# normalized_scores.reshape(-1, 1) 将 (3,) 变成 (3, 1)
# np.tile(..., [1, response_length]) 将每一行复制5次
advantages = np.tile(normalized_scores.reshape(-1, 1), [1, response_length]) * eos_mask

print(f"\nAdvantages shape: {advantages.shape}")  # (3, 5)
print("\nAdvantages (每个token的advantage):")
print(advantages)

print("\n解释：")
for i in range(batch_size):
    valid_count = int(eos_mask[i].sum())
    print(f"  Response {i}: 前{valid_count}个token都得到advantage={normalized_scores[i]:.4f}")

# ==================== 更真实的例子 ====================
print("\n" + "=" * 80)
print("更真实的多步推理例子（100 tokens）")
print("=" * 80)

# 模拟一个100 token的response
real_response_length = 100
real_token_rewards = np.zeros(real_response_length)
real_token_rewards[99] = 0.7  # 只有最后一个token有reward

print(f"\nToken rewards: [{', '.join(['0.0']*99)}, 0.7]")
print(f"长度: {len(real_token_rewards)} tokens")

real_score = real_token_rewards.sum()
print(f"\nSum操作: sum([0, 0, ..., 0, 0.7]) = {real_score}")

print("\n这意味着：")
print("  - 虽然有100个token，但sum操作正确地提取出唯一的reward值")
print("  - 如果未来每10个token都有reward，sum操作同样适用")
print("  - 这就是sum操作的灵活性！")

# ==================== 为什么这样设计？ ====================
print("\n" + "=" * 80)
print("为什么使用Sum操作？设计思想解析")
print("=" * 80)

print("""
1. 【灵活性】支持多种reward分配策略：

   Outcome Supervision (当前实现):
   token_rewards = [0, 0, 0, ..., 0, R]
   sum = R ✓

   Process Supervision (可能的扩展):
   token_rewards = [r1, r2, r3, ..., rn]
   sum = r1 + r2 + ... + rn ✓

   混合模式:
   token_rewards = [0, r1, 0, 0, r2, ..., 0, R]
   sum = r1 + r2 + R ✓

2. 【代码简洁】一行代码搞定：

   scores = token_level_rewards.sum(dim=-1)

   不需要：
   - 手动找最后一个token的位置
   - 区分不同的reward分配策略
   - 写if-else逻辑

3. 【数学意义】符合GRPO的设计哲学：

   GRPO (Group Relative Policy Optimization):
   - 将整个trajectory视为一个单元
   - 基于最终结果（outcome）优化
   - Sum自然地聚合整个trajectory的价值

4. 【向量化计算】高效并行：

   scores = token_level_rewards.sum(dim=-1)  # GPU一次性处理所有样本
   vs.
   for i in range(batch_size):
       scores[i] = find_last_reward(token_rewards[i])  # 慢！
""")

# ==================== PyTorch实际代码 ====================
print("\n" + "=" * 80)
print("PyTorch中的实际代码")
print("=" * 80)

print("""
# 来自 core_algos.py:111-154

def compute_grpo_outcome_advantage(token_level_rewards, eos_mask, index, epsilon=1e-6):
    response_length = token_level_rewards.shape[-1]

    # 关键：sum操作！
    scores = token_level_rewards.sum(dim=-1)
    # 输入: (batch_size, response_length)
    # 输出: (batch_size,)

    # 后续的GRPO归一化...
    id2score = defaultdict(list)
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    # ... (归一化逻辑)

    # 最后广播回token维度
    advantages = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return advantages, returns
""")

print("\n" + "=" * 80)
print("演示完成！")
print("=" * 80)

print("\n总结：")
print("Sum操作是连接token-level和trajectory-level的桥梁")
print("  Token-level:  [0.0, 0.0, ..., 0.0, 0.7]  (100维向量)")
print("       ↓ sum(dim=-1)")
print("  Trajectory:   0.7                         (标量)")
print("       ↓ 归一化 & tile")
print("  Advantages:   [A, A, A, ..., A, A]        (100维向量，每个元素相同)")
