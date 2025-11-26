"""
Trajectory Reward计算流程详解
"""

print("=" * 80)
print("Trajectory Reward的完整计算流程")
print("=" * 80)

print("\n在这个项目中，有两个相关的概念：")
print("  1. sequence_score   - 原始的trajectory score（未减KL）")
print("  2. sequence_reward  - 最终的trajectory reward（减去KL penalty）")

print("\n" + "=" * 80)
print("步骤1: 计算 token_level_scores")
print("=" * 80)

print("""
代码位置: verl/trainer/ppo/ray_trainer.py:819-820

    reward_tensor = self.reward_fn(batch)
    batch.batch['token_level_scores'] = reward_tensor

这是从Reward Manager得到的**原始reward**，在token level上。
""")

print("示例数据:")
print("  token_level_scores = [0, 0, 0, ..., 0, 0.7]  # 只有最后1个token有值")
print("  shape: (batch_size, response_length)")

print("\n" + "=" * 80)
print("步骤2: 减去KL Penalty → token_level_rewards")
print("=" * 80)

print("""
代码位置: verl/trainer/ppo/ray_trainer.py:137-166 (apply_kl_penalty函数)

    token_level_scores = data.batch['token_level_scores']

    # 计算KL divergence
    kld = kl_penalty(old_log_probs, ref_log_prob, ...)

    # 减去KL penalty
    token_level_rewards = token_level_scores - beta * kld

    data.batch['token_level_rewards'] = token_level_rewards

这一步是为了惩罚模型偏离参考策略太远。
""")

print("KL Penalty的作用:")
print("  • beta: KL系数（自适应调整）")
print("  • kld: 每个token的KL散度")
print("  • token_level_rewards = scores - beta * kld")

print("\n示例（假设beta=0.1）:")
print("-" * 80)

# 模拟数据
token_level_scores = [0.0] * 10 + [0.7]
kld_values = [0.02, 0.03, 0.01, 0.02, 0.04, 0.03, 0.02, 0.01, 0.03, 0.02, 0.05]
beta = 0.1

token_level_rewards = []
for score, kld in zip(token_level_scores, kld_values):
    reward = score - beta * kld
    token_level_rewards.append(reward)

print("Token | score | kld   | beta*kld | reward")
print("-" * 60)
for i in range(len(token_level_scores)):
    print(f"  {i:2d}  | {token_level_scores[i]:5.2f} | {kld_values[i]:5.3f} | {beta * kld_values[i]:8.4f} | {token_level_rewards[i]:6.4f}")

print("\n注意：")
print("  • 前10个token: score=0, 但reward是负数（因为KL penalty）")
print("  • 最后1个token: score=0.7, reward=0.7-0.005=0.695")

print("\n如果不使用KL penalty (use_kl_loss=True):")
print("  token_level_rewards = token_level_scores  # 直接相等")

print("\n" + "=" * 80)
print("步骤3: Sum操作 → Trajectory-level")
print("=" * 80)

print("""
代码位置: verl/trainer/ppo/metric_utils.py:48-49

    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

通过sum操作，将token-level转为trajectory-level（也叫sequence-level）。
""")

print("计算示例:")
print("-" * 80)

sequence_score = sum(token_level_scores)
sequence_reward = sum(token_level_rewards)

print(f"  sequence_score  = sum(token_level_scores)")
print(f"                  = sum([0, 0, ..., 0, 0.7])")
print(f"                  = {sequence_score:.2f}")
print()
print(f"  sequence_reward = sum(token_level_rewards)")
print(f"                  = sum([-0.002, -0.003, ..., -0.005, 0.695])")
print(f"                  = {sequence_reward:.4f}")

print("\n区别:")
print("  • sequence_score:  只考虑任务完成度")
print("  • sequence_reward: 任务完成度 - KL惩罚")

print("\n" + "=" * 80)
print("完整流程可视化")
print("=" * 80)

print("""
┌────────────────────────────────────────────────────────────────┐
│ 1. Reward Manager                                              │
│    ↓                                                            │
│    token_level_scores = [0, 0, 0, ..., 0, 0.7]                │
│    (只考虑任务完成质量)                                         │
└────────────────────────────────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────────────┐
│ 2. Apply KL Penalty                                            │
│    ↓                                                            │
│    kld = compute_kl_divergence(old_log_probs, ref_log_probs)  │
│    token_level_rewards = token_level_scores - beta * kld      │
│                                                                │
│    结果: [-0.002, -0.003, ..., -0.005, 0.695]                 │
│    (减去了模型偏离参考策略的惩罚)                              │
└────────────────────────────────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────────────┐
│ 3. Sum操作 (Token → Trajectory)                                │
│    ↓                                                            │
│    sequence_score  = 0.7    (未减KL)                          │
│    sequence_reward = 0.67   (减去KL)                          │
│                                                                │
│    这就是trajectory-level的reward!                            │
└────────────────────────────────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────────────┐
│ 4. GRPO Advantage计算                                          │
│    ↓                                                            │
│    使用 sequence_reward 进行归一化                             │
│    advantage = (sequence_reward - mean) / std                 │
└────────────────────────────────────────────────────────────────┘
""")

print("\n" + "=" * 80)
print("在GRPO中的使用")
print("=" * 80)

print("""
代码位置: verl/trainer/ppo/core_algos.py:111-154

def compute_grpo_outcome_advantage(token_level_rewards, eos_mask, index):
    # 步骤1: Sum操作（得到trajectory reward）
    scores = token_level_rewards.sum(dim=-1)

    # 这里的scores就是trajectory-level的reward
    # 对于我们的例子: scores[i] = 0.67

    # 步骤2: 基于同一prompt的多个responses归一化
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    # 计算均值和标准差
    for idx in id2score:
        id2mean[idx] = mean(id2score[idx])
        id2std[idx] = std(id2score[idx])

    # 步骤3: 归一化得到advantage
    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

    # 步骤4: Tile到每个token
    advantages = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return advantages, scores
""")

print("\n" + "=" * 80)
print("术语对照")
print("=" * 80)

print("""
┌──────────────────────┬────────────────────────────────────────┐
│   代码中的名称       │              含义                      │
├──────────────────────┼────────────────────────────────────────┤
│ token_level_scores   │ 原始reward (token level)              │
│ token_level_rewards  │ 减去KL后的reward (token level)        │
│ sequence_score       │ 原始reward总和 (trajectory level)     │
│ sequence_reward      │ 最终reward总和 (trajectory level)     │
│                      │ = trajectory reward ← 你问的这个!     │
└──────────────────────┴────────────────────────────────────────┘
""")

print("\n" + "=" * 80)
print("关键代码位置总结")
print("=" * 80)

print("""
1. 计算token_level_scores:
   verl/workers/reward_manager/rm.py:210-348
   verl/trainer/ppo/ray_trainer.py:819-820

2. 计算token_level_rewards (减KL):
   verl/trainer/ppo/ray_trainer.py:137-166 (apply_kl_penalty函数)

3. 计算sequence_reward (sum):
   verl/trainer/ppo/metric_utils.py:48-49

4. 使用trajectory reward计算advantage:
   verl/trainer/ppo/core_algos.py:131 (sum操作)
   verl/trainer/ppo/ray_trainer.py:187-197 (GRPO分支)
""")

print("\n" + "=" * 80)
print("总结")
print("=" * 80)

print("""
Trajectory Reward的计算公式：

  trajectory_reward = sum(token_level_rewards)
                    = sum(token_level_scores - beta * kld)
                    = sum(token_level_scores) - beta * sum(kld)
                    = sequence_score - beta * total_kl

简化理解：
  trajectory_reward = 任务完成质量 - 偏离参考策略的惩罚

在GRPO的outcome supervision设置下：
  • token_level_scores: 只有最后1个token非零
  • sum操作提取这个唯一值，并减去所有token的KL penalty总和
  • 得到的trajectory_reward用于GRPO归一化
""")

print("=" * 80)
