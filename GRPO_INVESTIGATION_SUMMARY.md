# GRPO机制调查总结

## 研究问题

1. **GRPO是如何更新的？**
2. **是整个推理链结束后计算log，还是每一步计算一次log？**
3. **每一步都有reward标量吗？如何计算advantage？**
4. **100个token但只有1个reward如何处理？**

---

## 核心发现

### 1. 更新机制：整个推理链结束后一次性计算

**代码位置**: `verl/trainer/ppo/ray_trainer.py:786-790`

```python
# 生成完成后，才计算log概率
with _timer('old_log_prob', timing_raw):
    with torch.no_grad():
        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
```

**关键点**:
- ✅ 多步推理：最多4轮交互（search → obs → bbox → obs → answer）
- ✅ 一次计算log：整个序列生成完毕后，一次forward pass计算所有token的log概率
- ✅ 高效设计：生成用vLLM推理引擎，训练用完整模型

### 2. Reward分配：只有最后一个token有reward

**代码位置**: `verl/workers/reward_manager/rm.py:336`

```python
# 只给最后一个有效token赋值
reward_tensor[i, valid_response_length - 1] = score
```

**实例**:
```python
# 对于100个token的response
token_level_rewards = [0.0, 0.0, 0.0, ..., 0.0, 0.7]
#                      ↑────── 99个0 ──────↑  ↑最后1个
```

### 3. Sum操作：Token-level → Trajectory-level

**代码位置**: `verl/trainer/ppo/core_algos.py:131`

```python
# 关键的sum操作！
scores = token_level_rewards.sum(dim=-1)
# 输入: (batch_size, response_length)
# 输出: (batch_size,)
```

**作用**:
- 将变长的token序列压缩成单个标量
- 支持多种reward分配策略（outcome/process/混合）
- 高效的向量化计算

### 4. GRPO Advantage计算

**代码位置**: `verl/trainer/ppo/core_algos.py:111-154`

```python
def compute_grpo_outcome_advantage(token_level_rewards, eos_mask, index):
    # 步骤1: Sum
    scores = token_level_rewards.sum(dim=-1)

    # 步骤2: Group Normalization
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        id2mean[idx] = mean(id2score[idx])
        id2std[idx] = std(id2score[idx])

    # 步骤3: 归一化
    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

    # 步骤4: Tile（广播到每个token）
    advantages = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return advantages
```

---

## 真实案例演示

### 场景
- **问题**: "2023年公司的营收是多少？"
- **参考答案**: "5.2亿美元"
- **同一prompt生成5个responses**

### Response示例

| ID | Tokens | 交互 | 最终答案 | Reward | Advantage |
|----|--------|------|----------|--------|-----------|
| 1  | 1540   | 3轮  | 5.2亿美元 | 1.00   | +0.880    |
| 2  | 1200   | 2轮  | 5.2亿美元 | 1.00   | +0.880    |
| 3  | 1300   | 3轮  | 5.5亿美元 | 0.23   | -0.937    |
| 4  | 900    | 2轮  | None     | 0.00   | -1.468    |
| 5  | 1600   | 4轮  | 5.2亿美元 | 0.90   | +0.645    |

### 完整流程

```
┌──────────────────────────────────────────────┐
│ 1. 多步生成（vLLM推理引擎）                  │
│    Response 1: search → bbox → answer        │
│    Response 2: search → answer               │
│    ...                                       │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ 2. 计算Log Probs（一次forward pass）         │
│    log_probs[0] = [1540个值]                 │
│    log_probs[1] = [1200个值]                 │
│    ...                                       │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ 3. 计算Rewards（评估系统）                   │
│    token_rewards[0] = [0,0,...,0,1.00]       │
│    token_rewards[1] = [0,0,...,0,1.00]       │
│    token_rewards[2] = [0,0,...,0,0.23]       │
│    token_rewards[3] = [0,0,...,0,0.00]       │
│    token_rewards[4] = [0,0,...,0,0.90]       │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ 4. Sum操作（关键！）                         │
│    scores = [1.00, 1.00, 0.23, 0.00, 0.90]   │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ 5. GRPO归一化                                 │
│    mean = 0.625, std = 0.426                 │
│    advantages = [+0.880, +0.880, -0.937,     │
│                  -1.468, +0.645]             │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ 6. Tile广播（每个token获得相同advantage）    │
│    adv[0] = [0.880, 0.880, ..., 0.880]       │
│    adv[1] = [0.880, 0.880, ..., 0.880]       │
│    adv[2] = [-0.937, -0.937, ..., -0.937]    │
│    ...                                       │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ 7. PPO Loss & 梯度更新                        │
│    好的responses (1,2,5): 增加概率 ↑        │
│    差的responses (3,4): 降低概率 ↓          │
└──────────────────────────────────────────────┘
```

---

## 关键洞察

### 1. Sum操作的设计哲学

**为什么使用sum？**

✅ **灵活性**: 支持多种reward分配
- Outcome supervision: `[0,0,...,R]` → R
- Process supervision: `[r1,r2,...,rn]` → Σri
- 混合模式: `[0,r1,0,r2,...,R]` → r1+r2+R

✅ **简洁性**: 一行代码处理所有情况
```python
scores = token_level_rewards.sum(dim=-1)
```

✅ **效率**: GPU向量化计算，高度并行

✅ **数学一致性**: 符合GRPO论文定义

### 2. GRPO vs GAE

| 特性 | GRPO | GAE (PPO) |
|------|------|-----------|
| Value估计 | ❌ 不需要 | ✅ 需要critic |
| Advantage | outcome-based | bootstrapping |
| 每个token的A | 相同 | 不同 |
| Sum操作 | ✅ 核心操作 | ❌ 不适用 |
| 适用场景 | 多步推理 | 连续控制 |

### 3. 信用分配（Credit Assignment）

**关键问题**: 100个token但只有1个reward，如何分配？

**GRPO的答案**:
- 不区分每个token的贡献
- 整个trajectory作为一个单元评估
- 所有token共享相同的advantage
- 适合outcome supervision（只看最终结果）

**示例**:
```python
# Response 1: 1540个token，最终正确
advantage = [+0.880, +0.880, +0.880, ..., +0.880]  # 全部鼓励

# Response 3: 1300个token，最终错误
advantage = [-0.937, -0.937, -0.937, ..., -0.937]  # 全部惩罚
```

---

## 代码关键位置

### 1. 多步推理生成
- **文件**: `vrag_agent/generation.py`
- **函数**: `run_llm_loop()` (Line 372-511)
- **特点**: 最多4轮交互，支持search/bbox/answer操作

### 2. Log概率计算
- **文件**: `verl/workers/actor/dp_actor.py`
- **函数**: `compute_log_prob()` (Line 171-227)
- **特点**: 一次forward pass计算所有token

### 3. Reward计算
- **文件**: `verl/workers/reward_manager/rm.py`
- **函数**: `__call__()` (Line 210-348)
- **特点**: 格式(0.1) + 答案(0.7) + NDCG(0.2)

### 4. GRPO Advantage
- **文件**: `verl/trainer/ppo/core_algos.py`
- **函数**: `compute_grpo_outcome_advantage()` (Line 111-154)
- **特点**: Sum → Normalize → Tile

### 5. PPO Loss
- **文件**: `verl/workers/actor/dp_actor.py`
- **函数**: `update_policy()` (Line 229-328)
- **特点**: PPO clip，支持多modal inputs

---

## 演示文件

本次调查创建了以下演示文件：

### 1. 概念演示
- **`grpo_sum_example_pure_python.py`**: Sum操作的详细演示
  - 展示token-level → trajectory-level的转换
  - 纯Python实现，易于理解
  - 可直接运行

### 2. 实战案例
- **`grpo_real_example.py`**: 真实场景的完整模拟
  - 5个responses的完整流程
  - 实际的reward计算和GRPO归一化
  - 包含PPO loss计算示例
  - 可直接运行

- **`grpo_complete_example.md`**: 详细的文字描述
  - 从问题到训练的全流程
  - 每个步骤的token数量和计算细节
  - 适合深入学习

- **`grpo_visual_flow.txt`**: ASCII可视化流程图
  - 清晰的阶段划分
  - 数据流动的可视化
  - 适合快速理解

---

## 总结

### GRPO的核心机制（5步）

1. **多步推理**: 4轮交互生成完整response
2. **一次计算log**: 整个序列完成后forward pass
3. **Sum聚合**: `token_rewards[0,...,0,R]` → `R`
4. **GRPO归一化**: 同prompt的多个responses互相比较
5. **Tile广播**: trajectory-level advantage → token-level

### 设计优势

✅ **高效**: 生成用vLLM，训练用完整模型
✅ **灵活**: Sum操作支持多种reward策略
✅ **稳定**: Group normalization减少方差
✅ **简单**: 不需要critic网络
✅ **适用**: 完美匹配多步推理任务

### 最终效果

经过GRPO训练，模型学会：
- ✅ 正确使用`<search>`、`<bbox>`、`<answer>`标签
- ✅ 检索相关的页面和区域
- ✅ 生成准确的答案
- ✅ 在2-3个turns内高效完成任务

**结论**: GRPO通过outcome-based的方式，让模型学会完整的多步推理能力！🎯

---

## 参考资料

- GRPO论文: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- PPO原论文: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- 项目代码: `verl/trainer/ppo/` 和 `vrag_agent/`
