# GRPO完整实例：从问题到训练

## 场景设置

**问题**: "2023年公司的营收是多少？"
**参考答案**: "5.2亿美元"
**文档**: company_report_2023.pdf（包含财务数据的PDF）

---

## 第1步：初始Prompt生成

系统为同一个问题采样 **n_agent=5** 个responses（来自同一个prompt）

```python
# 初始prompt（所有5个response共享）
prompt = """
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
Question: 2023年公司的营收是多少？
Please search for information if needed.
<|im_end|>
<|im_start|>assistant
"""

# prompt的token长度: 假设50 tokens
prompt_ids = [151644, 8948, ..., 271]  # shape: (50,)
```

---

## 第2步：多步推理生成（max_turns=4）

让我们详细看 **Response 1** 的生成过程：

### Turn 1: 第一次生成

**模型输出**:
```
<think>我需要搜索2023年公司年报</think>
<search>2023年公司年报 营收</search>
```

**Token化**:
```python
turn1_tokens = [
    # <think>我需要搜索2023年公司年报</think>
    27, 1234, 567, ..., 28,  # 15 tokens
    # <search>2023年公司年报 营收</search>
    29, 5678, 234, ..., 30   # 12 tokens
]
# turn1_length = 27 tokens
```

**环境反馈**: 调用搜索API
```python
search_result = {
    "images": ["/path/to/page_5.jpg"],  # 检索到第5页的图片
    "text": "财务报表第5页"
}

# 系统添加observation
obs1_tokens = [
    # \n<|im_start|>user\n
    198, 151644, ...,
    # <|vision_start|><image_tokens><|vision_end|>
    151652, <image_pad_tokens>, 151653,  # 图片token
    # <|im_end|>\n<|im_start|>assistant\n
    151645, 198, 151644, 271
]
# obs1_length = 850 tokens (包含图片)
```

### Turn 2: 第二次生成

**当前输入** = prompt + turn1 + obs1
```python
current_input_ids = concat([
    prompt_ids,      # 50 tokens
    turn1_tokens,    # 27 tokens
    obs1_tokens      # 850 tokens
])
# total: 927 tokens
```

**模型输出**:
```
<think>图片显示的是总览，我需要看更清楚的数据</think>
<bbox>[100, 200, 300, 400]</bbox>
```

**Token化**:
```python
turn2_tokens = [
    # <think>...</think>
    27, 4321, ..., 28,      # 20 tokens
    # <bbox>[100, 200, 300, 400]</bbox>
    31, 100, 200, ..., 32   # 10 tokens
]
# turn2_length = 30 tokens
```

**环境反馈**: 裁剪图片
```python
# 系统返回裁剪后的图片
obs2_tokens = [
    # \n<|im_start|>user\n<|vision_start|><image><|vision_end|><|im_end|>...
    198, 151644, ..., 151652, <cropped_image_tokens>, 151653, ...
]
# obs2_length = 600 tokens
```

### Turn 3: 第三次生成

**当前输入** = prompt + turn1 + obs1 + turn2 + obs2

**模型输出**:
```
<think>现在我看到了清晰的数字，2023年营收是5.2亿美元</think>
<answer>5.2亿美元</answer>
```

**Token化**:
```python
turn3_tokens = [
    # <think>现在我看到了清晰的数字，2023年营收是5.2亿美元</think>
    27, 1357, 5432, ..., 28,  # 25 tokens
    # <answer>5.2亿美元</answer>
    33, 5, 2, 1101, ..., 34   # 8 tokens
]
# turn3_length = 33 tokens
```

**环境判断**: 发现`<answer>`标签，done=True，结束生成

---

## 第3步：完整Response构建

**Response 1的完整序列**:
```python
response_1 = concat([
    turn1_tokens,  # 27 tokens
    obs1_tokens,   # 850 tokens
    turn2_tokens,  # 30 tokens
    obs2_tokens,   # 600 tokens
    turn3_tokens,  # 33 tokens
])
# total_response_length = 1540 tokens

# 完整序列 = prompt + response
full_sequence_1 = concat([
    prompt_ids,    # 50 tokens
    response_1     # 1540 tokens
])
# total: 1590 tokens
```

---

## 第4步：同一Prompt的5个Responses

假设同一个问题生成了5个不同的responses：

```python
# Response 1: 成功找到答案，3次交互
response_1_length = 1540 tokens
final_answer_1 = "5.2亿美元"  # ✓ 正确

# Response 2: 成功找到答案，2次交互（运气好）
response_2_length = 1200 tokens
final_answer_2 = "5.2亿美元"  # ✓ 正确

# Response 3: 答案错误
response_3_length = 1300 tokens
final_answer_3 = "5.5亿美元"  # ✗ 错误

# Response 4: 格式错误，没有<answer>标签
response_4_length = 900 tokens
final_answer_4 = None  # ✗ 格式错误

# Response 5: 成功但检索图片不准确
response_5_length = 1600 tokens
final_answer_5 = "5.2亿美元"  # ✓ 正确但NDCG低
```

---

## 第5步：计算Log Probabilities

**一次性计算所有token的log概率**（在生成完成后）

```python
# 对每个response，进行一次forward pass
for i in range(5):
    full_input = concat([prompt_ids, response_i])

    # 前向传播（只计算，不训练）
    with torch.no_grad():
        logits = model(full_input)  # shape: (seq_len, vocab_size)

        # 计算每个生成token的log概率
        log_probs = compute_log_prob(logits, response_i)
        # shape: (response_length_i,)

# 结果：
old_log_probs[0] = [-0.5, -0.3, -0.8, ..., -0.4]  # 1540个值
old_log_probs[1] = [-0.6, -0.2, -0.7, ..., -0.5]  # 1200个值
old_log_probs[2] = [-0.4, -0.5, -0.6, ..., -0.3]  # 1300个值
old_log_probs[3] = [-0.7, -0.4, -0.9, ..., -0.6]  # 900个值
old_log_probs[4] = [-0.5, -0.6, -0.7, ..., -0.8]  # 1600个值
```

---

## 第6步：计算Rewards

### 6.1 初始化token_level_rewards

```python
token_level_rewards = torch.zeros_like(responses)
# Response 0: shape (1540,), 全部为0
# Response 1: shape (1200,), 全部为0
# ...
```

### 6.2 计算每个response的reward

**Response 1**:
```python
# 检查格式
format_score = 1.0  # 有<search>和<answer>标签 ✓

# 检查答案正确性（调用LLM judge）
answer_match = compute_score("5.2亿美元", "5.2亿美元")
answer_match = 1.0  # ANLS = 1.0 ✓

# 检查检索质量（NDCG）
retrieved_images = ["page_5.jpg"]
reference_pages = ["page_5"]
ndcg_score = 1.0  # 完美检索 ✓

# 最终reward（加权组合）
final_reward_1 = 0.7 * 1.0 + 0.1 * 1.0 + 0.2 * 1.0 = 1.0

# 只给最后一个token赋值
token_level_rewards[0, 1539] = 1.0  # 最后一个token
# token_level_rewards[0] = [0, 0, 0, ..., 0, 1.0]
```

**Response 2**:
```python
format_score = 1.0
answer_match = 1.0
ndcg_score = 1.0
final_reward_2 = 1.0

token_level_rewards[1, 1199] = 1.0
```

**Response 3**:
```python
format_score = 1.0
answer_match = 0.0  # 答案错误 ✗
ndcg_score = 1.0
final_reward_3 = 0.7 * 0.0 + 0.1 * 1.0 + 0.2 * 1.0 = 0.3

token_level_rewards[2, 1299] = 0.3
```

**Response 4**:
```python
format_score = 0.0  # 没有<answer>标签 ✗
# 格式错误直接给0
final_reward_4 = 0.0

token_level_rewards[3, 899] = 0.0
```

**Response 5**:
```python
format_score = 1.0
answer_match = 1.0
ndcg_score = 0.5  # 检索质量一般 ⚠
final_reward_5 = 0.7 * 1.0 + 0.1 * 1.0 + 0.2 * 0.5 = 0.9

token_level_rewards[4, 1599] = 0.9
```

---

## 第7步：GRPO Advantage计算

### 7.1 Sum操作（关键！）

```python
# 将token-level转为trajectory-level
scores = token_level_rewards.sum(dim=-1)

# 计算结果：
scores[0] = 0 + 0 + ... + 0 + 1.0 = 1.0
scores[1] = 0 + 0 + ... + 0 + 1.0 = 1.0
scores[2] = 0 + 0 + ... + 0 + 0.3 = 0.3
scores[3] = 0 + 0 + ... + 0 + 0.0 = 0.0
scores[4] = 0 + 0 + ... + 0 + 0.9 = 0.9

# scores = [1.0, 1.0, 0.3, 0.0, 0.9]
```

### 7.2 GRPO归一化

因为这5个response来自同一个prompt，需要group normalization：

```python
# 计算统计量
mean = (1.0 + 1.0 + 0.3 + 0.0 + 0.9) / 5 = 0.64
std = sqrt(((1.0-0.64)² + (1.0-0.64)² + (0.3-0.64)² + (0.0-0.64)² + (0.9-0.64)²) / 5)
    = sqrt((0.1296 + 0.1296 + 0.1156 + 0.4096 + 0.0676) / 5)
    = sqrt(0.1704)
    = 0.413

# 归一化
normalized[0] = (1.0 - 0.64) / 0.413 = +0.872  # 好！
normalized[1] = (1.0 - 0.64) / 0.413 = +0.872  # 好！
normalized[2] = (0.3 - 0.64) / 0.413 = -0.823  # 差
normalized[3] = (0.0 - 0.64) / 0.413 = -1.550  # 很差！
normalized[4] = (0.9 - 0.64) / 0.413 = +0.630  # 还可以
```

### 7.3 广播到每个token

```python
# 将标量advantage复制到每个token
advantages[0] = [0.872, 0.872, 0.872, ..., 0.872]  # 1540个
advantages[1] = [0.872, 0.872, 0.872, ..., 0.872]  # 1200个
advantages[2] = [-0.823, -0.823, ..., -0.823]     # 1300个
advantages[3] = [-1.550, -1.550, ..., -1.550]     # 900个
advantages[4] = [0.630, 0.630, 0.630, ..., 0.630] # 1600个

# 注意：每个response的所有token获得相同的advantage！
```

---

## 第8步：计算Policy Loss

现在我们有了：
- `old_log_probs`: 每个token的log概率
- `advantages`: 每个token的advantage（同一response内相同）

```python
# 对Response 1的某个mini-batch计算loss
# 假设取前100个token

mini_batch_old_log_probs = old_log_probs[0, :100]  # [-0.5, -0.3, ..., -0.8]
mini_batch_advantages = advantages[0, :100]         # [0.872, 0.872, ..., 0.872]

# 重新计算log probs（训练模式）
logits = model(input_ids[:100])
new_log_probs = compute_log_prob(logits, responses[:100])
# new_log_probs = [-0.48, -0.29, ..., -0.75]

# 计算ratio
ratio = exp(new_log_probs - mini_batch_old_log_probs)
# ratio = [exp(-0.48 - (-0.5)), exp(-0.29 - (-0.3)), ...]
#       = [exp(0.02), exp(0.01), ...]
#       = [1.020, 1.010, ..., 1.053]

# PPO clip
pg_loss1 = -advantages * ratio
pg_loss2 = -advantages * clip(ratio, 0.9, 1.1)

# 例如第1个token:
pg_loss1[0] = -0.872 * 1.020 = -0.889
pg_loss2[0] = -0.872 * 1.020 = -0.889  # 没被clip

# 例如第50个token（假设ratio=1.15，超出范围）:
pg_loss1[50] = -0.872 * 1.15 = -1.003
pg_loss2[50] = -0.872 * 1.1 = -0.959   # 被clip到1.1

# 取最大值（最小的负数，即最小的loss）
pg_loss = max(pg_loss1, pg_loss2).mean()
```

---

## 第9步：实际的Loss值

让我们看看5个responses的实际loss贡献：

```python
# Response 1: advantage = +0.872 (鼓励)
loss_1 ≈ -0.872 * 1.02 = -0.889  # 负数，鼓励这个行为

# Response 2: advantage = +0.872 (鼓励)
loss_2 ≈ -0.872 * 1.01 = -0.881

# Response 3: advantage = -0.823 (惩罚)
loss_3 ≈ -(-0.823) * 0.98 = +0.807  # 正数，惩罚这个行为

# Response 4: advantage = -1.550 (严厉惩罚)
loss_4 ≈ -(-1.550) * 0.97 = +1.504  # 大正数，严厉惩罚

# Response 5: advantage = +0.630 (轻微鼓励)
loss_5 ≈ -0.630 * 1.03 = -0.649

# 总loss（平均）
total_loss = (-0.889 - 0.881 + 0.807 + 1.504 - 0.649) / 5
           = -0.022  # 接近0，因为有好有坏
```

---

## 第10步：梯度更新

```python
# 反向传播
total_loss.backward()

# 梯度更新的效果：
# ✓ Response 1, 2: 增加生成这种"正确答案"序列的概率
# ✗ Response 3, 4: 降低生成"错误答案"或"格式错误"的概率
# ⚠ Response 5: 轻微增加，但同时鼓励更好的检索

# 优化器步进
optimizer.step()
```

---

## 完整流程总结

```
┌──────────────────────────────────────────────────────────────┐
│ 输入: 问题 + PDF文档                                         │
└──────────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ 生成阶段（vLLM推理引擎）                                      │
│ • Turn 1: <search>查询</search>                              │
│ • Obs 1:  返回图片                                           │
│ • Turn 2: <bbox>裁剪</bbox>                                  │
│ • Obs 2:  返回裁剪图片                                       │
│ • Turn 3: <answer>5.2亿美元</answer>                        │
│                                                              │
│ 重复5次 → 得到5个不同的responses                            │
└──────────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ 计算Log Probs（Actor模型，一次forward pass）                 │
│ • Response 1: [1540个log_probs]                             │
│ • Response 2: [1200个log_probs]                             │
│ • ...                                                        │
└──────────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ 计算Rewards（调用评估API + 规则）                            │
│ • token_level_rewards[i, last] = final_score                │
│ • Response 1: [..., 0, 1.0]                                 │
│ • Response 2: [..., 0, 1.0]                                 │
│ • Response 3: [..., 0, 0.3]                                 │
│ • Response 4: [..., 0, 0.0]                                 │
│ • Response 5: [..., 0, 0.9]                                 │
└──────────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ GRPO Advantage计算                                           │
│ 1. Sum: [1.0, 1.0, 0.3, 0.0, 0.9]                          │
│ 2. Normalize: [+0.872, +0.872, -0.823, -1.550, +0.630]     │
│ 3. Tile: 每个token复制相同的advantage                       │
└──────────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ PPO Loss计算 & 梯度更新                                       │
│ • 好的responses (1,2,5): 增加概率 ↑                         │
│ • 差的responses (3,4): 降低概率 ↓                           │
└──────────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ 模型改进！下次更可能生成正确的多步推理                       │
└──────────────────────────────────────────────────────────────┘
```

---

## 关键洞察

1. **多步推理**：3轮交互（搜索→裁剪→回答），每步都生成token
2. **一次计算log**：1540个token的log_probs一次算完
3. **Sum的作用**：`[0,0,...,0,1.0]` → `1.0`（提取最终reward）
4. **GRPO归一化**：5个responses互相比较，相对排名决定优劣
5. **信用分配**：整个trajectory的1540个token都获得相同的+0.872 advantage

这就是GRPO在实际RAG任务中的完整工作流程！🎯
