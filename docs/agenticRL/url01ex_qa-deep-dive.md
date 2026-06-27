# url01ex：Step 1 深度问答 —— 面试向 FAQ

> 本文是对 `url01_dataloader-to-batch-explained.md` 的扩展，聚焦面试常见问题及详细答案。
> 适用场景：Agentic RL / LLM Training 实习或秋招岗位面试准备。

---

## 目录

1. [Q1: 发给 vLLM 的是什么？留在 driver 的是什么？](#q1-发给-vllm-的是什么留在-driver-的是什么)
2. [Q2: 为什么 RL 训练不按 epoch 遍历？](#q2-为什么-rl-训练不按-epoch-遍历)
3. [Q3: 每次都 iter() 不会重复采样吗？](#q3-每次都-iter-不会重复采样吗)
4. [Q4: attention_mask 就是一个打标记的吗？](#q4-attention_mask-就是一个打标记的吗)
5. [面试题库](#面试题库)
   - [Q5: 完整训练流程](#q5-请描述一个完整的-agentic-rl-训练流程)
   - [Q6: action_mask 的设计原理](#q6-action_mask-为什么设计成-action1-obs0反过来行不行)
   - [Q7: GRPO vs PPO](#q7-grpo-和-ppo-的区别为什么用-grpo)
   - [Q8: old_log_prob 的 detach](#q8-old_log_prob-为什么要重新计算为什么需要-detach)
   - [Q9: 超长序列处理](#q9-多轮-agent-交互中超长序列怎么处理)
   - [Q10: token-level mask vs 索引](#q10-为什么用-token-level-mask-而不是直接在-loss-里取-action-token-的索引)
   - [Q11: pipeline 改进方向](#q11-如果让你改进这个-agentic-rl-pipeline你会怎么做)
   - [Q12: Qwen2-VL 三维位置编码](#q12-解释一下-position_ids-为什么是-2-3-5120)

---

## Q1: 发给 vLLM 的是什么？留在 driver 的是什么？

### 核心答案

**发给 vLLM（gen_batch）**：让 vLLM "看到图片、读取 prompt"，才能生成 action token。

**留在 driver（batch）**：元数据，告诉 driver 这是哪个任务、怎么打分，不需要 tensor 数据本身。

### 发给 vLLM 的 gen_batch

```
gen_batch.batch (tensor 数据，通过 Ray RPC 序列化传输):
┌──────────────────────┬───────────────┬──────────────────────────────┐
│ 字段                  │ shape          │ 实际内容                      │
├──────────────────────┼───────────────┼──────────────────────────────┤
│ input_ids            │ (2, 5120)      │ prompt 的 token ids + PAD    │
│ attention_mask       │ (2, 5120)      │ 有效 token=1, PAD=0          │
│ position_ids         │ (2, 3, 5120)   │ Qwen2-VL 三维位置编码         │
└──────────────────────┴───────────────┴──────────────────────────────┘

gen_batch.non_tensor_batch:
  - multi_modal_data:          [PIL.Image, PIL.Image]  ← 原始图片
  - origin_multi_modal_data:   [PIL.Image, PIL.Image]  ← 原图给 sandbox
  - multi_modal_inputs:        [dict, dict]            ← vLLM 格式的图片输入
  - raw_prompt:                ["分析这张图...", ...]   ← 文本 prompt
  - raw_prompt_ids:            [[101,...], [201,...]]  ← 不等长原始 token ids
  - env_name:                  "deepeyesv2"            ← sandbox 环境类型
  - extra_info:                {...}                   ← reward function 额外信息
```

**vLLM 拿到这些后做了什么**：

```text
1. Vision Encoder 把 multi_modal_data 编码成 image embeddings
2. Text Embedding 把 input_ids 编码成 text embeddings
3. 两者拼接 → 完整的多模态序列表示
4. 自回归生成:
   → 吐出 <think>...</think>
   → 吐出 <code>```python ...```</code>   (action)
   → 停止（遇到 stop token: </code> 或 </tool_call>）
```

### 留在 driver 的 batch

```
batch.batch:  {}   ← 空了！所有 tensor 都被 pop 走了

batch.non_tensor_batch:
  - data_source:   ["perception", "perception"]  ← 数据集来源
  - reward_model:  [{"style": "rule"}, ...]      ← rule-based 还是 LLM judge
  - index:         [0, 1]                        ← 原始数据集中的索引
```

**为什么 driver 不需要 tensor**：

```text
Driver 进程 (CPU)                     vLLM Worker (GPU)
┌──────────────────────┐             ┌──────────────────────┐
│ 持有元数据             │             │ 持有 tensor 数据       │
│ 调度 & 协调            │   Ray RPC   │ 实际生成              │
│                       │ ◄─────────► │ 执行 sandbox 代码     │
│ 等待结果 → union()    │             │ 返回 gen_batch_output │
└──────────────────────┘             └──────────────────────┘

driver 是"调度中心"，不是"执行者"：
- rollout 完成后收到 gen_batch_output
- 调用 batch.union(gen_batch_output) 合并
- 触发 reward 计算
- 把完整数据发给 FSDP worker 更新参数
```

### 合并时的数据流

```text
gen_batch_output (从 vLLM 返回):
  {
    "responses":      Tensor(2, 1024),   ← 模型生成的 token ids
    "action_mask":    Tensor(2, 5120),   ← 全长 mask (action=1, obs=0, pad=0)
    "attention_mask": Tensor(2, 5120),   ← 扩展后的完整 attention mask
    "env_reward":     Tensor(2, 1024),   ← 环境即时 reward
    "position_ids":   Tensor(2, 3, 5120),← 扩展后的 position_ids
  }

batch.union(gen_batch_output) → 完整 batch:
  现在 combine 了:
  - 原始元数据: data_source, reward_model, index
  - 生成结果:   responses, action_mask, attention_mask, env_reward
  → 可以进入 Step 3（reward 计算）
```

---

## Q2: 为什么 RL 训练不按 epoch 遍历？

### 核心原因：RL 的数据是 "on-policy" 的，过期很快

| | 监督学习 (SFT) | RL (PPO/GRPO) |
|---|---|---|
| **数据来源** | 静态数据集，人工标注 | 模型**实时生成** → 环境反馈 |
| **数据分布** | 固定不变的 | **随策略更新而变化** |
| **数据时效** | 不会过期 | 策略一更新，旧数据就 "过期" |
| **训练逻辑** | "把这批数据学好" | "生成一批→学一次→再生成新的" |
| **过拟合** | 多 epoch 可能过拟合 | 数据每次都是新的，不需要多 epoch |

### 更深层的原因：PPO 的 Importance Sampling 假设

```python
# PPO 的 loss 函数
ratio = π_θ_new(a|s) / π_θ_old(a|s)  # 新旧策略的概率比
#                      ↑
#                      这个分母是在 "旧策略" 下计算的 log_prob

# PPO 要求：
# 1. rollout 数据必须在 π_θ_old 下生成
# 2. ratio 太大会被 clip（通常 clip_range=0.2）
# 3. 如果 π_θ_new 和 π_θ_old 差别太大，clip 生效，梯度信号变弱

# 如果你用 π_θ_step0 生成的数据去训练 π_θ_step1000：
# ratio 会非常大 → 大部分数据被 clip → 几乎没有有效梯度
# = 白训练了
```

### 训练循环的本质

```python
for step in range(total_steps):
    # Step A: 用当前策略 π_θ 生成 rollout
    gen_batch_output = vllm_rollout(batch)  # ← π_θ 下采样

    # Step B: 计算 old_log_prob (π_θ 下的 log 概率)
    old_log_prob = compute_log_prob(model, batch)

    # Step C: 用这批数据更新策略
    # π_θ → π_θ'  (策略改变了！)
    loss = ppo_loss(log_prob_new, old_log_prob, advantages)
    loss.backward()
    optimizer.step()

    # Step D: 下一轮用 π_θ' 重新生成 → 数据是新的
    # 同一个 prompt + π_θ' → 不同的 response → 不同的 reward
```

---

## Q3: 每次都 iter() 不会重复采样吗？

### 会重复，但有三个机制保证这不是问题

**机制 1: RL 中 prompt 重复 ≠ 训练信号重复**

```python
# Step 0:
prompt = "这张图里有几个苹果？"
response = model.generate(prompt)  # → "图里有 3 个苹果"
reward = 1.0  # 正确
# 训练：鼓励 "3 个苹果" 这种回答

# Step 100 (同一个 prompt 再次被采样到):
prompt = "这张图里有几个苹果？"  # ← prompt 相同
response = model.generate(prompt)  # → "经过分析，答案是 3"
reward = 1.0  # 正确
# 训练：鼓励 "经过分析，答案是 3" 这种回答
#      但是！模型现在的策略和 step 0 完全不同了
#      同样的 prompt + 不同的策略 → 不同的 response → 不同的训练信号
```

**机制 2: shuffle 保证每次顺序不同**

```python
# DataLoader 配置
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,  # ← 关键！
)

# 每次 iter() 创建新迭代器 → sampler 重新 shuffle

# Step 0: next(iter(dataloader))  →  [sample_3, sample_7]
# Step 1: next(iter(dataloader))  →  [sample_1, sample_9]
# Step 2: next(iter(dataloader))  →  [sample_5, sample_2]
# Step 3: next(iter(dataloader))  →  [sample_4, sample_1]  ← sample_1 重复了
#                                    但中间隔了很多步，策略已经变了
```

**机制 3: 数据量通常远大于训练步数**

```python
# 典型配置
dataset_size = 10000   # 数据集大小
batch_size = 8         # 每步样本数
total_steps = 500      # 训练步数

# 总共消耗: 500 × 8 = 4000 条
# 远小于数据集大小 (10000)
# 大部分数据都没被重复采到
```

### 更稳健的工业做法

```python
# 无限循环 DataLoader（很多 RL 框架的实现）
class InfiniteDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __iter__(self):
        while True:
            for batch in self.dataloader:
                yield batch
            # epoch 结束时自动重新 shuffle，继续循环

# 使用
for batch in InfiniteDataLoader(train_dataloader):
    train_step(batch)
    if step >= total_steps:
        break
```

---

## Q4: attention_mask 就是一个打标记的吗？

### 是的，但需要精确理解 "标记 = 0 模型看不到" 的机制

**attention_mask 的语义**：

```
1 = 这个位置是真实 token → 参与 attention 计算
0 = 这个位置是 PAD      → attention score 变成 -inf
```

### 屏蔽的数学实现

```python
# Transformer 内部 attention 计算（简化）

# 输入: Q, K, V, attention_mask
# Q, K, V 的 shape: (bs, num_heads, seq_len, head_dim)

# 1. 计算 attention scores
scores = Q @ K.transpose(-2, -1) / sqrt(d_k)
# shape: (bs, num_heads, seq_len, seq_len)
# scores[i, :, 3, :] 表示 "第 3 个 token 对序列中每个 token 的关注度"

# 2. 应用 mask：mask=0 的位置 score = -inf
scores = scores.masked_fill(attention_mask == 0, float('-inf'))
#                                          ↑
#                          所有 PAD token 位置的 score → -inf

# 3. softmax
weights = F.softmax(scores, dim=-1)
# softmax(-inf) = 0
# → PAD token 的 attention weight 精确为 0
# → "模型看不到 PAD token"

# 4. 加权求和
output = weights @ V
```

### 本项目中的三层 mask 对比

这是最容易混淆的知识点——**三个不同的 mask，各管各的**：

| Mask | Shape | 谁产生的 | 控制什么 |
|------|-------|---------|----------|
| `attention_mask` | `(bs, total_len)` | DataLoader (collate_fn) | Transformer forward: 哪些 token 参与 attention |
| `action_mask` | `(bs, total_len)` | Agent Loop | Loss backward: 哪些 token 参与 policy loss |
| `response_mask` | `(bs, response_len)` | 从 action_mask 截取 | 同上，但只取 response 部分 |

### 关键区分：observation token 的两种 mask

```
一个 observation token 的两个视角：

┌─────────────────────────────────────────────────────────────┐
│  attention_mask: 1                                          │
│  "模型需要看到执行结果，才能决定下一步做什么"                    │
│  → Forward 时参与 attention 计算                              │
│  → 影响 logits 的生成                                        │
├─────────────────────────────────────────────────────────────┤
│  action_mask: 0                                             │
│  "这个 token 不是模型生成的，不能 reward/punish 模型"          │
│  → Backward 时梯度为 0                                       │
│  → 不影响参数更新                                            │
└─────────────────────────────────────────────────────────────┘

注意：attention_mask 的 obs=1 ≠ action_mask 的 obs=0
     "forward 看到" ≠ "backward 管到"
```

### 完整示例

```python
# 一条样本的 token 序列
sequence:        [P0..P99 | A0..A49 | O0..O29 | A0..A39 | PAD...PAD]
                  ↑ prompt ↑ action1 ↑ obs1   ↑ action2 ↑ padding

attention_mask:  [1...1   | 1...1   | 1...1   | 1...1   | 0...0]
                  ↑ 全 1 (除了 PAD)

action_mask:     [1...1   | 1...1   | 0...0   | 1...1   | 0...0]
                  ↑ prompt  ↑ action  ↑ obs!!  ↑ action  ↑ PAD
                  也是 1               全是 0

# 为什么 prompt 在 action_mask 中也是 1？
# 虽然 prompt 不是模型生成的，但在单轮（非 multi-turn）模式下
# response_mask 只截取 [-response_len:] 部分
# 所以 prompt 的 mask 值实际上不会被用到
```

---

## 面试题库

---

## Q5: 请描述一个完整的 Agentic RL 训练流程

**高频指数**：⭐⭐⭐⭐⭐（必问）

### 答案

从 prompt 到参数更新的 7 个步骤：

```
Step 1 — DataLoader → batch:
  从数据集取出 input_ids (bs, prompt_len), attention_mask, 图像等
  包装成 DataProto，pop() 分离 gen_batch

Step 2 — Agent Rollout:
  vLLM 接收 gen_batch → 多轮生成:
    Turn 1: 模型生成 <code>...python code...</code> (action tokens)
            sandbox 执行代码 → 返回 stdout/stderr/images (obs tokens)
    Turn 2: 模型基于 obs 继续生成 → 输出 <answer>...</answer>
            done=True，停止
  返回: responses, action_mask, attention_mask, env_reward

Step 3 — Reward:
  Judge LLM / rule-based 评分
  Outcome reward 放在最后一个有效 token 上（稀疏 reward）

Step 4 — Old Log Prob:
  FSDP 模型 forward → 计算 π_θ_old 下的 log_prob
  torch.no_grad() 下执行，不参与梯度

Step 5 — Advantage (GRPO):
  同 prompt 的多条 response 组内归一化:
    A_i = (r_i - mean_group) / (std_group + ε)
  扩展到 token-level: advantages = scores * response_mask
  obs token 的 advantage = 0

Step 6 — Policy Loss:
  ratio = exp(log_prob_new - log_prob_old)
  pg_loss = -advantages * ratio (clipped)
  final_loss = sum(pg_loss * response_mask) / sum(response_mask)
  → backward()

Step 7 — Optimizer Step:
  clip_grad_norm_() → optimizer.step() → zero_grad()
```

### 加分细节

> 强调 action_mask 的核心作用——它区分了 "模型的行为"（action token）和 "环境的反馈"（obs token），确保模型只学习决策能力，不学习预测环境输出。

---

## Q6: action_mask 为什么设计成 action=1, obs=0？反过来行不行？

**高频指数**：⭐⭐⭐⭐

### 答案

**不行。这反映了 Agentic RL 要优化的目标：模型的决策能力，而非预测能力。**

```
action_mask 标记的是 "这个 token 是谁产生的"：

  action token → 模型生成 → mask=1 → 参与 loss → 有梯度 → 参数更新
  obs token    → 环境返回 → mask=0 → 不参与 loss → 无梯度 → 参数不变

如果反过来（obs=1, action=0）：
  → 模型在学习 "预测环境返回什么"
  → 而不是 "在什么状态下做什么决策"
  → 浪费模型容量
```

### 具体例子

```python
# 假设模型输出了错误答案
action:  "答案是 5"  →  judge: 错误，reward=0
obs:     "错误，正确答案是 4"

# action_mask=1, obs_mask=0 (正确做法):
# 模型因为 action "答案是 5" 被惩罚 (advantage=0)
# 模型学习：下次不要说 "5"

# action_mask=0, obs_mask=1 (错误做法):
# 模型因为 obs "正确答案是 4" 被奖励
# 模型学习：当我犯错时，老师会告诉我答案 = 太好了！
# → 模型学会了 "犯错然后等老师纠正"，而不是 "做出正确决策"
```

### 类比

```
学生考试：
  ✓ 奖励/惩罚的是 "他写的答案"（action）
  ✗ 不应该奖励 "老师在卷子上批改的内容"（observation）

模型训练：
  ✓ 梯度信号来自 action token 的 log_prob
  ✗ obs token 的 advantage 被 mask 清零
```

---

## Q7: GRPO 和 PPO 的区别？为什么用 GRPO？

**高频指数**：⭐⭐⭐⭐⭐

### 答案

```
核心区别：GRPO 不需要 Critic（价值网络）

PPO:
  advantage = r_t + γ*V(s_{t+1}) - V(s_t)
  #                      ↑ 需要额外的 Value Network 估计
  # Actor (策略网络) + Critic (价值网络) 同时训练

GRPO:
  同一 prompt 生成 n 条 response: {o_1, ..., o_n}
  group_mean = mean(r_1, ..., r_n)
  group_std  = std(r_1, ..., r_n)
  advantage_i = (r_i - group_mean) / (group_std + ε)
  #                      ↑ 组内归一化，不需要 Critic
```

### GRPO 相比 PPO 的优势

| 维度 | PPO | GRPO |
|------|-----|------|
| 参数量 | Actor + Critic (2x) | 只有 Actor (1x) |
| 超参数 | GAE λ, Critic lr, Critic loss weight... | 基本没有额外参数 |
| 训练稳定性 | Critic 可能训崩 | 组内归一化天然稳定 |
| Memory | 需要存 GAE 中间值 | 只需要 reward |
| Reward scale 敏感性 | 敏感，需要调 | 不敏感，组内归一化消除 scale |

### 为什么在 LLM 场景下 GRPO 更好

```
1. LLM 本身就很强 → 不需要 Critic 提供精细的逐 token 反馈
   outcome reward (最终结果对/错) 足够作为训练信号

2. LLM 参数量巨大 → 去掉 Critic 省一半显存

3. DeepSeek-R1 在数学推理上验证了 GRPO 的效果
   用简单的 outcome reward (答案对/错) + GRPO → 涌现了 chain-of-thought

4. 工业界趋势：GRPO 正在成为 LLM RL 训练的默认选择
```

### PPO 仍然有用的场景

```
- rollout_n=1（每个 prompt 只有一条 response）→ 无法组内归一化
- 需要精细 credit assignment（每个 action 的好坏需要独立评估）
- 环境中 reward 信号很密集且需要时序建模
```

### 数学公式

```
GRPO Advantage:
  For prompt q, responses {o_1, ..., o_n}, rewards {r_1, ..., r_n}:

  μ_q = (1/n) * Σ r_i
  σ_q = sqrt((1/n) * Σ (r_i - μ_q)²)

  A_i = (r_i - μ_q) / (σ_q + ε)

  解释:
  - r_i > μ_q → A_i > 0 → 鼓励这种回答
  - r_i < μ_q → A_i < 0 → 惩罚这种回答
  - 组内 comparison → 不受 reward scale 影响
```

---

## Q8: old_log_prob 为什么要重新计算？为什么需要 detach？

**高频指数**：⭐⭐⭐⭐

### 答案

**为什么要重新计算**：

```
1. vLLM 生成时的 logits 不可靠:
   - tensor parallelism (TP) 切分导致 logits 不完整
   - KV cache 优化可能改变数值精度
   - vLLM 和 FSDP 模型的权重可能不同步

2. 需要确保 old_log_prob 和当前 Actor 权重的一致性:
   ratio = π_θ_new(a|s) / π_θ_old(a|s)
   如果 π_θ_old 不准，ratio 就没意义了
```

**为什么要 detach**：

```python
# 正确做法：
with torch.no_grad():
    old_log_prob = compute_log_prob(model, batch)
old_log_prob = old_log_prob.detach()  # 显式切断梯度

# ratio = exp(log_prob_new - old_log_prob)
# 如果 old_log_prob 有梯度：
#   → 反向传播会同时更新 log_prob_new 和 old_log_prob
#   → 但 old_log_prob 代表的是 "旧策略"，不应该被更新
#   → PPO loss 的定义被破坏

# 数学上：
# L = E[min(ratio * A, clip(ratio) * A)]
# ratio = π_θ(s,a) / π_θ_old(s,a)  ← 分母是常数
#
# 如果分母也参与求导：
#   ∂L/∂θ = ∂L/∂ratio * (∂π_θ/∂θ * 1/π_θ_old + π_θ * ∂(1/π_θ_old)/∂θ)
#                                                    ↑ 多出来的项！
# → loss 不再是 "当前策略相对于旧策略的改进"
# → PPO 的 trust region 理论保障失效
```

### 分布式训练中的执行流程

```text
Driver 进程:
  old_log_prob = actor_wg.compute_log_prob(batch)
  #              ↑ RPC 调用 FSDP worker
  #              FSDP worker 在 torch.no_grad() 模式下执行
  #              返回的 tensor 已经 detach

  # 用 old_log_prob 计算 advantage
  advantages = compute_grpo_advantage(reward, response_mask)

  # 发给 FSDP worker 做 update
  actor_wg.update_policy(batch, old_log_prob, advantages)
  # 在 update_policy 内部：
  #   log_prob_new = forward(model, batch)  ← 有梯度
  #   ratio = exp(log_prob_new - old_log_prob)  ← old_log_prob 是常数
  #   loss = ppo_loss(ratio, advantages)
  #   loss.backward()  ← 梯度只通过 log_prob_new 流向模型参数
```

---

## Q9: 多轮 Agent 交互中，超长序列怎么处理？

**高频指数**：⭐⭐⭐

### 答案

**四层保护机制**：

```
Layer 1 — vLLM max_tokens:
  sampling_params.max_tokens = max_response_length
  单次生成超出的 token 直接截断

Layer 2 — max_turns:
  最多交互 max_turns 轮
  即使模型没输出 <answer>，也强制停止

Layer 3 — 总长度截断:
  running_states = [state[:max_total_length] for state in running_states]
  prompt_len + response_len = total_len (如 5120)

Layer 4 — PAD 对齐:
  pad_2d_list_to_length(running_states, pad_value, max_total_length)
  不等长的序列 pad 到统一长度，确保能 stack 成 batch
```

### 不截断会怎样？

```python
# 1. position_ids 长度不匹配
position_ids: (bs, 3, 5120)  ← 预分配了固定长度
# 如果序列实际长度 > 5120 → 位置编码越界 → 前向报错

# 2. 跨样本不对齐
样本0: 220 个 token
样本1: 580 个 token
# torch.stack 要求所有 tensor 同 shape → 必须 pad
```

### 设计 tradeoff

```
response_len 太小:
  → 复杂推理被截断
  → 模型学不会多步推理
  → reward 计算不准（截断的回答可能不完整）

response_len 太大:
  → 大量 PAD token 浪费显卡显存
  → attention 计算量 = O(seq_len²)
  → 训练速度变慢

实际做法:
  统计训练集中 action + obs 的长度分布
  取 p95 或 p99 分位数作为 response_len
  典型值: 1024 (简单任务) ~ 4096 (复杂推理)
```

---

## Q10: 为什么用 token-level mask 而不是直接在 loss 里取 action token 的索引？

**高频指数**：⭐⭐⭐

### 答案

**token-level mask 在不等长序列处理上更优雅**：

```python
# 方案 A: token-level mask（当前做法）
final_loss = sum(pg_loss * response_mask) / sum(response_mask)
# 优点:
# 1. 一行代码
# 2. 不等长序列自动处理（mask 值就是 0）
# 3. 广播友好
# 4. 梯度自动为 0（乘法 → autograd 自动处理）

# 方案 B: 索引筛选（假设不用 mask）
losses = []
for i in range(bs):
    for j in range(seq_len):
        if is_action_token(i, j):  # 需要精确知道边界
            losses.append(pg_loss[i, j])
final_loss = sum(losses) / len(losses)
# 缺点:
# 1. 需要 for 循环
# 2. 每条样本的 action/obs 边界不同，需要额外记录
# 3. 破坏了 tensor 的向量化操作
# 4. 不能和 attention_mask 复用逻辑
```

### 具体例子

```python
# 两条样本，action/obs 边界不同
样本0: [A A A | O O | A A A A]  (3 action + 2 obs + 4 action)
样本1: [A A | O O O | A A A]    (2 action + 3 obs + 3 action)

# 用 mask:
action_mask[0] = [1, 1, 1, 0, 0, 1, 1, 1, 1]
action_mask[1] = [1, 1, 0, 0, 0, 1, 1, 1]

loss = (pg_loss * action_mask).sum() / action_mask.sum()
# ← 一行搞定，每条样本自动处理不同边界

# 用索引:
losses = []
losses.extend(pg_loss[0, :3])   # 需要知道样本0的前3个是 action
losses.extend(pg_loss[0, 5:9])  # 需要知道样本0的5-8是 action
losses.extend(pg_loss[1, :2])   # ...
losses.extend(pg_loss[1, 5:8])
# ← 边界管理噩梦
```

### 为什么广播友好

```python
# advantages 从标量扩展到 token-level
scores = torch.tensor([0.8, 0.0])          # (bs,)
advantages = scores.unsqueeze(-1) * mask    # (bs, 1) * (bs, seq_len) = (bs, seq_len)
# obs token 自动变成 0，action token 自动变成对应的 advantage

# 如果用索引 → 需要手动分配到每个 position
```

---

## Q11: 如果让你改进这个 Agentic RL Pipeline，你会怎么做？

**高频指数**：⭐⭐⭐⭐（看思考深度和创新性）

### 答案框架：从六个维度切入

#### 1. Credit Assignment 改进

```
当前: outcome reward 只给最后一个 token（稀疏信号）
改进: Process Reward Model (PRM)
  - 给每个 reasoning step 打分
  - 提供中间监督信号
  - 训练: 用人类标注的 step 正确性 + 自动标注（最终答案对 → 中间步骤也对）

tradeoff: PRM 训练数据贵，且 PRM 本身可能 reward hacking
```

#### 2. Rollout 效率

```
当前: 同步 rollout → 等所有样本生成完 → 一起 compute reward
改进: 异步 rollout
  - 生成完一条 → 马上发回去 compute reward
  - reward 计算和生成并行
  - 理论加速: ~2x（省掉等待最长生成的时间）

挑战:
  - 需要协调异步梯度
  - advantage 归一化需要同组数据（GRPO 依赖 batch 内的 mean/std）
  - 可以用 streaming normalization (Welford's algorithm)
```

#### 3. Multi-turn 稳定性

```
当前: max_turns 硬截断
改进:
  a) Learned early-stop:
     - 模型学习输出 <stop> token
     - reward 中加入 step penalty (-0.01/turn) 防止无限循环

  b) Adaptive max_turns:
     - 简单问题 1-2 轮够了
     - 复杂问题给更多轮
     - 基于 prompt 复杂度预测所需轮数

  c) Tool-use efficiency reward:
     - 奖励 "用更少轮次得到正确答案"
     - = accuracy_reward + α * efficiency_reward
```

#### 4. Reward Design

```
当前: 单一 outcome reward（对/错 + 格式）
改进: 多维度 reward

  reward = w1 * correctness     (答案对错)
         + w2 * format          (格式规范)
         + w3 * code_quality    (代码是否有冗余/错误)
         + w4 * tool_selection  (是否选了正确的工具)
         + w5 * efficiency      (是否多次重复调用)
         + w6 * safety          (是否有危险操作)

  Learned Reward Model:
  - 用 Bradley-Terry 模型从人类偏好中学习
  - 比 rule-based 更灵活，能捕捉 "风格/质量" 等主观维度
```

#### 5. 数据效率

```
当前: on-policy → 每批数据只用一次
改进: 加入 replay buffer (off-policy)

  挑战:
  - LLM 参数巨大 → off-policy correction 困难
  - 旧数据和新策略的 ratio 可能非常大

  可能的做法:
  - "温和" off-policy: 保留最近 K 个 step 的数据
  - importance sampling correction 只在 ratio 不太大时有效
  - 或者用 policy distillation: 用 rollouts 训练一个较小的 policy
```

#### 6. 多样性和探索

```
当前: temperature sampling
改进:
  a) Best-of-N sampling: 生成 2N 条，选最好的 N 条更新
  b) Entropy bonus: loss 中加入 entropy 项，鼓励策略探索
  c) Curiosity-driven exploration: 奖励 "模型未见过的状态"
  d) Diverse reward functions: 用不同 judge 从不同角度评价
```

### 总结表格

| 维度 | 当前做法 | 可能改进 |
|------|---------|---------|
| Credit Assignment | Outcome reward | PRM (process reward model) |
| Rollout 效率 | 同步 | 异步 rollout |
| Multi-turn | max_turns 硬截断 | Learned early-stop + step penalty |
| Reward | 单一 outcome | 多维 + learned reward model |
| 数据效率 | On-policy | Replay buffer + IS correction |
| 探索 | Temperature | Entropy bonus + curiosity |

---

## Q12: 解释一下 position_ids 为什么是 (2, 3, 5120)？

**高频指数**：⭐⭐⭐（视觉-语言模型岗位专属）

### 答案

**普通 Transformer vs Qwen2-VL 的区别**：

```
普通 GPT / Qwen-Text:
  position_ids: (bs, seq_len)
  含义: 第 k 个元素 = k
  因为文本是线性的，"第 5 个 token" 只需要一个数字描述位置

Qwen2-VL:
  position_ids: (bs, 3, seq_len)
  三个维度分别代表: [temporal, height, width]
```

### 为什么图像需要三维位置编码

```
一张图被 Vision Encoder 切成 patches:
┌────┬────┬────┬────┐
│ 0,0│ 0,1│ 0,2│ 0,3│  ← h=0 (第一行)
├────┼────┼────┼────┤
│ 1,0│ 1,1│ 1,2│ 1,3│  ← h=1 (第二行)
├────┼────┼────┼────┤
│ 2,0│ 2,1│ 2,2│ 2,3│  ← h=2 (第三行)
└────┴────┴────┴────┘
  ↑w=0  ↑w=1  ↑w=2  ↑w=3

每个 patch 有两个空间坐标 (h, w)
→ 如果用一维位置编码，就丢失了空间结构
→ 需要至少二维位置编码
```

### MRoPE 的工作原理

```
MRoPE = Multi-Resolution Rotary Position Embedding

RoPE 对每个 head_dim 的子空间独立旋转:

  head_dim 分成 3 段:
  ┌──────────┬──────────┬──────────┐
  │ d_t      │ d_h      │ d_w      │
  │ temporal │ height   │ width    │
  └──────────┴──────────┴──────────┘

  RoPE_t(t)  → 第一段子空间按 temporal 坐标旋转
  RoPE_h(h)  → 第二段子空间按 height 坐标旋转
  RoPE_w(w)  → 第三段子空间按 width 坐标旋转

  最终: pos_encoding = concat([RoPE_t(t), RoPE_h(h), RoPE_w(w)])
```

### 文本 token 的位置编码

```
纯文本 token 在三个维度上共享同样的递增位置:

  "这是一张图片" → tokenize → [t1, t2, t3, t4, t5]

  temporal: [0, 1, 2, 3, 4]
  height:   [0, 1, 2, 3, 4]   ← 三个维度相同
  width:    [0, 1, 2, 3, 4]

图像 patch 后的文本 token:
  "图中有 3 个苹果" → 在图像 patches 之后

  temporal: [..., 5, 6, 7, 8, 9]
  height:   [..., 5, 6, 7, 8, 9]
  width:    [..., 5, 6, 7, 8, 9]
  ↑ 继续递增，和前面图像 patches 的位置保持连续性
```

### 好处

```
1. 保留二维空间结构:
   patch(0,0) 和 patch(0,1) 在 width 维度的距离 = 1
   patch(0,0) 和 patch(1,0) 在 height 维度的距离 = 1
   → 模型能学到 "左右邻居" 和 "上下邻居" 的区别

2. 多分辨率兼容:
   大图 → 更多 patches → 更大的 h,w 坐标范围
   小图 → 更少 patches → 更小的 h,w 坐标范围
   MRoPE 通过插值支持任意分辨率

3. 视频扩展自然:
   temporal 维度给视频帧编号
   → (t, h, w) 三个坐标建模视频 patches
```

---

## 面试速记表

| 编号 | 问题 | 3 秒答案 |
|------|------|---------|
| Q1 | gen_batch vs batch | gen_batch 给 vLLM 生成用（tensor+图像），batch 留 driver 调度用（元数据） |
| Q2 | 不按 epoch | RL 是 on-policy，策略更新后旧数据过期，需要重新生成 |
| Q3 | 重复采样 | shuffle + 策略每次不同 → prompt 重复但训练信号不重复 |
| Q4 | attention_mask | 标记 1=真实 token 参与 forward，0=PAD → score=-inf → softmax=0 |
| Q5 | 完整流程 | prompt→rollout→reward→old_log_prob→advantage→loss→backward→step |
| Q6 | action_mask | action=模型生成=1 参与 loss，obs=环境返回=0 不参与 loss |
| Q7 | GRPO vs PPO | 不需要 Critic、组内归一化、参数减半、DeepSeek-R1 验证 |
| Q8 | old_log_prob | vLLM 不可靠需重算；detach 保证 PPO 的分母是常数 |
| Q9 | 超长处理 | max_tokens + max_turns + 截断 + PAD，response_len 根据任务统计分布设定 |
| Q10 | mask vs 索引 | mask 一行代码 + 自动广播 + 不等长友好 + 梯度自动为 0 |
| Q11 | 改进方向 | PRM / 异步 rollout / Learned early-stop / 多维 reward / replay buffer |
| Q12 | 3D position | MRoPE：(t, h, w) 独立编码图像 patches 的时空位置，保留二维结构 |
