# url04：Step 4 详解 —— Old Log Prob（旧策略 Log 概率计算）

> 本文是对 `2026-06-02_rl-forward-backward-shapes-beginner-zh.md` 中 **Step 4** 的逐行深入解释。
> 目标读者：理解了 Step 2 的 rollout 和 Step 3 的 reward，想深入理解 "为什么要重算旧策略的 log 概率" 的同学。

---

## 目录

1. [Step 4 整体做什么](#1-step-4-整体做什么)
2. [前置：Log Probability 是什么](#2-前置log-probability-是什么)
3. [为什么需要重算 old_log_prob](#3-为什么需要重算-old_log_prob)
4. [compute_log_prob 完整流程](#4-compute_log_prob-完整流程)
5. [_forward_micro_batch 详解](#5-_forward_micro_batch-详解)
6. [logprobs_from_logits 数学原理](#6-logprobs_from_logits-数学原理)
7. [关键切片：为什么是 `-response_len-1:-1`](#7-关键切片为什么是--response_len-1-1)
8. [Remove Padding 优化](#8-remove-padding-优化)
9. [完整 Shape 变化追踪](#9-完整-shape-变化追踪)
10. [面试题库](#10-面试题库)

---

## 1. Step 4 整体做什么

**一句话**：用 FSDP 模型重新前向传播一次，算出每个 response token 在**当前（旧）策略**下的 log 概率，作为 PPO 中 `ratio` 的分母。

```text
┌──────────────────────────────────────────────────────────────┐
│  Step 4: Old Log Prob                                         │
│                                                               │
│  输入:                                                         │
│    batch.batch["input_ids"]        (bs, total_len=5120)      │
│    batch.batch["attention_mask"]   (bs, total_len=5120)      │
│    batch.batch["position_ids"]     (bs, 3, total_len=5120)  │
│    batch.batch["responses"]        (bs, response_len=1024)  │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  ① RPC 调用 FSDP Worker                                │    │
│  │     old_log_prob = actor_wg.compute_log_prob(batch)   │    │
│  │                                                         │    │
│  │  ② FSDP Worker 以 eval 模式 + torch.no_grad() 执行:    │    │
│  │     a) 模型前向: logits = model(input_ids)             │    │
│  │     b) 截取 response 部分 logits                       │    │
│  │     c) logprobs_from_logits → log_prob per token       │    │
│  │                                                         │    │
│  │  ③ 返回结果                                             │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                               │
│  输出:                                                         │
│    old_log_probs:  (bs, response_len=1024)                   │
│    entropys:       (bs, response_len=1024)  (可选)           │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. 前置：Log Probability 是什么

### 定义

```
Log Probability = 模型认为 "某个 token 在此处出现的概率" 的对数值

假设 vocab_size = 151936, token "4" 在当前位置的预测概率 = 0.3
→ log_prob("4") = ln(0.3) ≈ -1.204

token "dog" 的预测概率 = 0.001
→ log_prob("dog") = ln(0.001) ≈ -6.908
```

**为什么要用 log？**

```
1. 数值稳定性:
   概率很小（如 10^{-6}）→ 直接乘容易 underflow
   log 后 → 乘法变加法 → 数值稳定

2. PPO 中 ratio = exp(log_prob_new - log_prob_old)
   log 空间做减法 = 概率空间做除法 → 数值更稳定

3. 梯度友好:
   ∂(log p)/∂θ = (1/p) * ∂p/∂θ
   对于小概率事件，1/p 很大 → 梯度大 → 对小概率的正确 token 学习更快

4. 信息论解释:
   -log_prob 就是交叉熵 = 模型对正确答案的 "惊讶程度"
   高概率 → 低惊讶 → 小的 -log_prob
   低概率 → 高惊讶 → 大的 -log_prob
```

### 直观理解

```python
# 假设模型输出 <answer> 后的下一个 token 是 "4"
# 模型预测的概率分布 (简化，实际有 151936 维):
vocab_probs = {
    "4":   0.30,   # ← 正确答案
    "5":   0.15,
    "6":   0.10,
    "a":   0.05,
    ...
}

# log_prob("4") = ln(0.30) ≈ -1.2
# 如果模型非常确定是 "4" (prob=0.9) → log_prob ≈ -0.1
# 如果模型非常不确定 (prob=0.01) → log_prob ≈ -4.6
```

---

## 3. 为什么需要重算 old_log_prob

### 三个原因

**原因 1：vLLM 生成的 logits 不可靠**

```
vLLM 的优化牺牲了 logits 精度：
  - Tensor Parallelism (TP): logits 分布在多 GPU → 拼接可能有数值误差
  - PagedAttention KV cache: 内存管理优化 → 可能影响 attention 计算精度
  - FP16 vs BF16: vLLM 可能用 FP16，FSDP 用 BF16 → 数值差异
  - Continuous batching: 多个请求共享 batch → 可能影响 logits

直接后果: vLLM 的 log_prob 和 FSDP 重算的 log_prob 可能不一致
```

**原因 2：确保 old_log_prob 和当前权重严格一致**

```python
# PPO 的 ratio:
ratio = exp(log_prob_new - log_prob_old)
#        ↑               ↑
#        算 loss 时重算    compute_log_prob 时重算
#        都用 FSDP 模型    确保数值一致性

# 如果 old_log_prob 来自 vLLM:
# ratio 的分子分母用不同精度/实现 → 数值偏差 → PPO clip 失效
```

**原因 3：`torch.no_grad()` 确保 old_log_prob 是常数**

```python
# dp_actor.py 第 317 行
with torch.no_grad():  # ← 关键！
    entropy, log_probs = self._forward_micro_batch(...)

# torch.no_grad() 的作用:
#   1. 不构建计算图 → 省显存（不存中间激活值）
#   2. 返回的 log_probs 没有 .grad 属性
#   3. 在 PPO loss 中 old_log_prob 是常数 → 梯度只通过 log_prob_new 流动
```

---

## 4. compute_log_prob 完整流程

### 入口（dp_actor.py:271-335）

```python
def compute_log_prob(self, data: DataProto, calculate_entropy=False):
    # ① 切换到 eval 模式
    self.actor_module.eval()
    #   作用: 关闭 dropout、batch_norm 的 training 行为
    #   保证 log_prob 是确定性的（同一个输入 → 同一个输出）

    # ② 按 micro_batch_size 拆分
    micro_batches = batch.split(micro_batch_size)
    #   batch_size=2, micro_batch_size=1
    #   → 2 个 micro_batch，每个 (1, 5120)

    # ③ 对每个 micro_batch 前向计算
    log_probs_lst = []
    for micro_batch in micro_batches:
        with torch.no_grad():  # ← 不构建计算图
            entropy, log_probs = self._forward_micro_batch(
                micro_batch,
                temperature=temperature,
                calculate_entropy=calculate_entropy,
            )
        log_probs_lst.append(log_probs)

    # ④ 合并所有 micro_batch
    log_probs = torch.concat(log_probs_lst, dim=0)  # (bs, response_len)

    return log_probs, entropys
```

### 流程图

```text
data: DataProto
  batch_size = 2
  micro_batch_size = 1
        │
        ▼
┌─────────────────────────────────────┐
│  split(micro_batch_size=1)           │
│                                      │
│  micro_batch_0: (1, 5120)           │
│  micro_batch_1: (1, 5120)           │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  for each micro_batch:              │
│    with torch.no_grad():            │
│      _forward_micro_batch()         │
│        → log_probs (1, 1024)        │
│        → entropy   (1, 1024)        │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  concat(dim=0)                      │
│  → log_probs (2, 1024)             │
│  → entropys  (2, 1024)             │
└─────────────────────────────────────┘
```

---

## 5. _forward_micro_batch 详解

### 5.1 整体流程

```python
def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False):
    response_length = micro_batch["responses"].size(-1)  # = 1024
    input_ids = micro_batch["input_ids"]                  # (1, 5120)
    attention_mask = micro_batch["attention_mask"]         # (1, 5120)
    position_ids = micro_batch["position_ids"]             # (1, 3, 5120)

    # ① 模型前向
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = self.actor_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,  # 不是生成，不需要 KV cache
        )

    # ② 取 response 部分的 logits
    logits = output.logits  # (1, 5120, vocab_size), vocab_size ≈ 151936
    logits = logits[:, -response_length - 1 : -1, :]
    #                ↑                          ↑
    #                倒数第 1025 个到倒数第 2 个

    # ③ 计算 log_prob
    log_probs = logprobs_from_logits(
        logits=logits,                        # (1, 1024, 151936)
        labels=micro_batch["responses"],       # (1, 1024)
    )
    # → (1, 1024): 每个 response token 在当前位置的 log 概率

    return entropy, log_probs
```

### 5.2 为什么要 `logits[:, -response_len-1:-1, :]`？

这是最容易被问到的问题。答案涉及**自回归模型的预测机制**。

```text
自回归模型的核心: 用前面的 token 预测 "下一个" token

序列: [P0, P1, ..., P4095, R0, R1, ..., R1023]
       ↑── prompt ──↑  ↑── response ──↑
                          ↑ response_len=1024

模型在每个位置预测的是 next token:
  logits[0, 4094, :] → 预测 token 4095 (第 4095 个位置应该是什么) → 也就是 R0
  logits[0, 4095, :] → 预测 token 4096 (第 4096 个位置应该是什么) → 也就是 R1
  ...
  logits[0, 4094+k, :] → 预测 token 4095+k → 也就是 Rk
  ...
  logits[0, 4094+1023, :] → 预测 token 4095+1023 → 也就是 R1023

所以:
  logits[:, -response_len-1:-1, :] = logits[:, 4095:5119, :]  (取 1024 个)
  这 1024 个 logits 分别预测 R0, R1, ..., R1023

  而 labels = responses = [R0, R1, ..., R1023]  (1024 个)

一一对应:
  logits[4095] 预测 → labels[0] = R0   ✓
  logits[4096] 预测 → labels[1] = R1   ✓
  ...
  logits[5119] 预测 → labels[1023] = R1023  ✓
```

### 5.3 图示

```text
token 序列 (全长 5120):
位置:  0     1    ...  4094  4095  4096  ...  5119
       P0    P1   ...  P4094 R0    R1    ...  R1023
                         ↑ prompt结束  ↑ response开始

logits 的预测对象:
  logits[0]     → 预测 position 1 的 token (P1)
  ...
  logits[4094]  → 预测 position 4095 的 token (R0)  ← response 第一个
  logits[4095]  → 预测 position 4096 的 token (R1)
  ...
  logits[5118]  → 预测 position 5119 的 token (R1023)
  logits[5119]  → 预测 position 5120 的 token (不存在，这是最后一个 logits)

所以取 logits[:, 4095:5119, :]  → 对应预测 R0 到 R1023 的 1024 个 logits

切片等价于:
  logits[:, -response_len-1:-1, :]
  = logits[:, -(1024+1):-1, :]
  = logits[:, -1025:-1, :]
  = 从倒数第 1025 个取到倒数第 2 个 (不含倒数第 1 个)
  = 1024 个 logits
```

---

## 6. logprobs_from_logits 数学原理

### 源码（torch_functional.py）

```python
def logprobs_from_logits_naive(logits, labels):
    # ① log_softmax: 把 logits 转成 log 概率分布
    logp = F.log_softmax(logits, dim=-1)
    # log_softmax(x_i) = x_i - log(sum(exp(x_j)))
    # 输出: (batch, seq_len, vocab_size), 每个元素是 log(P(token_i))

    # ② gather: 取 label 对应位置的 log 概率
    logpy = gather_from_labels(logp, labels)
    # labels[i] = 目标 token 的 id
    # logp[i, labels[i]] = 模型对正确 token 的 log 概率

    return logpy  # shape: (batch, seq_len)
```

### 数值例子

```python
# 假设 vocab_size=5, 当前预测下一个 token
logits = torch.tensor([2.0, 1.0, 0.5, -1.0, 3.0])  # (5,)

# ① log_softmax
# softmax = [0.24, 0.09, 0.05, 0.01, 0.61]  (近似)
# log_softmax = [-1.43, -2.43, -2.93, -4.43, -0.50]
log_probs = F.log_softmax(logits, dim=-1)

# ② 正确答案是 token 4 (索引 4)
label = torch.tensor(4)
log_prob = log_probs[label]  # ≈ -0.50

# 解释: 模型给 token 4 的概率最高 (softmax=0.61)
#       log_prob 接近 0 (概率接近 1)
#       → 模型对这个 token 很 "确定"

# 如果正确答案是 token 3 (索引 3)
label = torch.tensor(3)
log_prob = log_probs[label]  # ≈ -4.43

# 解释: 模型给 token 3 的概率很低 (softmax=0.01)
#       log_prob 负得很大
#       → 模型对这个 token 很 "惊讶"
```

### 在 PPO 中的作用

```python
# Step 6 中:
old_log_prob = compute_log_prob(model, batch)  # 旧策略下的 log 概率
new_log_prob = forward_micro_batch(model, batch)  # 新策略下的 log 概率

ratio = exp(new_log_prob - old_log_prob)
# = exp(log(P_new(token))) - log(P_old(token)))
# = exp(log(P_new / P_old))
# = P_new / P_old

# 如果 ratio > 1: 新策略更倾向于这个 token → 增大该路径概率（如果 advantage > 0）
# 如果 ratio < 1: 新策略更不倾向于这个 token → 减小该路径概率（如果 advantage < 0）
```

---

## 7. 关键切片：为什么是 `-response_len-1:-1`

### 两条路径的实现

```python
# 路径 1: remove_padding 模式 (use_remove_padding=True)
# 第 123 行: 用 torch.roll 把 input_ids 左移一位作为 labels
input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)
# [A, B, C, D] → [B, C, D, A]
# 含义: 每个位置的 label = 下一个位置的 token

# 然后 pad 回原始长度后 (第 223 行):
log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]
# 去掉最后一个位置 (因为 roll 到的是不存在的 future token)
# 去掉 prompt 部分
# → 只保留 response 部分的 log_prob

# 路径 2: 非 remove_padding 模式
logits = logits[:, -response_length - 1 : -1, :]  # 取预测 response token 的 logits
log_probs = logprobs_from_logits(logits, responses)  # labels = responses
```

### 两种做法的对比

```text
方法 A (remove_padding): 用 torch.roll 得到 next token 作为 labels
  input_ids:         [P0, P1, ..., P4094, P4095, R0, R1, ..., R1023]
  input_ids_rolled:  [P1, P2, ..., P4095, R0,   R1, R2, ..., R1023, P0]
                      ↑                    ↑                      ↑
                      每个位置预测下一个       P4095→R0 ✓           最后一个→P0 ✗
  然后取 [-1025:-1]  → 去掉最后一个无效的对齐

方法 B (非 remove_padding): 用 logits 位置对齐
  logits:  [l0, l1, ..., l4094, l4095, l4096, ..., l5119]
  取 [-1025:-1] = [l4095, l4096, ..., l5118]  (1024 个)
  labels = responses = [R0, R1, ..., R1023]  (1024 个)
  l4095 → R0, l4096 → R1, ..., l5118 → R1023  ✓
```

---

## 8. Remove Padding 优化

### 动机

```python
# 问题: 序列中有大量 PAD token
input_ids = [P0, P1, ..., P99, PAD, PAD, ..., PAD]
            ↑ 100 个      ↑ 5020 个 PAD

# 普通 forward:
#   attention 在 5120 个 token 上计算 → 5020 个 token 是纯浪费

# Remove Padding:
#   去掉所有 PAD → 只对 100 个有效 token 做 attention
#   → 计算量: 100² vs 5120² → 约 2600x 加速!
```

### 实现（dp_actor.py:113-124）

```python
# 1. unpad: 去掉 PAD
input_ids_rmpad, indices, *_ = unpad_input(
    input_ids.unsqueeze(-1), attention_mask
)
# input_ids: (1, 5120) → input_ids_rmpad: (1, total_nnz)
# attention_mask=1 的位置被保留, =0 的位置被移除
# indices 记录了哪些位置被保留 → 用于后续 pad 回来

# 2. 只对有效 token 做 forward
output = self.actor_module(
    input_ids=input_ids_rmpad,       # (1, total_nnz)
    attention_mask=None,              # 不需要了！所有 token 都是有效的
    position_ids=position_ids_rmpad,  # 同步 unpad 的 position_ids
)

# 3. logprobs_from_logits
input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)
log_probs = logprobs_from_logits(logits, input_ids_rmpad_rolled)

# 4. pad 回去
full_log_probs = pad_input(
    hidden_states=log_probs.unsqueeze(-1),
    indices=indices,
    batch=batch_size,
    seqlen=seqlen,
)
# → (1, 5120), 有效位置有值, PAD 位置是 0

# 5. 只取 response 部分
log_probs = full_log_probs[:, -response_len-1:-1]  # (1, 1024)
```

---

## 9. 完整 Shape 变化追踪

```text
═══════════════════════════════════════════════════════════════════
输入
═══════════════════════════════════════════════════════════════════
  data.batch["input_ids"]       (bs=2, total_len=5120)
  data.batch["attention_mask"]  (bs=2, total_len=5120)
  data.batch["position_ids"]    (bs=2, 3, total_len=5120)
  data.batch["responses"]       (bs=2, response_len=1024)

═══════════════════════════════════════════════════════════════════
拆分 (micro_batch_size=1)
═══════════════════════════════════════════════════════════════════
  micro_batch_0: (1, 5120)
  micro_batch_1: (1, 5120)

═══════════════════════════════════════════════════════════════════
每个 micro_batch 的 _forward_micro_batch()
═══════════════════════════════════════════════════════════════════
  ① unpad (如果 use_remove_padding):
     input_ids: (1, 5120) → (1, total_nnz)  例: (1, 220)

  ② forward:
     input:  (1, total_nnz)
     logits: (1, total_nnz, 151936)

  ③ logprobs_from_logits:
     logits:  (1, total_nnz, 151936)
     labels:  (total_nnz,)  ← input_ids_rolled
     log_probs: (total_nnz,)

  ④ pad 回去:
     full_log_probs: (1, 5120)

  ⑤ 截取 response:
     log_probs: (1, 1024)  ← [:, -1025:-1]

═══════════════════════════════════════════════════════════════════
合并
═══════════════════════════════════════════════════════════════════
  concat(dim=0):
    old_log_probs: (bs=2, response_len=1024)
    entropys:      (bs=2, response_len=1024)  (如果 calculate_entropy)

═══════════════════════════════════════════════════════════════════
输出 (装进 DataProto 返回给 driver)
═══════════════════════════════════════════════════════════════════
  old_log_prob.batch["old_log_probs"]  (2, 1024)
  old_log_prob.batch["entropys"]       (2, 1024)
```

---

## 10. 面试题库

---

### Q1: Log Probability 是什么？为什么 PPO 用 log 而不直接用概率？

**高频指数**：⭐⭐⭐⭐⭐

**答案**：

```
Log Probability = ln(P(token | context))

用 log 的原因:
1. 数值稳定: 概率很小 (1/vocab_size ≈ 6e-6) → log 后在 -12 左右，不会 underflow
2. PPO ratio 的自然形式:
   ratio = P_new / P_old = exp(log P_new - log P_old)
   log 空间减法 = 概率空间除法 → 数值更稳定
3. 梯度缩放:
   ∂(log p)/∂θ = (1/p) * ∂p/∂θ
   对于小概率的正确 token → 1/p 很大 → 梯度大 → 更快学会
4. 与交叉熵一致: -log_prob 就是交叉熵 loss
```

---

### Q2: 为什么要重新计算 old_log_prob，而不是直接用 vLLM 的 logits？

**高频指数**：⭐⭐⭐⭐⭐

**答案**：

```
1. vLLM 的 logits 不可靠:
   - TP 切分后拼接有数值误差
   - FP16 vs BF16 精度差异
   - KV cache 优化可能影响精度

2. 必须和 FSDP 模型的计算一致:
   ratio = exp(log_prob_new - log_prob_old)
   分子 (Step 6) 和分母 (Step 4) 都用 FSDP 计算
   → 保证数值一致性 → ratio 准确

3. 分布式异步问题:
   vLLM 和 FSDP 可能在不同 GPU/节点
   vLLM 生成时的权重可能已经被 FSDP 更新了
   重算确保 old_log_prob 严格对应 rollout 时的权重
```

---

### Q3: 为什么 `torch.no_grad()` 是必须的？

**高频指数**：⭐⭐⭐⭐

**答案**：

```python
with torch.no_grad():
    entropy, log_probs = self._forward_micro_batch(...)

# 如果没有 no_grad():
# 1. PyTorch 会构建完整的计算图 (5120 个 token × 模型层数)
#    → 显存爆炸 (需要存所有中间激活值)
# 2. old_log_prob 会有 .grad
#    → 在 Step 6 的 loss 中, backward 同时更新 log_prob_new 和 old_log_prob
#    → PPO 公式被破坏

# PPO 的数学定义:
# L = E[min(r * A, clip(r) * A)], r = π_new / π_old
# π_old 是常数 → ∂L/∂θ 只通过 π_new
```

---

### Q4: `logprobs_from_logits` 的数学原理？

**高频指数**：⭐⭐⭐⭐

**答案**：

```python
# 两步:
# Step 1: log_softmax → 把 logits 转成 log 概率分布
logp_i = logits_i - log(sum(exp(logits_j)))
# 等价于: log(exp(logits_i) / sum(exp(logits_j)))
# 等价于: log(softmax(logits)_i)

# Step 2: gather → 取 label 对应位置的 log 概率
log_prob = logp[labels]
# labels = 正确 token 的 id
# 只取 "模型认为正确 token 的概率"

# 等价于交叉熵的负值:
# cross_entropy = -log_prob
# 所以 logprobs_from_logits 返回的就是 -cross_entropy(labels)

# 实际实现 (logprobs_from_logits_v2):
logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1))
logsumexp_values = torch.logsumexp(logits, dim=-1)
logprobs_labels = logits_labels - logsumexp_values
# = log(exp(x_label)) - log(sum(exp(x))) = log(x_label / sum(exp(x)))
```

---

### Q5: 为什么是 `[:, -response_len-1:-1, :]` 而不是 `[:, -response_len:, :]`？

**高频指数**：⭐⭐⭐⭐

**答案**：

```
自回归模型用位置 i 的 logits 预测位置 i+1 的 token:

序列: [P0, ..., P4094, R0, R1, ..., R1023]
                               ↑          ↑
                            位置 4095   位置 5119

要预测 response token [R0, ..., R1023] (1024 个):
  logits[4094]  → R0     (模型看到 P0...P4094, 预测下一个 = R0)
  logits[4095]  → R1
  ...
  logits[5118]  → R1023

所以取 logits[4095:5119] 共 1024 个 = logits[-1025:-1]

如果用 [-1024:]:
  = logits[4096:5120]:
  logits[4096] → R2     ← 错过了 R0!
  logits[5119] → ???    ← 预测的是 position 5120 (不存在的 future token)

偏移了 1 位: R0 没人预测, 最后一个预测的是不存在的东西
```

---

### Q6: `compute_log_prob` (Step 4) 和 `update_policy` 中的 `_forward_micro_batch` (Step 6) 有什么不同？

**高频指数**：⭐⭐⭐⭐

**答案**：

| | Step 4: compute_log_prob | Step 6: update_policy 中的 forward |
|---|---|---|
| **模式** | `model.eval()` | `model.train()` |
| **梯度** | `torch.no_grad()` | 正常构建计算图 |
| **目的** | 计算 π_old 下的 log_prob | 计算 π_new 下的 log_prob |
| **显存** | 低（不存激活值） | 高（需要存激活值给 backward） |
| **会影响参数吗** | 不会 | 会（通过 backward → optimizer.step） |
| **temperature** | 通常 = 1.0 | 可能调整（用于 exploration） |

```python
# Step 4:
self.actor_module.eval()
with torch.no_grad():
    log_probs = _forward_micro_batch(...)
# → 不记录梯度, 不更新参数

# Step 6:
self.actor_module.train()
log_prob_new = _forward_micro_batch(...)  # 有梯度
loss = ppo_loss(log_prob_new, old_log_probs, advantages)
loss.backward()  # 只更新 log_prob_new 对应的参数
```

**面试追问**：为什么 Step 4 的 temperature 可能和 Step 6 不同？

> Step 4 用 temperature=1.0 → 计算 "真实" 的 log_prob（无缩放），保证 old_log_prob 是 π_old 的真实概率。
> Step 6 可能用 temperature>1.0 → 软化概率分布 → 增加 exploration。但通常两处都用 temperature=1.0 以保持一致性。

---

### Q7: Remove Padding 优化是什么？为什么不总是用它？

**高频指数**：⭐⭐⭐

**答案**：

```
Remove Padding = 去掉 PAD token, 只对有效 token 做 attention

优点:
  - 计算量: O(seq_len²) → O(nnz²)
  - 5120² vs 220² → 约 540x 加速
  - 特别适合 prompt 长度不一致的场景

为什么不总是用:
  - 需要 flash_attn_varlen 支持
  - position_ids 的 unpad/pad 有额外开销
  - 对于序列长度均匀的场景收益不大
  - 多模态输入 (images) 的 unpad 实现更复杂
```

---

### Q8: Entropy 是什么？为什么要计算它？

**高频指数**：⭐⭐⭐

**答案**：

```
Entropy = -sum(P_i * log(P_i))
= 模型对当前 token 预测的 "不确定程度"

高 entropy (如 10.0): 模型很犹豫, 所有 token 概率差不多
低 entropy (如 0.1): 模型很确定, 某个 token 概率接近 1

用途:
  1. 监控指标: 如果 entropy 突然骤降 → 模型可能在模式坍塌
  2. Exploration bonus: 在 PPO loss 中加入 entropy bonus
     loss = pg_loss + β * entropy
     → 鼓励模型保持一定的探索性
  3. KL 散度近似:
     KL(π_new || π_old) ≈ (log_prob_old - log_prob_new)
     和 entropy 独立但相关
```

---

### Q9: 面试速记表

| 编号 | 问题 | 3 秒答案 |
|------|------|---------|
| Q1 | Log Probability | ln(P(token\|context))，数值稳定 + PPO ratio = exp(log_new - log_old) |
| Q2 | 为什么重算 | vLLM 精度不可靠 + 必须和 FSDP 数值一致 + torch.no_grad() 需要 |
| Q3 | no_grad 必要性 | 省显存（不存计算图）+ 保证 old_log_prob 在 PPO 中是常数 |
| Q4 | logprobs_from_logits | log_softmax → gather at label positions = -cross_entropy |
| Q5 | -1025:-1 切片 | 位置 i 的 logits 预测 i+1 的 token，需要错开一位 |
| Q6 | Step 4 vs Step 6 区别 | eval vs train、no_grad vs 需要梯度、计算 baseline vs 计算 loss |
| Q7 | Remove Padding | 去掉 PAD 只算有效 token，O(seq²)→O(nnz²)，需 flash_attn_varlen |
| Q8 | Entropy | 预测不确定度，监控模式坍塌 + exploration bonus |
