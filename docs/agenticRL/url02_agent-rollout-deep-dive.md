# url02：Step 2 详解 —— Agent Rollout（多轮生成 + 环境交互）

> 本文是对 `2026-06-02_rl-forward-backward-shapes-beginner-zh.md` 中 **Step 2** 的逐行深入解释。
> 目标读者：理解 Step 1 的数据准备，想深入理解 "模型怎么和环境交互多轮生成" 的同学。

---

## 目录

1. [Step 2 整体做什么](#1-step-2-整体做什么)
2. [前置：关键数据结构](#2-前置关键数据结构)
3. [Step 2a：初始化 running states](#3-step-2a初始化-running-states)
4. [Step 2b：vLLM 生成 action](#4-step-2bvllm-生成-action)
5. [Step 2c：追加 action tokens](#5-step-2c追加-action-tokens)
6. [Step 2d：env.step() 执行工具](#6-step-2denvstep-执行工具)
7. [Step 2e：追加 observation tokens](#7-step-2e追加-observation-tokens)
8. [Step 2f：多轮循环](#8-step-2f多轮循环)
9. [Step 2g：循环结束，padding 到统一长度](#9-step-2g循环结束padding-到统一长度)
10. [完整 Shape 变化追踪](#10-完整-shape-变化追踪)
11. [面试题库](#11-面试题库)

---

## 1. Step 2 整体做什么

**一句话**：把 gen_batch 发给 vLLM 做多轮生成，每轮模型输出 action → sandbox 执行 → 环境返回 observation → 追加到序列 → 下一轮...直到模型输出终止信号或达到轮次上限。

```text
┌─────────────────────────────────────────────────────────────────────┐
│  Step 2: Agent Rollout                                               │
│                                                                       │
│  输入: gen_batch (DataProto)                                         │
│    - input_ids:       (bs, 5120)   ← prompt token ids               │
│    - attention_mask:  (bs, 5120)   ← 有效 token=1, PAD=0            │
│    - position_ids:    (bs, 3, 5120)← Qwen2-VL MRoPE                │
│    - multi_modal_data: [Image, ...]                                  │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │  FOR each turn (最多 max_turns=2):                            │     │
│  │                                                               │     │
│  │  ① vLLM 生成 action tokens                                   │     │
│  │     "分析图片... <code>```python\n...```</code>"              │     │
│  │                                                               │     │
│  │  ② 追加 action tokens 到序列                                  │     │
│  │     action_mask = 1  (模型生成的)                             │     │
│  │                                                               │     │
│  │  ③ env.step() 执行工具                                       │     │
│  │     解析 code → HTTP POST sandbox → 执行 → 返回结果           │     │
│  │                                                               │     │
│  │  ④ 追加 observation tokens 到序列                             │     │
│  │     action_mask = 0  (环境返回的，不参与 loss)                │     │
│  │                                                               │     │
│  │  ⑤ 检查 done? → 是则退出循环                                  │     │
│  └─────────────────────────────────────────────────────────────┘     │
│                                                                       │
│  输出: gen_batch_output (DataProto)                                  │
│    - response:       (bs, 1024)   ← 生成的 response 部分            │
│    - action_mask:    (bs, 5120)   ← 全长 mask (action=1,obs=0)      │
│    - attention_mask: (bs, 5120)   ← 扩展后的 attention mask         │
│    - env_reward:     (bs, 1024)   ← 环境即时 reward                  │
│    - position_ids:   (bs, 3, 5120)← 扩展后的 position_ids           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 前置：关键数据结构

在深入代码前，理解这五个 list 是理解整个 Step 2 的关键：

```python
# 每个样本 i 有 5 个并行增长的 list
running_states[i]          # list of token ids: prompt + action1 + obs1 + action2 + ...
running_action_masks[i]    # 标记每个 token 是谁产生的 (1=模型, 0=环境/PAD)
running_attn_masks[i]      # 标记每个 token 是否有效 (1=有效, 0=PAD)
reward_tensor_list[i]      # 每个 token 的即时 reward (基本全 0)
vllm_input_list[i]         # vLLM 的输入格式: {"prompt_token_ids": [...], "multi_modal_data": ...}
```

**初始状态**（以 bs=1, 有效 prompt 长度=100 为例）：

```
running_states[0]:       [P0, P1, P2, ..., P99]         shape: (100,)
running_action_masks[0]: [1,  1,  1,  ...,  1 ]         shape: (100,)
running_attn_masks[0]:   [1,  1,  1,  ...,  1 ]         shape: (100,)
reward_tensor_list[0]:   [0,  0,  0,  ...,  0 ]         shape: (100,)
vllm_input_list[0]:      {"prompt_token_ids": [P0...P99], "multi_modal_data": {...}}
```

**关键设计**：这 5 个 list 同步增长。每次追加 action token 时，5 个 list **同时** `torch.cat` 追加对应内容，保证每个位置的索引一致。

---

## 3. Step 2a：初始化 running states

### 代码（parallel_env.py:168-181，简化）

```python
for i in range(batch_size):
    # 从 gen_batch 取出第 i 个样本的 prompt
    prompt_ids = prompts.batch['input_ids'][i, :].clone()       # (5120,) → 克隆到独立内存
    prompt_mask = prompts.batch['attention_mask'][i, :].clone() # (5120,)

    # 5 个 list 同步初始化
    running_states.append(prompt_ids)            # list[Tensor(5120,)]
    running_action_masks.append(prompt_mask)     # list[Tensor(5120,)]
    running_attn_masks.append(prompt_mask)       # list[Tensor(5120,)]
    reward_tensor = torch.zeros_like(prompt_ids) # (5120,) 全 0
    reward_tensor_list.append(reward_tensor)

    # vLLM 的输入格式
    vllm_input_list.append({
        "prompt_token_ids": prompt_ids[prompt_mask == 1].tolist(),
        #                     ↑ 只取有效 token（去掉 PAD）
        "multi_modal_data": multi_modal_data[i],
    })
```

### 逐行解释

**1. `.clone()` 的作用**

```python
prompt_ids = prompts.batch['input_ids'][i, :].clone()
#                                         ↑
# 创建独立副本，防止后续 torch.cat 影响原始数据
```

`prompts.batch['input_ids']` 是整个 DataProto 的共享数据。如果不 clone，后续 `torch.cat` 可能会在原始 tensor 上原地修改，造成数据污染。clone 后每个样本有自己独立的 tensor，可以安全地动态增长。

**2. `prompt_mask == 1` 过滤 PAD**

```python
prompt_ids[prompt_mask == 1].tolist()
# 例: prompt_mask = [1,1,1,...,1,0,0,...,0]
#                 ↑ 100个1   ↑ 5020个0
# prompt_ids[prompt_mask == 1] → 只取前 100 个有效 token
```

vLLM 接收的是不等长的 prompt token ids list，不需要 padding。padding 只是为了 DataLoader 的 batch 拼接，发送给 vLLM 时要把 PAD 去掉。

**3. reward_tensor 初始化为全 0**

```python
reward_tensor = torch.zeros_like(prompt_ids)
# dtype 继承 prompt_ids 的 dtype，值为 0
# prompt 部分的 reward 是 0：模型不应该因为 "被问到什么问题" 而被奖励
```

### 初始化后的状态

```
样本 0 (有效 prompt=100):
  running_states[0]:       [P0..P99 | PAD...PAD]           shape: (5120,)
  running_action_masks[0]: [1...1  | 0...0  ]              shape: (5120,)
  running_attn_masks[0]:   [1...1  | 0...0  ]              shape: (5120,)
  reward_tensor_list[0]:   [0...0  | 0...0  ]              shape: (5120,)
  vllm_input_list[0]:      prompt_token_ids = [P0..P99]    length: 100

样本 1 (有效 prompt=300):
  running_states[1]:       [P0..P299 | PAD...PAD]          shape: (5120,)
  running_action_masks[1]: [1...1   | 0...0  ]             shape: (5120,)
  running_attn_masks[1]:   [1...1   | 0...0  ]             shape: (5120,)
  reward_tensor_list[1]:   [0...0   | 0...0  ]             shape: (5120,)
  vllm_input_list[1]:      prompt_token_ids = [P0..P299]   length: 300
```

**注意**：此时 action_mask 中 prompt 部分是 1。但在 Step 5/6 计算 loss 时，用的是 `response_mask = action_mask[:, -response_len:]`，只取最后 1024 个位置。所以 prompt 的 mask 值在单轮（非 multi-turn）模式下不影响 loss。这个设计是为了代码的统一性——用同一个 mask 覆盖所有模式。

---

## 4. Step 2b：vLLM 生成 action

### 代码（parallel_env.py:192-196，简化）

```python
# 筛出本轮还需要继续的样本（done=False 的样本）
active_indices = [i for i in range(bs) if not done[i]]
active_vllm_inputs = [vllm_input_list[i] for i in active_indices]

# 调用 vLLM engine 生成
actions = vllm_engine.generate(
    prompts=active_vllm_inputs,
    sampling_params=SamplingParams(
        temperature=1.0,
        top_p=0.99,
        max_tokens=1024,
        stop=["</code>", "</tool_call>", "<|im_end|>"],
    ),
)
```

### vLLM 内部做了什么

```text
vLLM Engine (独立 GPU 进程):
┌──────────────────────────────────────────────────────┐
│ 1. 接收 prompt_token_ids + multi_modal_data         │
│                                                      │
│ 2. Vision Encoder: 图像 → image embeddings           │
│    Text Embedding:   token ids → text embeddings     │
│    拼接 → 多模态序列表示                              │
│                                                      │
│ 3. 自回归生成 (autoregressive generation):           │
│    for step in range(max_tokens):                   │
│        logits = model.forward(input_ids, kv_cache)  │
│        next_token = sample(logits, temp, top_p)     │
│        if next_token in stop_tokens: break          │
│        input_ids.append(next_token)                 │
│                                                      │
│ 4. 返回:                                             │
│    act.outputs[0].token_ids: [151665, 1234, ..., stop]│
│    act.outputs[0].text:      "解码后的文本"           │
└──────────────────────────────────────────────────────┘
```

### 生成结果示例

```python
# 模型输出 (50 个 token):
act = vllm_outputs[0]
act.outputs[0].token_ids   # → [151665, 1234, 5678, ..., 151658]
#                              ↑<think>                  ↑</code>
act.outputs[0].text        # → "<think>I need to analyze this image...</think>\n
#                                 <code>\n```python\nfrom PIL import Image\n...\n```\n</code>"

# Shape: (50,)
# Dtype: int64
```

### 为什么用 stop tokens 而不是 max_tokens 控制长度

```python
stop=["</code>", "</tool_call>", "<|im_end|>"]

# max_tokens = 1024 是安全上限（防止无限生成）
# stop tokens 是语义边界（"模型你该停在这里了"）

# 实际停止：
# 1. 模型吐出 </code> → vLLM 停止（即使还没到 1024）
# 2. 模型吐出 <answer>...</answer> 后继续吐 <|im_end|> → 停止
# 3. 如果模型一直在胡说八道 → max_tokens 兜底，强制截断
```

---

## 5. Step 2c：追加 action tokens

### 代码（parallel_env.py:207-223，简化）

```python
response_token_ids = torch.tensor(
    act.outputs[0].token_ids, dtype=torch.int64
)  # shape: (50,)

# ① 追加到完整序列
running_states[0] = torch.cat([running_states[0], response_token_ids])
# (5120,) + (50,) = (5170,)  ← 超出预分配长度！
# 没关系，后面会截断

# ② 创建全 1 的 action_mask
action_mask = torch.ones_like(response_token_ids)
# → [1, 1, 1, ..., 1]  shape: (50,)

# ③ action_mask 和 attn_mask 都追加
running_action_masks[0] = torch.cat([running_action_masks[0], action_mask])
# (5120,) + (50,) = (5170,)  ← 全 1

running_attn_masks[0] = torch.cat([running_attn_masks[0], action_mask])
# (5120,) + (50,) = (5170,)  ← 全 1
# 注意：对 attn_mask 也是 1，因为模型生成时 attend to 自己前面的 action token

# ④ reward 追加全 0，最后一个 token 加即时 reward
action_reward = torch.zeros_like(response_token_ids)  # (50,) 全 0
reward_tensor_list[0] = torch.cat([reward_tensor_list[0], action_reward])
# (5120,) + (50,) = (5170,)
reward_tensor_list[0][-1] += immediate_reward  # 最后一个 token 加即时 env reward
```

### 逐行解释

**① `torch.cat` 追加序列**

```python
running_states[0] = torch.cat([running_states[0], response_token_ids])

# 之前: [P0..P99 | PAD(0)...PAD(0)]      shape: (5120,)
# 之后: [P0..P99 | PAD(0)...PAD(0) | A0..A49]  shape: (5170,)
#                                      ↑ 50 个 action token 追加到了最后
```

**为什么允许超出 5120？** 此时不截断，等整个循环结束后统一处理。循环中的序列长度可能先超再被截断——Python list 动态增长，`torch.cat` 后 shape 自然变大。

**② `torch.ones_like` 创建 action mask**

```python
action_mask = torch.ones_like(response_token_ids)
# response_token_ids 是 (50,)，dtype=int64
# action_mask 也是 (50,)，dtype=int64，全是 1
```

"这些 token 是模型生成的 → 标记为 1 → 将来参与 loss 计算"。

**③ attention_mask 也是全 1**

```python
running_attn_masks[0] = torch.cat([running_attn_masks[0], action_mask])
# 这里的 action_mask 是全 1
```

模型在生成下一个 token 时，需要看到之前所有自己生成的 token（自回归的 causal attention）。所以 attention mask 对 action token 是 1。

**④ reward 全 0，只有最后一个 token 加即时 reward**

```python
action_reward = torch.zeros_like(response_token_ids)  # 50 个 0
reward_tensor_list[0][-1] += immediate_reward  # 通常为 0
```

这里 env_reward 通常为 0。真正的 reward（judge 打分）在 Step 3 才来。env_reward 的设计是为了一些 "即时反馈" 场景（如格式错误给 -0.1），但在 smoke config 中基本不用。

### 追加后的状态

```
running_states[0]:       [P0..P99 | 0..0      | A0..A49]
                          ↑ 100    ↑ 5020 PAD  ↑ 50 action
                          shape: (5170,)

running_action_masks[0]: [1...1   | 0...0     | 1...1]
                          ↑ prompt ↑ PAD       ↑ action=1

running_attn_masks[0]:   [1...1   | 0...0     | 1...1]
                          ↑          ↑ PAD=0    ↑ action=1

reward_tensor_list[0]:   [0...0   | 0...0     | 0...0]
                          ↑ 全 0 (这一步还没 reward)
```

---

## 6. Step 2d：env.step() 执行工具

### 代码（parallel_env.py:199-204，简化）

```python
# 筛选本轮有 action 的样本
active_actions = [actions[i] for i in range(len(actions))]
active_indices = [...]

# 调用环境执行
obs_results = env.step(active_indices, active_actions)
observations, rewards, dones, info = obs_results
```

### env.step 内部流程（deepeyesv2.py:124-175）

```text
┌────────────────────────────────────────────────────────────────┐
│  env.step(action) → observation + reward + done               │
│                                                                │
│  1. extract_python_code(action.text):                         │
│     从 <code>```python ... ```</code> 中提取纯 Python 代码     │
│                                                                │
│  2. request_jupyter_execution(code):                          │
│     HTTP POST → sandbox (Jupyter Kernel)                      │
│     sandbox 执行 python 代码                                   │
│     返回: {stdout, stderr, images, error}                     │
│                                                                │
│  3. 构造 observation 文本:                                     │
│     "<|im_end|>                                               │
│      <|im_start|>user                                         │
│      Code execution result:                                   │
│      stdout:                                                  │
│      ```                                                      │
│      [执行输出/图片描述]                                        │
│      ```                                                      │
│      stderr:                                                  │
│      ```                                                      │
│      [错误输出]                                                │
│      ```                                                      │
│      <|im_end|>                                               │
│      <|im_start|>assistant                                    │
│      <think>"                                                 │
│                                                                │
│  4. 检查 done:                                                │
│     - 模型输出了 <answer>...</answer> → done=True             │
│     - 达到 max_turns → done=True                              │
│     - 否则 → done=False                                       │
└────────────────────────────────────────────────────────────────┘
```

### 关键设计：为什么 observation 文本以 `<|im_start|>assistant\n<think>` 开头

```
这个设计是为了让模型在下一轮自然地 "继续思考"：

Turn 1 结束时的序列:
  ...<code>```python...```</code>  ← 模型的 action

Observation 文本:
  <|im_end|>                       ← 结束上一轮 assistant 的消息
  <|im_start|>user                 ← 用户（环境）的消息开始
  Code execution result: ...       ← 工具执行结果
  <|im_end|>                       ← 用户消息结束
  <|im_start|>assistant            ← assistant 开始
  <think>                          ← 引导模型继续思考

这样 vLLM 看到 "<think>" 后，自然会接着生成思考内容和下一步 action。
这是 prompt engineering 在 token 层面的体现。
```

---

## 7. Step 2e：追加 observation tokens

### 代码（parallel_env.py:235-260，简化）

```python
# 获取 observation token ids
obs_token_ids_model = obs['prompt_token_ids_model'].to(device)
# shape: (30,)  — 假设 observation 文本被 tokenize 成 30 个 token

# ① 追加到完整序列
running_states[0] = torch.cat([running_states[0], obs_token_ids_model])
# (5170,) + (30,) = (5200,)

# ② ★ 创建全 0 的 obs_mask
obs_mask = torch.zeros(len(obs_token_ids_model), dtype=torch.int64)
# torch.zeros(30) → [0, 0, 0, ..., 0]  shape: (30,)
#   ↑
#   关键: observation token → mask=0

# ③ action_mask 追加全 0
running_action_masks[0] = torch.cat([running_action_masks[0], obs_mask])
# (5170,) + (30,) = (5200,)
# 追加的 30 个全是 0

# ④ ★★ attention_mask 追加全 1
attn_mask = torch.ones(len(obs_token_ids_model), dtype=torch.int64)
# torch.ones(30) → [1, 1, 1, ..., 1]  shape: (30,)
running_attn_masks[0] = torch.cat([running_attn_masks[0], attn_mask])
# (5170,) + (30,) = (5200,)
# 追加的 30 个全是 1

# ⑤ reward 追加全 0
obs_reward = torch.zeros(len(obs_token_ids_model))
reward_tensor_list[0] = torch.cat([reward_tensor_list[0], obs_reward])
# (5170,) + (30,) = (5200,)
```

### 核心区分：两个 mask 在 obs token 上的分歧

这是整个 Step 2 最重要的设计点：

```
对于同一个 observation token (比如 "Code execution result:\nstdout:\n[plot]\n"):

┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  action_mask = 0                                               │
│  "你不是模型生成的 → 不参与 policy loss → 不产生梯度"           │
│  → 如果这个 obs 包含了正确答案，模型不会因为 "预测这个 obs"     │
│    而被奖励                                                     │
│                                                                │
│  attention_mask = 1                                            │
│  "但模型需要看到你 → 才能决定下一步做什么"                       │
│  → 在 next token prediction 时 attend to 这个 obs              │
│  → 影响 logits 的生成                                          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 追加后的完整状态（Turn 1 结束）

```
running_states[0]:       [P0..P99 | 0..0      | A0..A49 | O0..O29]
                          ↑ 100    ↑ 5020 PAD  ↑ 50 act  ↑ 30 obs
                          shape: (5200,)

running_action_masks[0]: [1...1   | 0...0     | 1...1   | 0...0]
                          ↑ prompt ↑ PAD       ↑ action  ↑ obs=0 ★

running_attn_masks[0]:   [1...1   | 0...0     | 1...1   | 1...1]
                          ↑ prompt ↑ PAD       ↑ action  ↑ obs=1 ★★

reward_tensor_list[0]:   [0...0   | 0...0     | 0...0   | 0...0]
                          ↑ 全 0
```

---

## 8. Step 2f：多轮循环

### 循环控制

```python
# 循环最多 max_turns 轮
for turn in range(max_turns):  # max_turns=2
    # 1. 筛出活跃的样本（done=False）
    active_indices = [i for i in range(bs) if active_mask[i]]
    if not active_indices:
        break  # 所有样本都 done 了

    # 2. vLLM 生成 → action tokens → 追加
    # 3. env.step() → obs tokens → 追加
    # 4. 检查 done，更新 active_mask

    # 如果 done=True（模型输出了 <answer>）:
    #   active_mask[i] = False  → 不再参与下一轮
```

### Turn 2 示例

假设 Turn 1 后模型拿到了 observation，现在 Turn 2 继续：

```python
# vLLM 生成 action2:
# "<think>The answer is 4</think>\n<answer>4</answer>"
# act2.outputs[0].token_ids → shape: (40,)

# 追加 action2 tokens:
running_states[0] = torch.cat([running_states[0], action2_token_ids])
# (5200,) + (40,) = (5240,)

running_action_masks[0] = torch.cat([running_action_masks[0], ones(40)])
# + 40 个 1

running_attn_masks[0] = torch.cat([running_attn_masks[0], ones(40)])
# + 40 个 1

reward_tensor_list[0] = torch.cat([reward_tensor_list[0], zeros(40)])
# + 40 个 0

# 因为 <answer> 出现 → done=True → active_mask[0]=False
# 不再进入 Turn 3
```

### Turn 2 结束的完整状态

```
running_states[0]:       [P0..P99|0..0       |A0..A49|O0..O29|A0..A39]
                          ↑ 100  ↑ 5020 PAD   ↑ 50    ↑ 30    ↑ 40
                          shape: (5240,)

running_action_masks[0]: [1...1 |0...0       |1...1  |0...0  |1...1]
                          ↑prompt↑ PAD        ↑act1  ↑obs1  ↑act2

running_attn_masks[0]:   [1...1 |0...0       |1...1  |1...1  |1...1]
                          ↑prompt↑ PAD=0      ↑act1  ↑obs1  ↑act2

reward_tensor_list[0]:   [0...0 |0...0       |0...0  |0...0  |0...0]
                          ↑ 全 0 (现阶段)
```

---

## 9. Step 2g：循环结束，padding 到统一长度

### 代码（parallel_env.py:277-319，简化）

```python
# 第 1 步：截断到 max_total_length
max_total_length = prompt_len + response_len = 4096 + 1024 = 5120
running_states = [state[:max_total_length] for state in running_states]

# 第 2 步：pad 到 max_total_length
state_tensor = pad_2d_list_to_length(
    running_states, pad_value=pad_token_id, max_len=max_total_length
)
# 输入: list of 不等长 Tensor → 输出: (bs, 5120)

action_mask_tensor = pad_2d_list_to_length(
    running_action_masks, pad_value=0, max_len=max_total_length
)
# → (bs, 5120)

attn_mask_tensor = pad_2d_list_to_length(
    running_attn_masks, pad_value=0, max_len=max_total_length
)
# → (bs, 5120)

reward_tensor = pad_2d_list_to_length(
    reward_tensor_list, pad_value=0.0, max_len=max_total_length
)
# → (bs, 5120)
```

### 截断 + pad 的详细过程

```python
# 以样本 0 为例: 实际长度=5240（超出 max_total_length=5120）

# Step 1: 截断
running_states[0] = running_states[0][:5120]
# 从 5240 截断到 5120
# 被截掉的部分: 最后 120 个 token (部分 action2 tokens)
# → 这是一个问题: action2 被截断了！

# Step 2: pad（长度不足 5120 的补齐）
# 样本 1: 实际长度=4800（没超）
running_states[1] = pad_to_length(running_states[1], 5120, pad_token_id)
# [P0..P299 | A0..A30 | O0..O20 | A0..A20 | PAD...PAD]
# ↑ 4800 个有效 token                     ↑ 320 个 PAD
```

### pad_2d_list_to_length 的工作原理

```python
def pad_2d_list_to_length(tensors, pad_value, max_len):
    """
    tensors: list[Tensor], 每个 Tensor 是一维的 (长度不等)
    max_len: 目标长度

    返回: (len(tensors), max_len) 的二维 Tensor
    """
    padded = []
    for t in tensors:
        if len(t) < max_len:
            # 补 PAD
            padding = torch.full((max_len - len(t),), pad_value, dtype=t.dtype)
            t = torch.cat([t, padding])
        else:
            # 截断
            t = t[:max_len]
        padded.append(t)
    return torch.stack(padded)  # (bs, max_len)
```

### 截断的后果

```python
# 如果 response_len=1024 不够大:
# → action2 的后半部分被截断
# → <answer> 可能被截断 → reward 为 0（judge 看不到完整答案）
# → 模型学不到有用的信号

# 设计 tradeoff:
# response_len 必须大于典型的多轮交互总长度
# smoke config: 1024 (用于快速测试)
# 正式训练: 通常 2048-4096
```

### 最终返回的 DataProto

```python
return DataProto.from_dict(tensors={
    "response":       state_tensor[:, -response_len:],   # (bs, 1024)
    #                 ↑ 只取最后 response_len 个 token
    #                 例: state_tensor[0, -1024:] → 取样本0的最后1024个token

    "action_mask":    action_mask_tensor,                 # (bs, 5120)
    #                 全长 mask

    "attention_mask": attn_mask_tensor,                   # (bs, 5120)
    #                 全长 mask

    "position_ids":   position_ids_tensor,                # (bs, 3, 5120)
    #                 扩展后的 position_ids

    "env_reward":     reward_tensor[:, -response_len:],   # (bs, 1024)
    #                 只取最后 response_len 的环境 reward

    "tool_cnt":       tool_call_tensor,                   # (bs, 1)
    #                 记录每个样本调用了几次工具
})
```

### 为什么 response 只取 [-response_len:]？

```python
# 设计考量:
# 1. prompt 部分在 driver 已经有了 (input_ids)
# 2. response 部分才是新增的 (模型生成的 + 环境返回的)
# 3. 取最后 1024 个 token 而非 "新增的全部"
#    → 因为输入 shape 是固定的 (bs, response_len)
#    → 如果实际 response 长度 < 1024，左边补 PAD
#    → 如果实际 response 长度 > 1024，左边被截断

# response 的结构 (最后 1024 个 token):
# [action1 | obs1 | action2 | PAD...PAD]
#  位置取决于实际长度和截断情况
```

---

## 10. 完整 Shape 变化追踪

以一条样本为例，跟踪从输入到输出的完整 shape 变化：

```text
═══════════════════════════════════════════════════════════════════
初始状态 (输入 gen_batch)
═══════════════════════════════════════════════════════════════════
  input_ids:       (2, 5120)  ← 其中有效 prompt = 100/300 个 token
  attention_mask:  (2, 5120)
  position_ids:    (2, 3, 5120)

═══════════════════════════════════════════════════════════════════
初始化 running states (Step 2a)
═══════════════════════════════════════════════════════════════════
  样本 0:
    running_states[0]:         (5120,)  ← clone 的 input_ids[0]
    running_action_masks[0]:   (5120,)  ← clone 的 attention_mask[0]
    running_attn_masks[0]:     (5120,)  ← clone 的 attention_mask[0]
    reward_tensor_list[0]:     (5120,)  ← zeros_like
    vllm_input_list[0]:        prompt_token_ids = [P0..P99]  (长度 100)

═══════════════════════════════════════════════════════════════════
Turn 1: vLLM 生成 (Step 2b)
═══════════════════════════════════════════════════════════════════
  act.outputs[0].token_ids:  (50,)  ← action1 tokens

═══════════════════════════════════════════════════════════════════
Turn 1: 追加 action (Step 2c)
═══════════════════════════════════════════════════════════════════
  running_states[0]:         (5120,) + (50,)  = (5170,)
  running_action_masks[0]:   (5120,) + 1s(50) = (5170,)
  running_attn_masks[0]:     (5120,) + 1s(50)  = (5170,)
  reward_tensor_list[0]:     (5120,) + 0s(50)  = (5170,)

═══════════════════════════════════════════════════════════════════
Turn 1: env.step() + 追加 obs (Step 2d-2e)
═══════════════════════════════════════════════════════════════════
  obs_token_ids:             (30,)  ← observation tokens

  running_states[0]:         (5170,) + (30,)  = (5200,)
  running_action_masks[0]:   (5170,) + 0s(30) = (5200,)  ← obs=0
  running_attn_masks[0]:     (5170,) + 1s(30)  = (5200,)  ← obs=1
  reward_tensor_list[0]:     (5170,) + 0s(30)  = (5200,)

═══════════════════════════════════════════════════════════════════
Turn 2: vLLM 生成 + 追加 action (Step 2f)
═══════════════════════════════════════════════════════════════════
  act2.outputs[0].token_ids: (40,)  ← action2 tokens (含 <answer>)

  running_states[0]:         (5200,) + (40,)  = (5240,)
  running_action_masks[0]:   (5200,) + 1s(40) = (5240,)
  running_attn_masks[0]:     (5200,) + 1s(40)  = (5240,)
  reward_tensor_list[0]:     (5200,) + 0s(40)  = (5240,)

  done=True → 停止

═══════════════════════════════════════════════════════════════════
Padding (Step 2g)
═══════════════════════════════════════════════════════════════════
  截断: running_states[0] = running_states[0][:5120]  → (5120,)
  pad:   统一到 max_len=5120

═══════════════════════════════════════════════════════════════════
最终输出 (gen_batch_output)
═══════════════════════════════════════════════════════════════════
  response:       (2, 1024)   ← state_tensor[:, -1024:]
  action_mask:    (2, 5120)   ← 全长
  attention_mask: (2, 5120)   ← 全长
  env_reward:     (2, 1024)   ← reward_tensor[:, -1024:]
  position_ids:   (2, 3, 5120)
  tool_cnt:       (2, 1)
```

---

## 11. 面试题库

---

### Q1: Agent Rollout 的完整流程？vLLM 和 Sandbox 分别负责什么？

**高频指数**：⭐⭐⭐⭐⭐（必问）

**答案**：

```
vLLM 负责 "大脑"：接收 prompt + 图像 → 生成 action token → 返回文本
Sandbox 负责 "手"：接收 action 文本中的 Python 代码 → 执行 → 返回 stdout/stderr/images

一次完整的交互:
  1. vLLM 生成 <code>```python\nplt.plot(x,y)\nplt.show()\n```</code>
  2. extract_python_code() 提取纯 Python 代码 "plt.plot(x,y)\nplt.show()"
  3. HTTP POST → Sandbox (Jupyter Kernel) 执行代码
  4. Sandbox 返回 {"stdout": "", "images": [base64_png], "stderr": ""}
  5. 构造 observation 文本: "Code execution result:\nstdout:\n[image]\n"
  6. tokenize → obs_token_ids → 追加到序列
  7. 下一轮 vLLM 基于扩容后的序列继续生成
```

**加分回答**：解释为什么需要 Sandbox 而不是直接 exec()——安全性（防止 rm -rf /）、隔离性（多用户并发）、可复现（干净的 Python 环境）。

---

### Q2: 为什么 action_mask 对 obs token 是 0，attention_mask 对 obs token 是 1？

**高频指数**：⭐⭐⭐⭐

**答案**：

```
这是 "forward 看到，backward 不管" 的设计：

attention_mask 控制 Transformer forward 时的 attention:
  - obs token 包含工具执行结果
  - 模型必须 "看到" 这些结果才能决定下一步
  - 所以 attention_mask=1（参与 attention 计算）

action_mask 控制 loss backward 时的梯度:
  - obs token 是环境产生的，不是模型生成的
  - 如果 obs 参与 loss，模型会学习 "预测环境反馈"
  - 而不是学习 "在什么状态下做什么决策"
  - 所以 action_mask=0（不产生梯度）

类比:
  学生解题时 "看到题目" (= attention_mask=1)
  但 "题目本身" 不是学生的回答，不能据此给分 (= action_mask=0)
```

---

### Q3: 为什么要用 stop tokens 控制生成，而不是只用 max_tokens？

**高频指数**：⭐⭐⭐

**答案**：

```
max_tokens = 1024: 安全上限，防止模型无限生成
stop tokens: 语义边界，告诉 vLLM "模型该在此处停止"

stop tokens 设计:
  stop=["</code>", "</tool_call>", "<|im_end|>"]

  1. </code> → action 代码块的结束标记
     模型生成了完整代码块后应当立刻停止
     不应该让模型在代码块后面继续胡言乱语

  2. </tool_call> → tool call 格式的结束标记
     和训练数据的格式对齐

  3. <|im_end|> → Qwen2 对话格式的结束标记
     完整的 assistant 消息的结束

为什么不能只用 max_tokens:
  - max_tokens 不知道语义边界
  - 可能在代码中间截断 → 代码解析失败 → reward=0
  - stop tokens 保证生成的完整性
  - max_tokens 是兜底机制（防止无限循环）
```

---

### Q4: 多轮交互中，如果有一轮的 action 或 obs 特别长怎么办？

**高频指数**：⭐⭐⭐

**答案**：

```
多层截断机制:

1. vLLM max_tokens: 单轮 action 最长 1024
2. max_total_length = prompt_len + response_len = 5120
3. running_states = [state[:5120] for state in running_states]
   所有超出 5120 的 token 直接截断

问题: 如果前面几轮太长，后面的轮次可能没机会生成
  → response_len 需要根据实际任务调整
  → 简单任务 1024 够用，复杂推理需要 2048-4096

更好的做法:
  - 分层截断: 保证最后一轮至少有 min_last_turn_tokens
  - Dynamic response_len: 根据 prompt 复杂度预估需要多长
  - 监控: 记录 truncation_rate → 如果过高就增大 response_len
```

---

### Q5: 为什么 observation 文本要以 `<|im_start|>assistant\n<think>` 结尾？

**高频指数**：⭐⭐⭐

**答案**：

```
这是 "prompt engineering on token level" 的体现:

observation 文本结构:
  <|im_end|>                    ← 结束之前的 assistant 消息
  <|im_start|>user              ← 环境 = "用户"
  Code execution result: ...    ← 工具返回
  <|im_end|>                    ← 用户消息结束
  <|im_start|>assistant         ← 模型 = "助手"
  <think>                       ← 引导模型开始思考

作用:
  1. 利用 Qwen2 的对话格式 (chat template) 区分角色
  2. <think> 是模型训练时见过的思考前缀
     → 模型看到它后自动进入 "分析-推理-回答" 模式
  3. 不需要额外的 prompt 指令 (如 "请继续分析")
     → 格式本身就隐含了 "该你说话了"

如果没有这个设计:
  → 模型可能不知道该干什么
  → 可能直接输出 <answer> 而不是先 <think>
  → 推理质量下降
```

---

### Q6: 多轮 Rollout 如何保持 position_ids 的连续性？

**高频指数**：⭐⭐⭐（VLM 岗位高频）

**答案**：

```python
# 每追加一段新 token，position_ids 需要同步扩展

# 初始: position_ids (3, 5120)
# 追加 action (50 token) 后，需要追加 (3, 50) 的 position_ids

# 对于纯文本 token（action 和 obs 都是文本）:
# 三维共享同一个递增的位置:
last_pos = current_length - 1  # 当前序列最后一个位置的 position

new_positions = torch.arange(last_pos + 1, last_pos + 1 + new_length)
# → [501, 502, 503, ..., 550]

# 三个维度都设为这个值:
new_pos_ids = new_positions.unsqueeze(0).expand(3, -1)
# → (3, 50), 每行都是 [501, 502, ..., 550]

# 对于图像 token（如果有新的图片输入）:
# height 和 width 维度用二维网格坐标
# temporal 维度保持一致
```

---

### Q7: 如果模型在多轮交互中输出了完全无法解析的 action 怎么办？

**高频指数**：⭐⭐⭐

**答案**：

```python
# 当前做法（deepeyesv2.py 中的 extract_python_code）:
try:
    code = extract_python_code(action_text)
    # 正则: r"```python\n(.*?)```"
except:
    code = None

if code is None or code.strip() == "":
    # 构造一个 "空 observation"
    observation = "Code execution result:\nNo valid Python code found.\n"
    reward = 0.0
    # 不标记 done=True → 模型有机会重试

# 更好的做法:
# 1. 给负 reward (format penalty = -0.1)
# 2. 记录 parse_error 频率 → 如果太高（>10%），说明需要改进指令
# 3. 在 prompt 中加 format instruction
# 4. 可以用 constrained decoding 强制输出符合格式的代码块
```

---

### Q8: vLLM 和 Sandbox 是同一个进程吗？它们之间怎么通信？

**高频指数**：⭐⭐⭐

**答案**：

```
不是同一个进程。三个独立的组件:

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Driver 进程     │     │  vLLM Engine    │     │  Sandbox        │
│  (CPU)          │     │  (GPU)          │     │  (CPU, HTTP)    │
│                 │     │                 │     │                 │
│  协调 rollout   │────▶│  接收 prompt     │     │  Jupyter Kernel │
│                 │     │  生成 action    │     │  Python 3.10    │
│                 │     │  返回 token_ids │     │  PIL, numpy...  │
│                 │     │                 │     │                 │
│                 │────────────────────────────▶│                 │
│                 │     HTTP POST code          │  执行代码       │
│                 │◀────────────────────────────│  返回结果       │
│                 │     {stdout, images, ...}   │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘

通信方式:
  1. Driver ↔ vLLM: Ray RPC (共享内存 + gRPC)
     - 高频调用 (每轮生成一次)
     - 大数据量 (token ids, images)

  2. Driver ↔ Sandbox: HTTP REST API
     - 每轮 action 执行一次
     - JSON 格式的请求/响应
     - sandbox 通常部署为独立的微服务 (Docker/K8s)
```

---

### Q9: 多轮 Rollout 中，如何处理不同样本的轮次不同？

**高频指数**：⭐⭐⭐⭐

**答案**：

```python
# 用 active_mask 标记哪些样本还需要继续
active_mask = [True, True]  # 初始：两个样本都活跃

# Turn 1:
for turn in range(max_turns):
    active_indices = [i for i in range(bs) if active_mask[i]]
    # → [0, 1]  两个样本都参与

    # vLLM 只对活跃样本生成
    actions = vllm_engine.generate(
        prompts=[vllm_input_list[i] for i in active_indices]
    )

    # env.step 只对活跃样本执行
    observations, rewards, dones, info = env.step(active_indices, actions)

    # 更新 active_mask
    for i, done in zip(active_indices, dones):
        if done:
            active_mask[i] = False

    # Turn 1 结束后:
    # 样本0: done=True  → active_mask[0] = False
    # 样本1: done=False → active_mask[1] = True

# Turn 2:
# active_indices = [1]  ← 只有样本1继续
# 样本0 不参与，它的 running_states 保持不变

# 循环结束后统一 pad 到相同长度
# 样本0 长度=200, 样本1 长度=500
# pad 后: 两个都是 5120，但 pad 位置不同
```

**加分回答**：这种设计的好处——vLLM 不需要等待慢的样本，快的样本提前结束可以节省计算。坏处——batch 利用率下降（batch 中样本越来越少）。

---

### Q10: 面试速记表

| 编号 | 问题 | 3 秒答案 |
|------|------|---------|
| Q1 | Rollout 流程 | vLLM 生成 action → sandbox 执行 → 追加 obs → 循环 → pad |
| Q2 | 两个 mask 的差异 | attention_mask: obs=1 (forward 看到), action_mask: obs=0 (backward 不管) |
| Q3 | stop tokens | 语义边界（代码块/消息结束标记），max_tokens 只是兜底 |
| Q4 | 超长处理 | max_tokens + 全局截断 + response_len tradeoff |
| Q5 | obs 格式 | 利用 Qwen2 chat template：`<|im_start|>assistant\n<think>` 引导继续思考 |
| Q6 | position_ids | 文本部分三维共享递增位置，图像部分 (h,w) 用二维网格 |
| Q7 | 解析失败 | 返回空结果 + 零 reward + 给模型重试机会 |
| Q8 | 进程通信 | Driver↔vLLM: Ray RPC, Driver↔Sandbox: HTTP REST |
| Q9 | 不等轮次 | active_mask 跟踪 + 统一 pad |

---

## 补充问答

---

### Q11: vLLM 就是我们要优化的 QwenVL 本地模型吗？

**高频指数**：⭐⭐⭐⭐⭐

**答案：是的，但两者是同一个模型的不同"身份"。**

```
┌──────────────────────────────────────────────────────────────┐
│  Qwen2-VL 模型（同一份权重）                                    │
│                                                               │
│  ┌─────────────────────┐    ┌─────────────────────────────┐  │
│  │ vLLM Engine (推理)   │    │ FSDP Worker (训练)           │  │
│  │                     │    │                              │  │
│  │ 作用: 生成 action    │    │ 作用: 计算 log_prob + 更新权重│  │
│  │ 模式: inference     │    │ 模式: train                  │  │
│  │ 精度: FP16          │    │ 精度: BF16 + 梯度            │  │
│  │ 优化: KV cache,     │    │ 优化: FSDP sharding,         │  │
│  │       continuous     │    │       gradient checkpointing │  │
│  │       batching       │    │                              │  │
│  │ 不需要梯度           │    │ 需要完整的计算图 + 梯度      │  │
│  └─────────────────────┘    └─────────────────────────────┘  │
│                                                               │
│  权重同步: 每次 optimizer.step() 后，FSDP 的权重 → vLLM       │
└──────────────────────────────────────────────────────────────┘
```

**关键流程**：

```python
# Step 2: vLLM 用当前权重做推理 → 无梯度
actions = vllm_engine.generate(prompts)  # 用 θ_current 生成

# Step 4: FSDP 用同样的权重计算 old_log_prob → 无梯度
old_log_prob = fsdp_worker.compute_log_prob(batch)  # 用 θ_current 计算

# Step 6: FSDP 更新权重 → 有梯度
loss.backward()
optimizer.step()  # θ_current → θ_new

# 权重同步：vLLM 拿到新权重
vllm_engine.update_weights(fsdp_worker.state_dict())
```

**为什么需要两套系统？**

```
vLLM 专为推理优化：
  - PagedAttention (显存高效 KV cache)
  - Continuous batching (动态批处理)
  - 高吞吐量生成 (tokens/s)
  - 但不支持反向传播

FSDP 专为训练优化：
  - 权重分片 (sharding) 到多 GPU
  - 梯度累积 + 同步
  - 但生成速度远不如 vLLM

所以分开部署：vLLM 做生成，FSDP 做训练
 → 各取所长，避免在同一个进程里同时做推理和训练
```

**面试可能追问**：

> Q: vLLM 和 FSDP 的权重怎么保持同步？
>
> A: 每次 `optimizer.step()` 后，FSDP worker 把更新后的权重通过 Ray RPC 或共享内存传给 vLLM engine。vLLM 调用 `load_weights()` 更新自己的 KV cache 相关结构。同步是 **阻塞的**——必须等 vLLM 拿到新权重后才能开始下一轮 rollout。

> Q: 为什么不让 vLLM 也支持 backward？
>
> A: vLLM 的核心优化（PagedAttention、prefix caching、continuous batching）都和反向传播不兼容。反向传播需要保存完整的中间激活值（activations），这会破坏 vLLM 的内存优化。分开部署是当前的最优实践。

---

### Q12: gen_batch 里的内容是程序员指定的吗？还是每种任务都差不多？

**高频指数**：⭐⭐⭐

**答案：一半是框架固定的（所有任务都需要），一半是任务相关的。**

```python
# ===== 固定部分：所有 Agentic RL 任务都需要（框架自动处理）=====
gen_batch.batch = {
    "input_ids":       ...,   # DataLoader 从数据集 tokenize
    "attention_mask":  ...,   # DataLoader 的 collate_fn 自动生成
    "position_ids":    ...,   # DataLoader 根据模型类型生成
}

# ===== 任务相关部分：由数据集构建者 + 配置文件决定 =====
gen_batch.non_tensor_batch = {
    "multi_modal_data":    [...],  # VLM 任务才有（纯文本任务没有）
    "raw_prompt":          [...],  # 取决于数据集内容
    "env_name":            "deepeyesv2",  # 配置文件中指定
    "extra_info":          {...},  # reward function 需要的信息
    "origin_multi_modal_data": [...],  # 原图给 sandbox
}
```

| 内容 | 谁决定 | 适用范围 |
|------|--------|---------|
| `input_ids` | DataLoader (数据集 tokenize 后) | ✅ 所有 LLM 任务 |
| `attention_mask` | DataLoader (collate_fn) | ✅ 所有 LLM 任务 |
| `position_ids` | DataLoader (根据模型架构) | ✅ 所有 LLM 任务 |
| `multi_modal_data` | 数据集构建者 | ❌ VLM 任务才有 |
| `env_name` | 配置文件 (yaml) | ❌ 每种 Agent 环境不同 |
| `extra_info` | 数据集 + reward function 设计者 | ❌ 每种任务不同 |
| `raw_prompt` | 数据集 | ✅ 通常都有 |

**程序员要做的事**：

```python
# 为新任务接入框架，需要写：
# 1. 数据预处理脚本：把原始数据 → tokenize → DataProto 格式
# 2. 环境适配器：实现 env.step(action) → observation
# 3. Reward function：实现 compute_score(response, extra_info) → float
# 4. 配置文件：指定 env_name, max_turns, response_len 等

# 框架代码（DataProto、pop、union、GRPO、FSDP）是通用的，不需要改
```

---

### Q13: `action_mask[:, -response_len:]` 语法详解

**高频指数**：⭐⭐（基础语法，但面试可能作为代码阅读题出现）

**答案**：

```python
action_mask[:, -response_len:]
#          ↑   ↑
#          │   └── Python 负索引切片：取最后 response_len 个元素
#          └────── 取所有行（batch 维度的所有样本）

# 完整拆解：
# action_mask:    shape (bs, total_len)    例如 (2, 5120)
# [:, -1024:]  → "所有行，最后 1024 列"
# 结果:           shape (2, 1024)
```

**直观图示**：

```python
action_mask = tensor([[1,1,1,0,0,1,1,1,0,0],    # shape: (2, 10)
                      [1,1,0,0,1,1,1,0,0,0]])
# total_len=10, response_len=4

action_mask[:, -4:]  # 取最后 4 列
# → tensor([[1,1,0,0],
#           [0,0,0,0]])

# 样本0: [., ., ., ., ., ., 1, 1, 0, 0]
#                                 ↑────↑ 取这4个
# 样本1: [., ., ., ., ., ., 1, 0, 0, 0]
#                                 ↑────↑ 取这4个
```

**负索引速查**：

```python
t = torch.tensor([0,1,2,3,4,5,6,7,8,9])  # (10,)

t[-1]    # tensor(9)           最后一个
t[-3:]   # tensor([7, 8, 9])   最后3个
t[:-3]   # tensor([0,1,2,3,4,5,6])   除了最后3个
t[-5:-2] # tensor([5, 6, 7])   倒数第5到倒数第2（不含）
```

**为什么语法中 `:` 前是空而不是 `0` 或 `:` 后没有指定？**

```python
# [:, -1024:] 等价于 [0:bs, total_len-1024:total_len]

# 第一个冒号：取所有行
[:, -1024:]    # dim=0: 冒号前后都省略 = 取全部
# 等价于:
[0:bs, -1024:] # 显式指定范围
[..., -1024:]  # 用 ... 表示 "前面所有维度"

# 第二个冒号：从 -1024 到末尾
[:, -1024:]   # dim=1: 冒号后省略 = 一直到末尾
# 等价于:
[:, -1024:total_len]  # 显式指定终点
[:, total_len-1024:]  # 用正索引
```

---

### Q14: 为什么 response 只取 `[-response_len:]`？

**高频指数**：⭐⭐⭐⭐

**答案：因为 `input_ids` 的 shape 是固定的 `(bs, total_len)`，所有 tensor 必须在预分配的固定尺寸内对齐。**

**根本原因**：

```text
input_ids 的预分配结构:
┌─────────────────────────┬──────────────────────────┐
│   prompt 区域 (0:4096)   │   response 区域 (4096:5120) │
│   [有效 prompt | PAD]   │   [全部是 PAD]             │
└─────────────────────────┴──────────────────────────┘
↑────── total_len = 5120 ──────────────────────────→

rollout 后 state_tensor 的实际内容:
┌──────────────────────────────────────────────────────────────┐
│ [P0..P99 | PAD...PAD | A0..A49 | O0..O29 | A0..A39 | PAD...]│
│ ↑── 4096 ──────────→│←────── response 区域 (1024) ───────→│
└──────────────────────────────────────────────────────────────┘
但 state_tensor 已经通过 pad_2d_list_to_length 强制对齐到 (bs, 5120)
```

**五个具体原因**：

```python
# 1. prompt 部分 driver 已经有了
#    原始 batch 的 input_ids 包含完整 prompt
#    不需要重复传输

# 2. response 部分才是 "这次 rollout 新产生的"
#    action1 + obs1 + action2 + ... 是模型和环境交互的结果
#    后续所有计算（reward、log_prob、loss）都只需要这部分

# 3. 后面计算 log_prob 时只需要 response 的 logits
logits = output.logits                                    # (1, 5120, vocab_size)
logits_response = logits[:, -response_len-1:-1, :]       # (1, 1024, vocab_size)
#                          ↑ 也只取最后 1024 个位置的 logits

# 4. reward_tensor 的 size 也必须是 response_len
reward_tensor = torch.zeros_like(responses)  # (bs, response_len)
# 如果不统一 → 不同样本长度不同 → 无法 stack 成 batch

# 5. attention_mask / action_mask / position_ids 的截取同理
response_mask = action_mask[:, -response_len:]  # (bs, 1024)
```

**设计本质**：

```text
固定长度的窗口 → 牺牲灵活性 → 换取批处理效率

(bs, 5120) 是一次 GPU forward 的固定输入尺寸
- 可以充分利用 GPU 并行
- 不需要动态 shape（PyTorch 对动态 shape 支持有限）
- Memory 预分配，避免反复 malloc

代价：
- 如果实际序列 > response_len → 左边被截断
- 如果实际序列 < response_len → 左边补 PAD
- response_len 需要事先估计好（根据任务统计分布）
```

**面试可能追问**：

> Q: 为什么取最后 1024 个，而不是最前面的 1024 个？
>
> A: 因为 prompt 在前面（0:4096），response 在后面（4096:5120）。取最后 1024 个就是取 response 部分。如果取最前面 1024 个，拿到的是 prompt 的前半部分——那完全没有意义，因为 prompt 的内容 driver 已经有了。

> Q: 截断发生在左边还是右边？有什么影响？
>
> A: `state_tensor[:, -response_len:]` 是在左边截断。对于太长的 response，**最左边的 token（通常是早期的 action/obs）被丢弃**，最右边的 token（通常是最后的 action，包含 `<answer>`）被保留。这是有意为之——最后包含答案的 action 比早期的中间步骤更重要。
