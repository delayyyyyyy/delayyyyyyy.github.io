# url01：Step 1 详解 —— 从 DataLoader 到 batch

> 本文是对 `2026-06-02_rl-forward-backward-shapes-beginner-zh.md` 中 **Step 1** 的逐行深入解释。
> 目标读者：理解 Python 基础、知道神经网络前向/反向传播概念，但对 PyTorch DataLoader 和本项目 DataProto 不熟悉的同学。

---

## 目录

1. [Step 1 整体做什么](#1-step-1-整体做什么)
2. [DataLoader 是什么](#2-dataloader-是什么)
3. [next(iter(...)) 做了什么](#3-nextiter-做了什么)
4. [batch_dict 里有什么](#4-batch_dict-里有什么)
5. [DataProto.from_single_dict 做了什么](#5-dataprotofrom_single_dict-做了什么)
6. [pop()：分离 gen_batch 和 batch](#6-pop分离-gen_batch-和-batch)
7. [Agent 模式的额外 pop](#7-agent-模式的额外-pop)
8. [分离后的全景图](#8-分离后的全景图)
9. [常见疑问](#9-常见疑问)

---

## 1. Step 1 整体做什么

**一句话概括**：从训练数据集中取出一批样本，打包成 `DataProto`，然后通过 `pop()` 把数据拆成两份——一份发给 vLLM 做生成（`gen_batch`），一份留在 driver 进程等生成结果回来后合并。

```text
┌──────────────────────────────────────────────────────────────────┐
│  Step 1: 数据准备                                                  │
│                                                                    │
│  train_dataloader  ──next(iter(...))──→  batch_dict               │
│                                              │                     │
│                                    DataProto.from_single_dict()    │
│                                              │                     │
│                                              ▼                     │
│                                     batch: DataProto               │
│                                        │                           │
│                                   pop() │ 分离                     │
│                                        │                           │
│                         ┌──────────────┴──────────────┐            │
│                         ▼                              ▼           │
│                    gen_batch                       batch           │
│                    (发给 vLLM)                     (留在 driver)    │
│                    - input_ids                    - data_source    │
│                    - attention_mask               - reward_model   │
│                    - position_ids                 - index          │
│                    - multi_modal_data             - ...            │
│                    - env_name (agent)                              │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. DataLoader 是什么

### 2.1 基本概念

`DataLoader` 是 PyTorch 提供的数据加载器，它封装了一个 `Dataset`，负责：

- **批量采样**：每次返回 `batch_size` 条数据
- **Shuffle**：每个 epoch 随机打乱数据顺序
- **并行加载**：多进程 (`num_workers`) 预取数据，避免 GPU 等待 CPU
- **Collate**：把单个样本拼成 batch

```python
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __getitem__(self, idx):
        return {"input_ids": ..., "attention_mask": ...}

    def __len__(self):
        return 10000

# 创建 DataLoader
dataloader = DataLoader(
    MyDataset(),
    batch_size=2,       # 每次取 2 条
    shuffle=True,        # 打乱
    num_workers=4,       # 4 个进程并行加载
    collate_fn=...,      # 自定义 batch 拼接函数
)
```

### 2.2 本项目中的 DataLoader

在本项目中（`ray_trainer.py`），DataLoader 在训练开始前被创建：

```python
# ray_trainer.py 中（简化）
self.train_dataloader = self._create_dataloader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn,  # 自定义 collate，将不等长序列 pad 到统一长度
)
```

**关键参数**（smoke config）：
| 参数 | 值 | 含义 |
|------|-----|------|
| `batch_size` | 2 | 每次取 2 条 prompt |
| `total_len (max_len)` | 5120 | 每条样本统一 pad/截断到 5120 |
| `prompt_len` | 4096 | prompt 最大长度 |
| `response_len` | 1024 | response 最大长度（5120 - 4096） |

### 2.3 为什么 DataLoader 是"iterable"

DataLoader 实现了 Python 的 `__iter__` 方法，所以可以用 `for batch in dataloader:` 来遍历。每个 epoch 从头到尾遍历一遍数据集。

但在 RL 训练中，通常**不按 epoch 遍历**，而是手动控制步数：

```python
# 不是 for batch in dataloader:  (这会自动走完一个 epoch)
# 而是手动取一个 batch：

batch_dict = next(iter(self.train_dataloader))
```

---

## 3. next(iter(...)) 做了什么

### 3.1 拆解这行代码

```python
batch_dict = next(iter(self.train_dataloader))
```

这一行分三步执行：

```python
# ① iter(dataloader) — 创建一个迭代器
dataloader_iterator = iter(self.train_dataloader)

# ② next(iterator) — 从迭代器中取出下一批数据
batch_dict = next(dataloader_iterator)

# ③ 等价于在 for 循环中取第一个 batch：
#    for batch_dict in self.train_dataloader:
#        break  # 只取第一个
```

### 3.2 迭代器的生命周期

```
         ┌──────────────────┐
         │   DataLoader     │
         │  (10000 条数据)   │
         └────────┬─────────┘
                  │ iter()
                  ▼
         ┌──────────────────┐
         │   迭代器 (iterator) │  ← 记住"读到哪了"
         │   current_pos = 0  │
         └────────┬─────────┘
                  │ next()  → batch 0: samples[0:2]
                  │ next()  → batch 1: samples[2:4]
                  │ next()  → batch 2: samples[4:6]
                  │ ...
                  │ next()  → batch 4999: samples[9998:10000]
                  │ next()  → StopIteration (数据取完了)
                  ▼
```

### 3.3 为什么每次 step 都调用 next(iter(...))

```python
# 训练循环 (ray_trainer.py，简化)
for step in range(total_steps):
    batch_dict = next(iter(self.train_dataloader))
    #                     ↑
    #                     每次都创建新的迭代器！
```

**注意**：这里每次 step 都重新调用 `iter()`，而不是重复使用同一个迭代器。为什么？

- **Shuffle 需求**：如果复用同一个迭代器，遍历顺序是固定的。每次都 `iter()` 可以确保 DataLoader 内部的 sampler 重新 shuffle。
- **无穷数据流**：RL 训练通常不按 epoch 来，而是跑固定步数。每次重新获取迭代器相当于"无限供应的随机数据流"。
- **实际上很多实现会用循环 DataLoader**：当数据取完时自动重新 shuffle 再开始。但最简单的方式就是每次都 `next(iter(...))`。

---

## 4. batch_dict 里有什么

### 4.1 整体结构

```python
batch_dict = next(iter(self.train_dataloader))
# batch_dict 是一个 Python dict，假设 batch_size=2：

batch_dict = {
    "input_ids":       torch.Tensor,   # shape: (2, 5120)
    "attention_mask":  torch.Tensor,   # shape: (2, 5120)
    "position_ids":    torch.Tensor,   # shape: (2, 3, 5120)  ← Qwen2-VL 特有的 MRoPE
    "raw_prompt_ids":  list,           # length: 2
    "multi_modal_data": list,          # length: 2
    "raw_prompt":      list,           # length: 2
    "data_source":     list,           # length: 2
    "reward_model":    list,           # length: 2
    "index":           list,           # length: 2
    # ... 更多字段
}
```

**关键区分**：
- `torch.Tensor` 类型的 → 会进入 `DataProto.batch`
- `list` / `ndarray` / 非 tensor 类型的 → 会进入 `DataProto.non_tensor_batch`

### 4.2 input_ids 详解

```python
batch_dict["input_ids"]  # shape: (2, 5120), dtype: int64 (torch.long)
```

这**不是**原始的 token 序列，而是经过 collate_fn 处理后的结果：

```
样本 0 原始 prompt: [101, 102, 103, ..., 200]  (100 个 token)
                    ↓ collate_fn: pad 到 5120
样本 0 input_ids:   [101, 102, 103, ..., 200, PAD, PAD, ..., PAD]
                    ↑── 100 个有效 token ──↑ ↑── 5020 个 PAD ──↑

样本 1 原始 prompt: [201, 202, 203, ..., 500]  (300 个 token)
                    ↓ collate_fn: pad 到 5120
样本 1 input_ids:   [201, 202, 203, ..., 500, PAD, PAD, ..., PAD]
                    ↑── 300 个有效 token ──↑ ↑── 4820 个 PAD ──↑
```

**重要**：此时 `input_ids` 中**只有 prompt 部分有效**，response 区域全是 PAD token（通常为 0 或 pad_token_id）。在 Step 2 中，模型生成的 token 会覆盖 response 区域。

```
input_ids (2, 5120):
┌──────────────────────────┬──────────────────────────┐
│    prompt (0:4096)        │   response (4096:5120)    │
│    [有效 prompt token]     │    [PAD PAD PAD ...]      │
│                           │    ← 训练时会被生成 token 覆盖│
└──────────────────────────┴──────────────────────────┘
```

### 4.3 attention_mask 详解

```python
batch_dict["attention_mask"]  # shape: (2, 5120), dtype: int64
```

每个位置标记这个 token 是否应该被 attention 看到：

- `1` = 有效 token，模型应该 attend to 它
- `0` = PAD token，模型应该忽略它（通过 attention mask 机制屏蔽）

```
样本 0 attention_mask: [1, 1, 1, ..., 1, 0, 0, ..., 0]
                        ↑── 100 个 1 ──↑ ↑── 5020 个 0 ──↑
                        ↑ 和 input_ids 的有效 token 数量一致
```

**在 Step 2 的 agent rollout 中，attention_mask 会被动态扩展**：
- Action token → mask=1（模型需要看到自己生成的 token）
- Observation token → mask=1（模型需要看到工具返回的结果）
- PAD token → mask=0

### 4.4 position_ids 详解（Qwen2-VL 特有）

```python
batch_dict["position_ids"]  # shape: (2, 3, 5120), dtype: int64
```

这是 Qwen2-VL 的 **MRoPE (Multi-Resolution Rotary Position Embedding)** 所需的 3D position IDs。

普通的 RoPE 只需要 1D position_ids（shape `(bs, seq_len)`），但 Qwen2-VL 处理图像时需要三维位置编码：

| 维度 | 名称 | 含义 |
|------|------|------|
| `dim=0` | batch | 第几个样本 |
| `dim=1` | 3 | 三个坐标轴（temporal, height, width） |
| `dim=2` | seq_len (5120) | 序列中每个 token 的位置索引 |

**为什么是 3 维？** Qwen2-VL 把图像切分成 patches，每个 patch 有 (时间, 高度, 宽度) 三个维度的位置信息。纯文本 token 在这三个维度上共享同一个递增的位置索引。

```python
# 简化示意：一个包含图像 + 文本的序列
# position_ids[0, :, :]  shape: (3, 5120)
#
# 文本部分：三个维度相同
# temporal:  [0, 1, 2, 3, ..., N, N, N]
# height:    [0, 1, 2, 3, ..., N, N, N]
# width:     [0, 1, 2, 3, ..., N, N, N]
#            ↑── 文本 token，位置递增 ──↑
#
# 图像部分：二维网格位置
# temporal:  [..., t, t, t, t, t, ...]
# height:    [..., 0, 0, 1, 1, 2, ...]
# width:     [..., 0, 1, 0, 1, 0, ...]
#            ↑── 图像 patch，height×width 网格 ──↑
```

### 4.5 non_tensor 字段

```python
# 这些是 list/ndarray/其他 Python 对象，不是 torch.Tensor
batch_dict["data_source"]     = ["perception", "perception"]  # 每个样本的数据集来源
batch_dict["reward_model"]    = [{"style": "rule"}, ...]      # reward 计算方式
batch_dict["raw_prompt"]      = ["请分析这张图片...", ...]      # 原始文本 prompt
batch_dict["multi_modal_data"] = [PIL.Image, PIL.Image]       # 图像数据
batch_dict["index"]           = [0, 1]                        # 样本索引
batch_dict["raw_prompt_ids"]  = [[101,102,...], [201,202,...]] # 不等长的原始 token ids
```

---

## 5. DataProto.from_single_dict 做了什么

### 5.1 源码逻辑

```python
batch: DataProto = DataProto.from_single_dict(batch_dict)
```

这是一个静态工厂方法，内部逻辑（简化）：

```python
@staticmethod
def from_single_dict(data: dict):
    batch = {}
    non_tensor_batch = {}
    meta_info = {}

    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value          # tensor → batch
        elif isinstance(value, np.ndarray):
            non_tensor_batch[key] = value  # ndarray → non_tensor_batch
        else:
            # Python list / dict / str / PIL.Image / ...
            non_tensor_batch[key] = value  # 其他类型 → non_tensor_batch

    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)
```

### 5.2 结果

```python
batch.batch = {
    "input_ids":       Tensor(2, 5120),
    "attention_mask":  Tensor(2, 5120),
    "position_ids":    Tensor(2, 3, 5120),
}

batch.non_tensor_batch = {
    "raw_prompt_ids":      [[101,...], [201,...]],
    "multi_modal_data":    [PIL.Image, PIL.Image],
    "raw_prompt":          ["请分析...", "请分析..."],
    "data_source":         ["perception", "perception"],
    "reward_model":        [{...}, {...}],
    "index":               [0, 1],
    # ...
}

batch.meta_info = {}  # 初始为空
```

---

## 6. pop()：分离 gen_batch 和 batch

### 6.1 为什么需要 pop()

RL 训练的架构决定了数据要**分两路走**：

| 去向 | 需要什么 | 为什么 |
|------|---------|--------|
| **gen_batch → vLLM (rollout worker)** | input_ids, attention_mask, position_ids, multi_modal_data, raw_prompt | vLLM 需要用这些做生成（image → text） |
| **batch → driver 进程** | data_source, reward_model, index | driver 需要这些来调度 reward 计算和日志记录 |

如果**不** pop，整个 `batch` 会通过网络（Ray RPC）发送给 vLLM worker。里面有些数据 worker 不需要（如 `reward_model`），传输浪费带宽。更重要的是，**vLLM worker 不应该修改 driver 的数据**——pop 是一种清晰的所有权转移。

### 6.2 pop() 的语义

```python
gen_batch = batch.pop(
    batch_keys=["input_ids", "attention_mask", "position_ids"],
    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "raw_prompt", ...],
)
```

`pop()` 和 Python dict 的 `pop()` 语义一致：

```python
# Python dict pop:
d = {"a": 1, "b": 2}
val = d.pop("a")   # val = 1, d = {"b": 2}
#  "a" 从 d 中被移除了
```

**DataProto.pop() 的行为**：

```python
# 伪代码
def pop(self, batch_keys, non_tensor_batch_keys):
    new_batch = {}
    new_non_tensor = {}

    for key in batch_keys:
        new_batch[key] = self.batch.pop(key)        # ← 从 self.batch 移除
    for key in non_tensor_batch_keys:
        new_non_tensor[key] = self.non_tensor_batch.pop(key)  # ← 从 self.non_tensor 移除

    return DataProto(batch=new_batch, non_tensor_batch=new_non_tensor)
```

### 6.3 操作前后的对比

**pop() 之前**：

```python
batch.batch = {
    "input_ids":       Tensor(2, 5120),    # ← 会被 pop
    "attention_mask":  Tensor(2, 5120),    # ← 会被 pop
    "position_ids":    Tensor(2, 3, 5120), # ← 会被 pop
}

batch.non_tensor_batch = {
    "raw_prompt_ids":      [...],           # ← 会被 pop
    "multi_modal_data":    [...],           # ← 会被 pop
    "raw_prompt":          [...],           # ← 会被 pop
    "data_source":         [...],           # → 留在 batch
    "reward_model":        [...],           # → 留在 batch
    "index":               [...],           # → 留在 batch
}
```

**pop() 之后**：

```python
# gen_batch (新的 DataProto，发给 vLLM)
gen_batch.batch = {
    "input_ids":       Tensor(2, 5120),
    "attention_mask":  Tensor(2, 5120),
    "position_ids":    Tensor(2, 3, 5120),
}
gen_batch.non_tensor_batch = {
    "raw_prompt_ids":      [...],
    "multi_modal_data":    [...],
    "raw_prompt":          [...],
}

# batch (原始 DataProto，留在 driver)
batch.batch = {}                            # ← 空了！所有 tensor 都被 pop 走了
batch.non_tensor_batch = {
    "data_source":         [...],            # ← 还在
    "reward_model":        [...],            # ← 还在
    "index":               [...],            # ← 还在
}
```

### 6.4 数据流后续

```text
gen_batch                        batch
    │                                │
    ▼                                │  (留在 driver)
┌──────────────────┐                │
│  vLLM Rollout     │                │
│  (agent 多轮生成)  │                │
│                    │                │
│  生成:              │                │
│  - responses       │                │
│  - action_mask     │                │
│  - env_reward      │                │
│                    │                │
└────────┬───────────┘                │
         │                            │
         ▼                            ▼
    gen_batch_output          ┌──────────────┐
         │                    │    union()    │
         └────────────────────┤  合并到一起   │
                              └──────────────┘
                                     │
                                     ▼
                              完整的 batch
                              (可以计算 reward + loss 了)
```

---

## 7. Agent 模式的额外 pop

### 7.1 触发条件

```python
if agent.activate_agent:
    gen_batch.non_tensor_batch["env_name"] = \
        batch.non_tensor_batch.pop("env_name")
    gen_batch.non_tensor_batch["origin_multi_modal_data"] = \
        batch.non_tensor_batch.pop("origin_multi_modal_data")
    gen_batch.non_tensor_batch["multi_modal_inputs"] = \
        batch.non_tensor_batch.pop("multi_modal_inputs")
    gen_batch.non_tensor_batch["extra_info"] = \
        batch.non_tensor_batch.pop("extra_info")
```

### 7.2 额外字段的含义

| 字段 | 类型 | 含义 | 用途 |
|------|------|------|------|
| `env_name` | `str` | 环境名称，如 `"deepeyesv2"` | 决定用哪个 sandbox 执行代码 |
| `origin_multi_modal_data` | `list[PIL.Image]` | 原始图像（未处理） | 传给 sandbox 用于代码执行 |
| `multi_modal_inputs` | `list[dict]` | vLLM 格式的多模态输入 | 传给 vLLM engine 做 image embedding |
| `extra_info` | `dict` | 额外元数据 | 传给 reward function 做评判 |

### 7.3 为什么要区分两个 multi_modal_data

```text
原始图像 (origin_multi_modal_data)
    │
    ├──→ sandbox (jupyter kernel): 需要 PIL.Image 来执行 `img.save()` 等操作
    │
    └──→ vLLM engine (multi_modal_inputs): 需要处理后的 tensor/路径来生成 image tokens
```

**两边的格式不同**：
- **Sandbox** 需要原始的 PIL.Image 对象，因为代码执行环境可能做 `Image.open()` / `img.resize()` 等操作
- **vLLM** 需要预处理后的输入（可能是 image path、pixel values tensor 等），和它的 vision encoder 对接

---

## 8. 分离后的全景图

```text
┌─────────────────────────────────────────────────────────────────────┐
│                        Step 1 完整数据流                              │
│                                                                       │
│  train_dataloader                                                     │
│       │                                                               │
│       │ next(iter(...))                                               │
│       ▼                                                               │
│  batch_dict (Python dict)                                             │
│       │                                                               │
│       │ DataProto.from_single_dict()                                  │
│       ▼                                                               │
│  batch: DataProto                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ batch (dict[str, Tensor]):                                   │    │
│  │   input_ids:       (2, 5120)  ← prompt token ids + padding  │    │
│  │   attention_mask:  (2, 5120)  ← 1=valid, 0=pad              │    │
│  │   position_ids:    (2, 3, 5120) ← Qwen2-VL MRoPE           │    │
│  │                                                               │    │
│  │ non_tensor_batch (dict[str, Any]):                            │    │
│  │   raw_prompt_ids:      [[...], [...]]  ← 不等长原始 token    │    │
│  │   multi_modal_data:    [Image, Image]  ← 图像                │    │
│  │   raw_prompt:          ["...", "..."]   ← 文本 prompt        │    │
│  │   data_source:         ["perception", ...]                   │    │
│  │   reward_model:        [{...}, {...}]                        │    │
│  │   index:               [0, 1]                                │    │
│  └─────────────────────────────────────────────────────────────┘    │
│       │                                                               │
│       │ pop(batch_keys=[...], non_tensor_batch_keys=[...])            │
│       │                                                               │
│       ├──────────────────────────────────────────┐                    │
│       ▼                                          ▼                    │
│  gen_batch: DataProto                      batch: DataProto          │
│  ┌──────────────────────────┐         ┌──────────────────────┐      │
│  │ batch:                   │         │ batch: {}  (空了)     │      │
│  │   input_ids (2, 5120)   │         │                       │      │
│  │   attention_mask (2,5120)│         │ non_tensor_batch:     │      │
│  │   position_ids (2,3,5120)│         │   data_source         │      │
│  │                          │         │   reward_model        │      │
│  │ non_tensor_batch:        │         │   index               │      │
│  │   raw_prompt_ids         │         └──────────────────────┘      │
│  │   multi_modal_data       │              │                          │
│  │   raw_prompt             │              │ (等待 rollout 结束)       │
│  │   [+ agent 字段]          │              │                          │
│  └──────────┬───────────────┘              │                          │
│             │                              │                          │
│             ▼                              │                          │
│       ┌──────────┐                         │                          │
│       │  vLLM    │                         │                          │
│       │  Rollout │                         │                          │
│       │  (Step 2)│                         │                          │
│       └────┬─────┘                         │                          │
│            │                                │                          │
│            ▼                                ▼                          │
│       gen_batch_output              batch.union(gen_batch_output)     │
│            │                                │                          │
│            └────────────┬───────────────────┘                          │
│                         ▼                                              │
│                  完整的 batch (进入 Step 3)                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 9. 常见疑问

### Q1: 为什么不直接在 DataLoader 里把 response 区域预分配好？

因为 response 是**模型实时生成**的，不是从数据集读的。数据集里只有 prompt。`total_len = prompt_len + response_len` 是为了给 response 预留空间，方便后续 `torch.cat` 操作直接在原地修改。

### Q2: 为什么 pop 而不是 copy？

两种考虑：
1. **性能**：tensor 很大（2×5120×4 bytes ≈ 40KB，加上 position_ids 更大），pop 避免了复制，直接把内存所有权转移
2. **语义**：driver 不应该再持有 "已被发送到 worker" 的数据引用，防止误用

### Q3: 如果 batch.batch 在 pop 后为空，driver 还有什么用？

driver 进程负责：
- 持有 `non_tensor_batch` 中的元数据（data_source, reward_model...）
- 等 rollout 完成后通过 `union()` 把生成结果合并回来
- 协调 reward 计算和参数更新

driver 是整个训练管道的 "调度中心"，不需要在 rollout 阶段持有 tensor 数据。

### Q4: position_ids 的 shape 为什么是 (2, 3, 5120) 而不是 (2, 5120)？

这是 Qwen2-VL 的特殊设计。Qwen2-VL 使用 **MRoPE (Multi-Resolution RoPE)**，需要三个独立的位置坐标（temporal, height, width）来编码图像 patches 的相对位置。普通文本模型只需要一维位置编码 `(bs, seq_len)`。

### Q5: batch 和 gen_batch 是在同一个进程里吗？

不是。`gen_batch` 通过 Ray RPC 发送给 vLLM worker 进程（可能在另一台 GPU 上），`batch` 留在 driver 进程。两者通过 Ray 的分布式对象存储通信。

```text
Driver 进程 (CPU)                    vLLM Worker 进程 (GPU 0)
┌─────────────────┐                 ┌─────────────────────┐
│  batch           │                 │  接收 gen_batch      │
│  (元数据)         │    Ray RPC      │  vLLM 生成 action    │
│                  │ ◄────────────── │  sandbox 执行代码     │
│  union() ←────── │   返回结果      │  构造 obs token      │
│                  │                 │  返回 gen_batch_output│
└─────────────────┘                 └─────────────────────┘
```

---

## 小结

Step 1 是整个训练管道的 **数据入口**。它做了三件事：

1. **取数据**：`next(iter(dataloader))` 获取一个 batch
2. **包装**：`DataProto.from_single_dict()` 把 dict 变成统一的数据容器
3. **分离**：`pop()` 把 rollout 需要的数据发给 vLLM，把元数据留在 driver

理解这一步的关键在于理解 **DataProto 的设计意图**——它是 driver 和 worker 之间传递数据的统一协议，`pop()` 和 `union()` 是数据所有权转移的核心操作。
