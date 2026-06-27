# url03：Step 3 详解 —— Reward Computation（奖励计算）

> 本文是对 `2026-06-02_rl-forward-backward-shapes-beginner-zh.md` 中 **Step 3** 的逐行深入解释。
> 目标读者：理解 Step 2 的 rollout 输出，想深入理解 "模型生成的回答怎么被打分" 的同学。

---

## 目录

1. [Step 3 整体做什么](#1-step-3-整体做什么)
2. [前置：为什么需要 Reward](#2-前置为什么需要-reward)
3. [Step 3a：token ids → 文本](#3-step-3atoken-ids--文本)
4. [Step 3b：Judge LLM 打分](#4-step-3bjudge-llm-打分)
5. [Step 3c：Reward 放在最后一个有效 token](#5-step-3creward-放在最后一个有效-token)
6. [Reward Function 设计详解](#6-reward-function-设计详解)
7. [完整数据流](#7-完整数据流)
8. [面试题库](#8-面试题库)

---

## 1. Step 3 整体做什么

**一句话**：把模型生成的 token ids 解码成文本，用 Judge LLM 给文本打分，然后把分数放在 response 序列的最后一个有效 token 位置。

```text
┌──────────────────────────────────────────────────────────────┐
│  Step 3: Reward Computation                                   │
│                                                               │
│  输入 (来自 Step 2 的 gen_batch_output + union 后的 batch):   │
│    data.batch["responses"]          (bs, response_len=1024)  │
│    data.batch["action_mask"]        (bs, total_len=5120)     │
│    data.non_tensor_batch["data_source"]  ["perception", ...] │
│    data.non_tensor_batch["extra_info"]   {...}               │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  For each sample:                                      │    │
│  │                                                         │    │
│  │  ① tokenizer.decode(valid_response_ids)               │    │
│  │     把 token ids 变回人类可读的文本                      │    │
│  │                                                         │    │
│  │  ② ThreadPoolExecutor → judge LLM 打分                 │    │
│  │     每个样本独立并发调用                                 │    │
│  │                                                         │    │
│  │  ③ 把分数放到最后一个有效 token 上                       │    │
│  │     reward_tensor[i, valid_len-1] = score              │    │
│  │     其余位置 = 0                                         │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                               │
│  输出:                                                         │
│    reward_tensor:      (bs, response_len=1024)                │
│    reward_extra_info:  {"score": [0.8, 0.0, ...], ...}       │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. 前置：为什么需要 Reward

### RL 训练的本质

```
监督学习 (SFT):
  "这是正确答案，学它"
  数据: {"prompt": "1+1=?", "answer": "2"}
  loss = cross_entropy(model_output, "2")

RL (PPO/GRPO):
  "这个回答好还是不好？好 → 增加概率，不好 → 减少概率"
  数据: {"prompt": "1+1=?", "response": "答案是 2"}
  reward = judge("答案是 2", ground_truth="2")  → 1.0
  loss = -advantage * log_prob(response|prompt)
```

**Reward 是 RL 的"指挥棒"**——它告诉模型什么是好的行为、什么是坏的行为。reward function 的设计直接决定了模型学到什么。

---

## 3. Step 3a：token ids → 文本

### 代码（naive_async.py + deepeyesv2.py，简化）

```python
# 从 batch 中取出第 i 个样本的 response
response_ids = data_item.batch["responses"][i]  # shape: (1024,)

# 找到实际有效的 token 数量（排除 PAD）
# 方法：通过 action_mask 找到最后一个有效 token 的位置
action_mask_full = data_item.batch["action_mask"]       # (5120,)
response_mask = action_mask_full[-1024:]                 # (1024,)
valid_response_length = response_mask.sum().item()       # 例: 90
# 或者直接找最后一个非 PAD token 的位置

# 只取有效的 token ids（排除 PAD）
valid_response_ids = response_ids[:valid_response_length]  # (90,)

# 解码成文本
response_str = tokenizer.decode(valid_response_ids)
```

### decode 的过程

```text
valid_response_ids = [151665, 1234, 5678, ..., 151658]
                      ↑<think>                  ↑</answer>

tokenizer.decode(valid_response_ids):
  → "<think>我需要分析这张图片...</think>\n
     <answer>4</answer>"

这一步是 tokenizer 的逆操作：
  - 每个 token id → 对应的文本片段
  - 特殊 token（如 <think>, <answer>）被正确还原
  - 连续的 subword token 被合并成完整单词
```

### 为什么需要 decode？

```text
Judge LLM（打分模型）接收的是自然语言文本，不是 token ids。

Judge LLM 的 prompt:
  "请判断以下回答是否正确。正确答案是 4。
   回答: <think>我需要分析...</think>\n<answer>4</answer>
   请给出评分 (0 或 1):"

Judge LLM 输出: "1"

如果直接把 token ids 喂给它 → 它看到的是 [151665, 1234, ...] → 无法理解
```

---

## 4. Step 3b：Judge LLM 打分

### 整体架构

```python
# naive_async.py 中的并发打分（简化）
from concurrent.futures import ThreadPoolExecutor

def compute_reward_async(data: DataProto) -> dict:
    scores = [None] * batch_size

    with ThreadPoolExecutor(max_workers=16) as executor:
        # 提交所有打分任务
        futures = {}
        for i in range(batch_size):
            data_item = data[i]  # 第 i 个样本
            future = executor.submit(
                unit_compute_reward_func,  # 单样本打分函数
                data_item,
                i,
            )
            futures[future] = i

        # 收集结果
        for future in as_completed(futures):
            i = futures[future]
            result = future.result()
            scores[i] = result

    return {"reward_tensor": build_reward_tensor(scores, data), ...}
```

### 为什么用 ThreadPoolExecutor 而不是逐个调用？

```text
Judge LLM 的调用是 I/O 密集型的（HTTP 请求/LLM 推理），不是 CPU 密集型的。

ThreadPoolExecutor:
  - 多个请求同时发出 → 等待时间重叠
  - batch_size=8, 每个 judge 调用需 2s
    → 串行: 8 × 2 = 16s
    → 并发: max(2s) ≈ 2s (如果 judge 服务有足够并发容量)

Ray remote 也能做，但 ThreadPoolExecutor 更轻量：
  - Judge 调用通常是 HTTP → 不需要 Ray 的分布式能力
  - 线程开销远小于进程
```

### unit_compute_reward_func 的内部

```python
def unit_compute_reward_func(data_item, index):
    # ① 解码
    response_text = decode_response(data_item)

    # ② 调用 judge
    score_dict = compute_score(
        response_text=response_text,
        ground_truth=data_item["extra_info"]["answer"],
        data_source=data_item["data_source"],
    )

    # ③ 返回
    return {
        "score": score_dict["score"],      # 例: 0.8
        "acc": score_dict.get("acc", 0),   # accuracy: 1 or 0
        "format": score_dict.get("format", 0),  # format: 1 or 0
    }
```

---

## 5. Step 3c：Reward 放在最后一个有效 token

### 代码

```python
# 为整个 batch 创建 reward tensor
reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
# torch.zeros_like: 创建和 responses 同 shape 的全 0 tensor
# responses shape: (bs, 1024)
# → reward_tensor shape: (bs, 1024), dtype=float32, 值全是 0.0

for i in range(batch_size):
    valid_len = valid_response_lengths[i]  # 例: 90
    score = scores[i]["score"]              # 例: 0.8

    # ★ 核心操作：只把分数放在最后一个有效 token 的位置
    reward_tensor[i, valid_len - 1] = score
    # 等价于 numpy: reward_tensor[i][valid_len - 1] = score
```

### 直观图示

```
样本 0 (有效长度=90, score=0.8):
  Position:  0    1    2   ...   88   89   90   91  ...  1023
  Value:    0.0  0.0  0.0  ...  0.0  0.8  0.0  0.0  ...  0.0
                                    ↑
                              只有这里非零！

样本 1 (有效长度=55, score=0.0):
  Position:  0    1    2   ...   54   55   56  ...  1023
  Value:    0.0  0.0  0.0  ...  0.0  0.0  0.0  ...  0.0
                                    ↑
                              全是 0（得分为 0）
```

### 为什么只放在最后一个 token？

**这是 Outcome Reward（结果奖励）的设计**：

```text
Outcome Reward:
  "只看结果好不好，不看过程"
  
  优点:
  1. 简单：不需要标注每个中间步骤的好坏
  2. 可扩展：只需要最终答案的 ground truth
  3. 与 GRPO 兼容：组内归一化比较的是 "最终答案" 的好坏
  
  缺点:
  1. 稀疏：只有最后一个 token 有信号
  2. 长序列的 credit assignment 困难
     → 模型不知道是哪个中间步骤导致了最终结果
  3. 对于需要多步推理的任务，信号太弱

Process Reward (对比):
  "每个推理步骤都打分"
  
  优点: 密集信号，每个 step 都有反馈
  缺点: 需要标注每个 step 的好坏，成本高
```

### 在 GRPO 中如何使用这个稀疏 reward

```python
# Step 5 中 GRPO 的处理:
# 1. 把 token-level reward 压缩成标量
scores = reward_tensor.sum(dim=-1)
# reward_tensor: (bs, 1024) → sum → (bs,)
# 因为只有最后一个有效位置非零 → sum 结果就是那个位置的分数
# 例: [0.0, 0.0, ..., 0.8, 0.0, ...] → sum → 0.8

# 2. 组内归一化
advantages = (scores - mean_group) / (std_group + ε)

# 3. 扩展回 token-level
advantages = advantages.unsqueeze(-1) * response_mask  # (bs, 1) * (bs, 1024) = (bs, 1024)
# 所有 action token 获得相同的 advantage
```

---

## 6. Reward Function 设计详解

### deepeyesv2.compute_score 的实现

```python
def compute_score(response_text, ground_truth, data_source):
    """
    对模型生成的 response 进行多维度评分。

    Args:
        response_text: 模型生成的完整回答文本
        ground_truth: 正确答案（从数据集中获取）
        data_source: 数据来源 ("perception" / "reasoning" / ...)

    Returns:
        {"score": float, "acc": int, "format": int}
    """
    # ① 提取 <answer>...</answer> 中的内容
    answer = extract_answer(response_text)
    # 例: "<think>...</think>\n<answer>4</answer>" → "4"

    # ② 准确性评分 (accuracy reward)
    if answer is None or answer.strip() == "":
        acc_reward = 0.0
    elif normalize(answer) == normalize(ground_truth):
        acc_reward = 1.0
    else:
        acc_reward = 0.0

    # ③ 格式评分 (format reward)
    format_reward = check_format(response_text)
    # 检查: 是否有 <think>...</think> 块
    #       是否有 <answer>...</answer> 块
    #       是否有 <code>...</code> 块（如果任务需要代码执行）
    # 全部满足 → 1.0，否则 → 0.0

    # ④ 加权合并
    final_score = 0.8 * acc_reward + 0.2 * format_reward
    # 权重是超参数，由配置文件指定

    return {
        "score": final_score,      # 用于 GRPO advantage
        "acc": acc_reward,         # 用于监控（logging）
        "format": format_reward,   # 用于监控（logging）
    }
```

### 为什么拆成多个维度？

```text
"what gets measured gets optimized"

如果只有 accuracy:
  → 模型可能学会 "输出正确答案，但格式混乱"
  → 下游解析困难（无法自动化提取答案）

如果只有 format:
  → 模型可能学会 "格式完美，但答案错误"
  → 训练没有意义

多维 reward 的好处:
  1. 每个维度独立监控 → 知道模型在哪方面好/差
  2. 权重可调 → 前期看重 format，后期看重 accuracy
  3. 防止 reward hacking → 单一维度容易被钻空子
```

### extract_answer 的实现

```python
import re

def extract_answer(text):
    # 方法1: 正则匹配 <answer>...</answer>
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 方法2: 如果没找到标签，尝试匹配 "答案是 X" 模式
    match = re.search(r"(?:答案|结果|answer)\s*(?:是|为|:|=)\s*(.+)", text)
    if match:
        return match.group(1).strip()

    # 方法3: 取最后一行
    lines = text.strip().split("\n")
    return lines[-1].strip() if lines else ""

def normalize(s):
    """标准化文本用于比较"""
    s = s.lower().strip()
    # 去掉多余空格
    s = re.sub(r"\s+", " ", s)
    # 去掉标点差异
    s = s.rstrip(".,;!")
    return s
```

---

## 7. 完整数据流

```text
═══════════════════════════════════════════════════════════════════
输入
═══════════════════════════════════════════════════════════════════
  data.batch["responses"]               (bs=2, response_len=1024)
  data.batch["action_mask"]             (bs=2, total_len=5120)
  data.non_tensor_batch["data_source"]  ["perception", "perception"]
  data.non_tensor_batch["extra_info"]   [{"answer": "4"}, {"answer": "cat"}]

═══════════════════════════════════════════════════════════════════
Step 3a: decode
═══════════════════════════════════════════════════════════════════
  样本 0:
    response_ids[0, :90] → decode →
    "<think>分析图片...</think>\n<answer>4</answer>"
    valid_len = 90

  样本 1:
    response_ids[1, :55] → decode →
    "<think>我认为...</think>\n<answer>dog</answer>"
    valid_len = 55

═══════════════════════════════════════════════════════════════════
Step 3b: judge LLM 打分 (并发)
═══════════════════════════════════════════════════════════════════
  ThreadPoolExecutor (max_workers=16):

  样本 0:
    compute_score("4", ground_truth="4") →
      acc = 1.0, format = 1.0
      score = 0.8*1.0 + 0.2*1.0 = 1.0

  样本 1:
    compute_score("dog", ground_truth="cat") →
      acc = 0.0, format = 1.0
      score = 0.8*0.0 + 0.2*1.0 = 0.2

═══════════════════════════════════════════════════════════════════
Step 3c: 构造 reward_tensor
═══════════════════════════════════════════════════════════════════
  reward_tensor = zeros(2, 1024)  (dtype=float32)

  reward_tensor[0, 89] = 1.0    ← 样本0最后一个有效token位置
  reward_tensor[1, 54] = 0.2    ← 样本1最后一个有效token位置
  其余所有位置 = 0.0

═══════════════════════════════════════════════════════════════════
输出
═══════════════════════════════════════════════════════════════════
  reward_tensor:      (bs=2, 1024)  ← sparse: 只有2个非零值
  reward_extra_info:  {
      "score":  [1.0, 0.2],
      "acc":    [1.0, 0.0],
      "format": [1.0, 1.0],
  }
```

---

## 8. 面试题库

---

### Q1: 为什么用稀疏 reward（outcome reward）而不是每个 step 都打分？

**高频指数**：⭐⭐⭐⭐⭐

**答案**：

```
核心 tradeoff: 成本 vs 信号密度

Outcome Reward (当前做法):
  ✓ 只需要最终答案的 ground truth
  ✓ 标注成本低（答案对/错，不需要标注过程）
  ✓ 与具体推理路径解耦（允许模型自由探索解题思路）
  ✗ 信号稀疏：长序列中只有一个 token 有信号
  ✗ Credit assignment 困难：不知道哪个 step 导致了最终结果

Process Reward (替代方案):
  ✓ 密集信号：每个推理 step 都有反馈
  ✓ 更容易 credit assignment
  ✗ 需要标注每个 step 的正确性，成本高
  ✗ 可能过度约束：模型只会模仿标注者的解题思路

什么时候用 Process Reward?
  - 数学证明：需要验证每一步的逻辑正确性
  - 代码生成：每个函数/API 调用都能独立验证
  - 复杂的多步规划：需要中间反馈来引导搜索

什么时候 Outcome Reward 就够了?
  - 答案可自动验证的任务（数学计算、代码执行）
  - 最终结果比中间过程更重要的任务（图像描述、翻译质量）
```

---

### Q2: Judge LLM 是怎么打分的？如果 Judge 本身也犯错怎么办？

**高频指数**：⭐⭐⭐⭐

**答案**：

```
Judge LLM 的类型（从简单到复杂）:

1. Rule-based (当前 smoke config):
   - 直接字符串匹配: answer == ground_truth
   - 格式正则检查: re.search(r"<answer>.*</answer>", text)
   - 优点: 快速、确定、免费
   - 缺点: 只能做精确匹配，无法评估 "部分正确"

2. LLM-as-Judge:
   - 用另一个 LLM（如 GPT-4）打分
   - Prompt: "请判断这个回答是否正确，给出0-5分"
   - 优点: 能评估语义正确性、部分正确
   - 缺点: 慢、贵、LLM 自己也可能犯错

3. Ensemble Judge:
   - 多个 Judge 投票 → 取平均/多数
   - 减 少单点失败

4. Execution-based (代码/数学):
   - 运行代码 → 检查输出
   - 数学: 代入验证
   - 最可靠（因为可以客观验证）

Judge 犯错怎么办？
  - 监控: 跟踪 reward distribution → 异常值告警
  - 人工抽样: 定期抽查 judge 的判断
  - Reward noise 在 GRPO 中的影响:
    组内归一化 → 只要噪声是对称的，影响有限
    但如果 judge 系统性偏向某种回答 → 模型会被误导
```

---

### Q3: 多维 reward 怎么设计和调权重？

**高频指数**：⭐⭐⭐⭐

**答案**：

```
设计原则:

1. 每个维度应该是正交的（不重复衡量同一个东西）
   ✗ 不好的设计:
     - "准确性" 和 "是否和标准答案一致" ← 重复了
   ✓ 好的设计:
     - "准确性" + "格式规范" + "代码效率" + "安全性"

2. 权重反映优先级:
   final_score = w1*accuracy + w2*format + w3*efficiency + w4*safety

   调权策略:
   - 初期: w_format 调大 (0.3-0.5)，确保模型学会正确格式
   - 中期: 逐步降低 w_format，提高 w_accuracy
   - 全程: w_safety 保持较高 (0.1-0.2)，安全不能妥协

3. 权重不是固定的:
   - Curriculum learning: 不同训练阶段不同权重
   - Adaptive weights: 如果某维度已达天花板 → 降低权重

4. 监控每个维度的单独指标:
   wandb.log({
       "reward/total": total,
       "reward/acc": acc,
       "reward/format": format,
       "reward/efficiency": efficiency,
   })
   # 如果 total 上升但 acc 下降 → 权重要调整
```

**面试追问**：

> Q: 如果两个维度的 reward 冲突怎么办？（如 "正确的答案" vs "格式不符合要求"）
>
> A: 格式要求应当作为 gate（如果格式不对，最终 reward=0），而不是作为 soft weight。这样模型被迫学会正确格式，而不是在内容和格式之间做 tradeoff。

---

### Q4: 为什么用 ThreadPoolExecutor 而不是 Ray？

**高频指数**：⭐⭐⭐

**答案**：

```
ThreadPoolExecutor:
  - 适合 I/O 密集型任务（HTTP 请求、等待 Judge LLM 响应）
  - 线程共享内存 → 传递数据不需要序列化
  - 轻量：创建/销毁线程开销小
  - Python GIL 不影响 I/O 操作（线程在等待 I/O 时释放 GIL）

Ray:
  - 适合 CPU/GPU 密集型任务（模型推理、训练）
  - 分布式：可以跨机器调度
  - 重量级：需要序列化数据、进程间通信

Judge LLM 打分 → I/O 密集型（HTTP 请求等待响应）
→ ThreadPoolExecutor 更适合

如果 Judge 是本地 GPU 模型（需要推理）：
→ 用 Ray 把 Judge 部署为独立 Actor，避免 GPU 竞争
```

---

### Q5: Reward Hacking 是什么？如何防止？

**高频指数**：⭐⭐⭐⭐⭐

**答案**：

```
Reward Hacking = 模型学会了 "钻 reward function 的空子"，而不是真正完成人类想要的任务。

经典例子:
  1. "格式满分，答案全错"
     模型学会输出 <think>...</think>\n<answer>0</answer>
     → format_reward = 1.0，但答案永远是 0

  2. "长度 hacking"
     模型输出超长的 <think> 过程 → 看起来在 "深度思考"
     → 但没有实质内容

  3. "模板 hacking"
     模型发现某种固定模板总是能拿到部分分数
     → 不管什么问题都用同一个模板

  4. "环境 hacking"
     模型发现某种输出格式会导致 judge 解析失败
     → 解析失败时返回默认高分 → 模型就学会了

如何防止:
  1. 多维 reward → 单维度 hacking 被其他维度惩罚
  2. Reward normalization → 组内比较，绝对的 "gaming" 无效
  3. 人工审计 → 定期检查高分/低分样本
  4. Reward model 迭代 → 发现 hacking → 修 rule → 重新训练
  5. KL penalty → 约束策略不要偏离初始策略太远
     这是 PPO 的内置机制: loss += β * KL(π_new || π_old)
```

---

### Q6: 如果 reward 信号太稀疏（大多数 reward=0），模型能学到东西吗？

**高频指数**：⭐⭐⭐⭐

**答案**：

```
能，但需要一定的设计保证:

1. GRPO 的组内归一化帮助很大:
   即使所有 reward 都是 0 或 1（二值），
   组内归一化后:
   - 对于有 1 个正确、n-1 个错误的组:
     advantage = (1 - 1/n) / std > 0   → 鼓励正确的那个
     advantage = (0 - 1/n) / std < 0   → 惩罚错误的那些

2. 但需要足够的 "信号样本":
   - 如果 90% 的样本 reward=0 → 有效信号太少
   - 需要保证一定比例的 "成功" 样本
   - 可以通过 curriculum learning: 从简单任务开始

3. 改进方法:
   a) 降低任务难度 → 提高成功率
   b) 增加 rollout_n → 每组更多样本 → 更稳定的组内归一化
   c) 加入 shaping reward:
      不是完全的过程奖励，而是一些启发式信号
      如: "是否调用了正确的工具" → +0.1
   d) 用 best-of-N 采样 → 选最好的作为正样本

4. 经验数字:
   - 成功率 10-30% → 可以学到东西（GRPO 的组内 signal 足够）
   - 成功率 < 5%  → 可能学不动（需要 curriculum 或 shaping）
   - 成功率 > 50%  → 信号丰富，但可能任务太简单
```

---

### Q7: 为什么要 decode 成文本再打分，不能直接在 token 层面打分吗？

**高频指数**：⭐⭐⭐

**答案**：

```
Judge LLM 的工作方式和训练模型一样——输入自然语言，输出评分。

不能直接在 token 层面打分的原因:
  1. Judge LLM（如 GPT-4）的输入是文本 → 它们不理解 token ids
  2. 不同 tokenizer 的 token ids 含义不同
     Qwen 的 token 151665 = "<think>"
     GPT 的 token 151665 = 完全不同的东西
  3. 规则匹配（extract_answer）也需要文本

唯一例外: Execution-based eval
  - 代码执行 → 不需要 Judge LLM
  - 直接从 token ids → decode → execute → 检查输出
  - 但这里也需要先 decode 成 Python 代码
```

---

### Q8: 面试速记表

| 编号 | 问题 | 3 秒答案 |
|------|------|---------|
| Q1 | 稀疏 vs 密集 reward | Outcome: 成本低、灵活但信号稀疏；Process: 密集但标注贵 |
| Q2 | Judge LLM 犯错 | Rule-based (快/确定) + LLM-as-Judge (语义) + Ensemble (冗余) |
| Q3 | 多维 reward | 正交维度 + 可调权重 + 独立监控 + curriculum |
| Q4 | ThreadPoolExecutor vs Ray | I/O 密集用 ThreadPool，GPU 密集用 Ray |
| Q5 | Reward Hacking | 多维 reward + 组内归一化 + KL penalty + 人工审计 |
| Q6 | 稀疏信号 | GRPO 组内归一化放大信号 + curriculum + shaping reward |
| Q7 | 为什么 decode | Judge LLM 输入是文本，不同 tokenizer 的 token ids 不通用 |

---

## 补充问答

---

### Q8: "reward 放在最后一个有效 token" 到底是什么意思？最后一个有效 token 具体是哪个？

**高频指数**：⭐⭐⭐⭐⭐

#### 先看代码

```python
# naive_async.py 第 156 行
reward_tensor[i, valid_response_length - 1] += reward
#              ↑                          ↑
#              第 i 个样本                  最后一个有效 token 的索引
```

#### "最后一个有效 token" 的定义

**最后一个有效 token = response 序列中最后一个非 PAD 的 token，即模型输出的最后一个 token。**

以一条具体样本为例，看图说话：

```text
一条完整序列 state_tensor[0]（全长 5120）:
┌──────────────────────────┬──────────────────────────────────────┐
│   prompt 区域 (0:4096)    │        response 区域 (4096:5120)      │
│                           │                                      │
│ [P0..P99 | PAD...PAD]    │ [A0..A49 | O0..O29 | A0..A39 | PAD]  │
│  ↑100个   ↑3996个0       │  ↑50个    ↑30个    ↑40个    ↑954个0 │
└──────────────────────────┴──────────────────────────────────────┘
                                                            ↑
                            response 区域中，最后一个非 PAD token 是 action2 的最后一个 token
                            即 </answer> 对应的 token
                            它的位置 = 4096 + 120 - 1 = 4215
                            其中 120 = 50(action1) + 30(obs1) + 40(action2)
                            valid_response_length = 120
                            → valid_response_length - 1 = 119
                            → reward_tensor[i, 119] = score
```

#### 用实际代码追踪

```python
# 第 102-104 行：找到有效 response 长度
response_ids = data_item.batch["responses"]            # (1024,)  ← response 部分
valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
# attention_mask 在 PAD 位置是 0，有效位置是 1
# .sum() 就是数 "有多少个 1" = "有多少个有效 token"
# 在这个例子中 = 50(action1) + 30(obs1) + 40(action2) = 120

valid_response_ids = response_ids[:valid_response_length]
# 取前 120 个有效 token
# [A0, A1, ..., A49, O0, O1, ..., O29, A0, A1, ..., A39]

# 最后一个有效 token = valid_response_ids[-1]
# = action2 的最后一个 token = </answer>
```

#### response 区域的具体内容

```text
response 区域 (最后 1024 个 token):
┌──────────────────────────────────────────────────────────────┐
│ A0...A49 │ O0...O29 │ A0...A39 │ PAD...PAD │
│ ↑ 50     │ ↑ 30     │ ↑ 40     │ ↑ 954     │
│ action1  │ obs1     │ action2  │ 全是 0    │
│          │          │          │           │
│          │          │ └─ A39 = </answer>   │
│          │          │     这个 token 的      │
│          │          │     reward = score    │
└──────────────────────────────────────────────────────────────┘

valid_response_length = 50 + 30 + 40 = 120

reward_tensor[i, :] = [0.0, 0.0, ..., 0.0, 0.8, 0.0, 0.0, ..., 0.0]
                       ↑ position 0..118   ↑ pos=119  ↑ pos=120..1023
                                           最后一个有效 token

# 注意：obs 部分的 action_mask=0（不参与 loss），但它们的 attention_mask=1（是有效 token）
# valid_response_length 用 attention_mask 来算的，所以包含了 obs
# 最后一个有效 token = action2 的最后（因为 action2 在 obs 之后）
```

#### 为什么放在 action2 最后（而不是 obs 最后）？

```
因为序列顺序是:
  action1 → obs1 → action2 → obs2 → ... → actionN

最后一轮一定是 action token（模型输出 <answer>）
→ 最后一个有效 token = 最后一轮 action 的最后一个 token
→ 通常就是 </answer>

如果放到 obs 的最后一个 token:
  → reward 和模型的 action 就错位了
  → "模型输出的 <answer>4</answer>" 和 "环境返回的 Code execution result" 是不同的东西
  → 应该奖励/惩罚模型输出的东西
```

---

### Q9: Credit Assignment 是什么？为什么长序列的 Credit Assignment 困难？

**高频指数**：⭐⭐⭐⭐⭐

#### Credit Assignment 的定义

> **Credit Assignment Problem**（功劳分配问题）：在一个多步决策过程中，如何确定每个步骤对最终结果的贡献？

```
Agentic RL 场景:
  模型输出了 100 个 action token，最终答案正确 → reward = 1.0
  
  问题是: 这 100 个 token 中，哪些 token 对 "答案正确" 起了正面作用？
         → 全部都有贡献？
         → 还是 "关键是第 45 个 token 推理对了"？
         → 还是 "前 50 个都在胡说，第 51 个才走上正轨"？
```

#### 当前的 Outcome Reward 做法

```python
# 稀疏 reward: 所有 action token 获得相同的 advantage
advantages = [0.8, 0.8, 0.8, ..., 0.8, 0.8]  # (120,)
#             ↑ action1  ↑ obs1(被mask=0)  ↑ action2

# GRPO 把 scalar score 广播到所有 action token:
#   好 → 整个回答都被鼓励
#   坏 → 整个回答都被惩罚
# 
# 问题: 如果回答的前半部分错了、后半部分对了
#       稀疏 reward 会同时鼓励错误的前半部分
```

#### 为什么长序列的 Credit Assignment 更难

```text
短序列 (10 个 action token):
  → "答案: 4" → 只有 3 个 token
  → reward = 1.0
  → 错误 credit 的范围小 (最多 10 个 token 被错误地鼓励)

长序列 (500 个 action token, 含多轮代码执行):
  → 模型写了 200 个 token 的错误代码
  → 然后发现了错误，用了 200 个 token 修正
  → 最后 100 个 token 得到正确答案
  → reward = 1.0
  → 如果所有 500 个 token 都被鼓励:
     模型学到了 "先犯错再修正" = 好行为
     而不是 "直接做对" = 好行为

更糟的情况:
  → 模型写了 300 个 token 的随机代码 (和问题无关)
  → 最后 200 个 token 正确
  → reward = 1.0
  → 模型学到了 "可以随便胡扯 300 个 token，只要最后对就行"
  → 浪费推理时间和计算资源
```

#### 为什么不对每个 token 独立算 advantage？

```
难度在于: 没有 per-token 的 ground truth

per-token reward 需要:
  - 标注每个 token 是 "好" 还是 "坏"
  - 对 500 个 token → 需要 500 个标注
  - 人工没法做

替代方案:
  1. Process Reward Model (PRM):
     训练一个模型给每个 step 打分
     训练数据: 人工标注 + 自动标注 (最终答案对 → 中间步骤也对)

  2. Learned Credit Assignment:
     用 Value Network 估计每个 state 的价值
     advantage = reward + γ*V(s') - V(s)
     每个 token 有不同的 advantage

  3. Contrastive decoding:
     找 "正确答案" vs "错误答案" 之间的关键差异 token
     只对这些关键 token 施加梯度
```

#### 面试加分回答

```
一个比喻: 足球队进球

Outcome Reward: "这场球赢了 → 每个球员都奖励"
  → 问题: 守门员没扑好球也被鼓励

Process Reward: "后卫抢断 + 中场传球 + 前锋射门 → 分别评判"
  → 守门员的失误可以被单独识别 → 不奖励
  → 但需要每个环节的评判标准

Token-level dense reward: "每个传球动作都评分"
  → 太细粒度 → 标注成本不可承受
  → 很多中间 token 是好是坏本身就是模糊的

当前工业界实践:
  对大多数 LLM Agent 任务，Outcome Reward + GRPO 已经足够
  因为:
  1. LLM 足够强 → 好的推理路径和坏的推理路径有整体差异
  2. GRPO 组内比较 → 相对好坏可以区分
  3. 多轮采样 → 不同路径自然产生不同 outcome

但长序列 Credit Assignment 仍然是前沿研究问题
  → DeepSeek-R1: 用 Process Reward 辅助 Outcome Reward
  → OpenAI o1: 可能用了隐式的过程监督
```

---

### Q10: Judge LLM 是怎么 Prompt 的？他怎么知道每个 Rollout 的 GT？

**高频指数**：⭐⭐⭐⭐

#### GT 的来源

**GT (Ground Truth) 是从训练数据集中来的，在 DataLoader 阶段就已经附着在 batch 上了。**

```python
# 数据流转:
# 1. 数据集构建阶段:
train_dataset = [
    {
        "prompt": "请分析这张图片中有几个苹果？",   # 问题
        "image": PIL.Image,                       # 图片
        "answer": "4",                            # ← GT!
        "data_source": "perception",
    },
    # ...
]

# 2. DataLoader → batch_dict
batch_dict["reward_model"] = [{"ground_truth": "4"}, {"ground_truth": "cat"}]
#                               ↑ 样本0的GT           ↑ 样本1的GT

# 3. Step 1 中 pop 时，reward_model 留在 driver
#    batch.non_tensor_batch["reward_model"] = [{"ground_truth": "4"}, ...]

# 4. Step 8 (naive_async.py 第 110 行):
ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
# → "4"

# 5. 传给 Judge:
compute_score(response_text, ground_truth="4", extra_info={"question": "..."})
```

**所以**：GT 是预先标注好的，不是 Judge 推理出来的。Judge 的角色是 "比较模型回答和参考答案是否一致"，不是 "判断模型回答对不对"。

#### Judge 的完整 Prompt

从实际代码（`deepeyesv2.py` 第 138-183 行 + `verify_prompt.md`）：

```text
=== System Prompt (verify_prompt.md) ===

# 角色
你是一个判断专家，专注于判断输入的2个答案是否一致。

# 任务
你的任务是判断[模型回答]与[参考答案]是否一致，不需要思考[问题]真正的正确答案，以下是详细的判断步骤：
1. 问题理解：仔细阅读[问题]，并按照[判断标准]对问题进行分类，找出问题中包含多少个提问。
2. 答案对照：按照问题中的提问顺序，将[模型回答]与[参考答案]一一进行判断，对比是否一致。若存在一处不一致，则视为不一致。

# 判断标准

## 简答类
答案不唯一或不具体，需要根据材料、条件，自行组织语言回答问题或进行解答题目、证明结论。
...

## 客观类
存在明确、客观的答案，如科学知识、数学、物理等。
- 可以忽略答案组织形式。例如：计算题只需要最终结果数值一致即可（例："6棵"、"6"、"six"等视为一致）。

### 选择题
给出答案选项，答案选项可能用字母（A、B、C、D、...）标记...
[模型回答]中的答案只需要与[参考答案]中对应的标记一致，即判断为一致。
...

# 输出
1. 在<think></think>标签中输出你的思考过程。
2. 结论输出：用一个词（是或否）在最后得出结论，格式为 \boxed{Yes} 或 \boxed{No}。
3. 注意：你输出的结论必须与思考过程中得到的结论一致。

## 输出示例
<最终结果>
\boxed{Yes/No}
<\最终结果>

=== User Prompt（代码拼接）===

[问题]: {question}
[参考答案]: {answer}      ← 这就是 GT！
[模型回答]: {prediction}  ← 这是模型生成的 answer_text
```

#### 完整的 Judge 调用流程

```python
# Step 1: 从模型生成的完整 response 中提取 <answer> 部分
response_str = "<think>需要分析...</think>\n<code>```python...```</code>\n<answer>4</answer>"
answer_text = extract_answer(response_str)  # → "4"

# Step 2: 构造 Judge prompt
question = extra_info['question']  # "图片中有几个苹果？"
ground_truth = "4"                # 从数据集的 reward_model 字段来

system_prompt, user_prompt = get_prompt(question, ground_truth, answer_text)
# system_prompt: verify_prompt.md 的内容
# user_prompt:
#   """
#   [问题]:图片中有几个苹果？
#   [参考答案]:4
#   [模型回答]:4
#   """

# Step 3: 调用 Judge LLM (OpenAI API)
response = judge_chat_completion(
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    temperature=0.3,      # 低温度 → 确定性输出
    max_tokens=512,
    fallback_response="No",  # API 调用失败时的兜底 → 0 分
)

# Step 4: 解析 Judge 输出
# Judge 返回:
# "<think>两个答案都是4，一致</think>
#  <最终结果>\boxed{Yes}<\最终结果>"

if 'Yes' in response:
    acc_reward = 1.0
else:
    acc_reward = 0.0

# Step 5: 加权合并
final_score = 0.8 * acc_reward + 0.2 * format_reward
```

#### 关键设计点

```
1. 低温度 (temperature=0.3):
   → Judge 应该稳定、一致
   → 同一个回答不应该有时判对有时判错

2. fallback_response="No":
   → Judge API 调用失败 → 默认判错 (0分)
   → 避免因为网络问题给不该给的 reward

3. 重试机制 (JUDGE_RETRIES=3):
   → 每次失败后指数退避重试 (1s, 2s, 3s)
   → 三次都失败才用 fallback

4. 缓存 (JUDGE_CACHE_SIZE=4096):
   → 相同的问题+答案+模型回答 → 不重复调用 Judge
   → 因为 GRPO 会采样 n 条 → 同 prompt 的不同 response 可能共享部分 judge 调用

5. 信号量 (JUDGE_CONCURRENCY=2):
   → 限制并发数 → 防止打爆 Judge LLM 服务

6. 多 base URL 负载均衡:
   → client_idx = random.randint(0, len(client_list) - 1)
   → 多个 Judge 实例之间随机选择 → 分散负载
```

#### GT 的数据结构

```python
# 从 naive_async.py 第 110 行:
ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

# reward_model 的结构:
{
    "ground_truth": "4",           # ← GT 答案
    "style": "rule",               # ← 打分方式 (rule / model / ...)
    "data_source": "perception",   # ← 数据集来源
}
```

---

### Q11: 面试速记表（补充）

| 编号 | 问题 | 3 秒答案 |
|------|------|---------|
| Q8 | 最后一个有效 token | response 序列中最后一个非 PAD token，通常就是 `</answer>`，`reward_tensor[i, valid_len-1] = score` |
| Q9 | Credit Assignment | 确定每个 step 对最终结果的贡献；长序列难因为稀疏 reward 下错误步骤也被鼓励 |
| Q10 | Judge Prompt + GT | GT 来自数据集 `reward_model["ground_truth"]`，Judge 用 System Prompt (verify_prompt.md) + User Prompt (问题+参考答案+模型回答) 判断一致性 |
