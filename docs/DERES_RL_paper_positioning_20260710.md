# DERES 强化学习方向论文定位与实验叙事

## 1. 方法定位

建议方法名：

**DERES: Diagnosis-guided Reward Function Self-Evolution with Large Language Models**

中文可写作：

**诊断引导的奖励函数自进化搜索方法**

本文不应把“LLM 能生成奖励函数”本身作为核心创新，因为 Eureka、Text2Reward、RF-Agent、RDA 等工作已经覆盖了 LLM reward design、reward code search、agentic reward optimization 等方向。DERES 更适合主张：

> 将失败奖励函数的训练结果转化为结构化诊断证据，并通过层级化、可归因的局部奖励干预，使奖励函数从低质量初始设计逐步自进化到可解设计。

也就是说，论文重点不是“生成一个好奖励”，而是“失败奖励不是废样本，它可以变成下一轮搜索的证据”。

## 2. 框架模块

8 个组件太碎，论文图和方法章节建议收敛成 5 个高层模块：

1. **Task Modeling with Environment Understanding LLM**  
   输入 masked environment interface，由环境理解 LLM 生成任务摘要和 environment card。  
   对应消融：`w/o Env LLM`。

2. **Initial Reward Proposal**  
   LLM 根据任务摘要生成初始奖励函数代码。  
   对应 baseline：`LLM-once`，即只使用 Iter1，不进行后续自进化。

3. **Policy Training and External Evaluation**  
   用生成奖励训练 PPO 策略，用环境外部评价分数判断策略是否解决任务。

4. **Reward Evidence System**  
   将训练与评估结果组织成反思 agent 可用的证据，包括外部分数、episode 结果、组件统计表、reward memory、上一轮代码和 best code。  
   对应消融：`w/o Evidence` 或 `Score-only Feedback`。

5. **Hierarchical Evolution Policy**  
   根据证据选择 L1/L2/L3 干预层级：尺度修复、数学结构变换、骨架重建，并默认每轮只修改一个目标组件。  
   对应消融：`w/o L1/L2/L3` 或 `Unconstrained Sequential`。

其中最适合写成核心贡献的是 **Hierarchical Evolution Policy + Reward Evidence System** 的组合：前者决定怎么改，后者决定凭什么改。

## 3. 当前已完成实验

### Env_001 主实验

实验前缀：`runs/env_001/paper_v4`  
任务 solved threshold：200  
设置：5 seed，最多 10 轮，每个候选 PPO 训练 1M steps，搜索评估 20 episodes。

| seed | Iter1 | best | solved |
|---:|---:|---:|:---:|
| 0 | -70.35 | 224.21 | yes |
| 1 | -42.74 | 240.60 | yes |
| 2 | -17.90 | 220.24 | yes |
| 3 | -19.59 | 253.71 | yes |
| 4 | 139.53 | 206.14 | yes |

汇总：

| metric | value |
|---|---:|
| Iter1 mean ± std | -2.21 ± 73.38 |
| best mean ± std | 228.98 ± 16.54 |
| solved seeds | 5/5 |

可写结论：

> Env_001 中 5 个 seed 的初始奖励均未达到 solved threshold，而 DERES 在最多 10 轮内将 5/5 个 seed 搜索到可解奖励函数，说明低质量初始奖励可以通过诊断式自进化被持续修复。

### Env_001：w/o Hierarchical Evolution Policy

实验前缀：`runs/env_001/ablation_unconstrained_v4`  
含义：保留顺序迭代、反馈、memory 和 best，但取消 L1/L2/L3 层级选择与单组件局部干预约束。

| seed | Iter1 | best | solved |
|---:|---:|---:|:---:|
| 0 | -121.66 | 169.90 | no |
| 1 | -107.52 | -40.47 | no |

当前结论：

> 取消层级演化策略后，LLM 仍可能偶尔提升分数，但修改过程更容易震荡，当前 2 个 seed 均未达到 solved threshold。这支持 DERES 的收益不只是“多轮调用 LLM”，而来自有层级、有归因边界的奖励干预策略。

注意：该消融目前已有初步证据，但若作为主文强结论，建议补到 5 seed。

### Env_001：LLM-once

`LLM-once` 可直接使用 `paper_v4` 的 Iter1 作为结果，因为它与 DERES 使用同一初始生成流程。

| seed | LLM-once score | solved |
|---:|---:|:---:|
| 0 | -70.35 | no |
| 1 | -42.74 | no |
| 2 | -17.90 | no |
| 3 | -19.59 | no |
| 4 | 139.53 | no |

可写结论：

> 单次生成在 Env_001 上 0/5 solved，而 DERES 最终 5/5 solved，说明自进化搜索是必要的。

### Env_001：Official PPO Reference

实验前缀：`runs/env_001/baseline_lander_official_ppo_v1`  
当前记录只有 2 个 seed：

| seed | official PPO score | solved |
|---:|---:|:---:|
| 0 | 263.63 | yes |
| 1 | 269.88 | yes |

汇总：266.76 ± 3.13，2/2 solved。

注意：这是官方奖励 PPO 参考上界，不是奖励函数搜索方法。若论文表格要放 official PPO，建议补齐到 5 seed。

## 4. Env_002 实验记录

### Env_002 主实验

实验前缀：`runs/env_002/paper_bipedal_main_v1`  
任务 solved threshold：300  
设置：5 seed，最多 10 轮，每个候选 PPO 训练 5M steps，搜索评估 20 episodes。

| seed | Iter1 | best | solved |
|---:|---:|---:|:---:|
| 0 | 270.71 | 320.02 | yes |
| 1 | 280.50 | 313.73 | yes |
| 2 | 272.07 | 307.92 | yes |
| 3 | 289.99 | 311.12 | yes |
| 4 | 103.03 | 304.92 | yes |

汇总：

| metric | value |
|---|---:|
| Iter1 mean ± std | 243.26 ± 70.45 |
| best mean ± std | 311.54 ± 5.17 |
| solved seeds | 5/5 |

更稳妥的论文表述：

> Env_002 表明 DERES 在连续控制步态任务上同样能产生可解奖励函数。但该环境中 4 个 seed 的初始奖励已经较强，因此它更适合作为跨任务有效性证据，而不是主要证明“迭代救援”的证据。seed4 从 103.03 提升到 304.92，可作为低质量初始奖励被修复的补充案例。

### Env_002：环境理解相关实验记录

当前可用记录如下：

| experiment | model / setting | env card | expert prior | score | note |
|---|---|:---:|:---:|---:|---|
| `ablation_direct_no_env_card_v1` | direct generation | no | unclear | 286.12 | 单次生成，未 solved |
| `ablation_direct_no_expert_v2` | DeepSeek-v4-pro | no | no | 311.33 | 强模型仍能 solved |
| `ablation_direct_no_expert_v3_chat` | deepseek-chat | no | no | -59.76 | 弱模型失败 |
| `test_chat_with_env_card` | deepseek-chat | yes | yes | 286.83 | 加入 env card 后明显恢复，但未 solved |
| `ablation_no_expert_profile_v1` | facts-only env analyzer | yes, facts-only | yes | 288.89 ± 17.75 | 4 seed，1/4 solved |

可写结论：

> 环境理解模块不是在所有模型上都显著提高上限。对 DeepSeek-v4-pro 这类强模型，Env_002 的 locomotion 结构较简单，模型即使缺少环境卡和专家先验也可能生成可解奖励。但对 deepseek-chat，缺少环境理解时奖励设计失败，而加入 environment card 后分数恢复到 286.83。这说明 Environment Understanding LLM 更像稳定化模块：它降低弱模型或不稳定生成条件下的任务误解风险，而不是单独决定最终性能上限。

## 5. PPO 参数设置

### Env_001 PPO

配置来源：`configs/env001_deepseek_rag.yaml`

| parameter | value |
|---|---:|
| env | LunarLander-v3 |
| policy | MlpPolicy |
| n_envs | 4 |
| total_timesteps per candidate | 1,000,000 |
| eval_episodes during search | 20 |
| n_steps | 1024 |
| batch_size | 64 |
| gamma | 0.999 |
| gae_lambda | 0.98 |
| n_epochs | 4 |
| ent_coef | 0.01 |
| normalize_obs | false |
| normalize_reward | false |

### Env_002 PPO

配置来源：`configs/env002_deepseek_rag.yaml`

| parameter | value |
|---|---:|
| env | BipedalWalker-v3 |
| policy | MlpPolicy |
| n_envs | 32 |
| total_timesteps per candidate | 5,000,000 |
| eval_episodes during search | 20 |
| n_steps | 2048 |
| batch_size | 64 |
| gamma | 0.999 |
| gae_lambda | 0.95 |
| n_epochs | 10 |
| ent_coef | 0.0 |
| learning_rate | 0.0003 |
| clip_range | 0.18 |
| normalize_obs | false |
| normalize_reward | false |

## 6. 待补实验优先级

### 已完成或基本完成

1. **LLM-once**：已完成，可直接使用 Env_001/Env_002 主实验 Iter1。
2. **w/o Hierarchical Evolution Policy**：Env_001 已有 2 seed 初步结果，建议补到 5 seed。

### 仍建议补充

1. **Budget-Matched Independent Search**  
   同样训练 K 个奖励，但每个奖励独立生成，不利用前一轮失败证据。  
   目的：证明 DERES 不是单纯靠“多采样”，而是利用失败证据进行顺序自进化。

2. **w/o Environment Understanding LLM**  
   已有 Env_002 单 seed/direct 证据，但如果要作为正式消融，建议做更规范的 3-5 seed。  
   目的：证明 environment card 对弱模型或不稳定设置有稳定化作用。

3. **w/o Reward Evidence System / Score-only Feedback**  
   只给总分、长度和终止信息，不给组件统计、memory、上一轮代码/best code。  
   目的：证明结构化证据比纯分数反馈更适合引导奖励修复。

### 不建议作为主消融

- **w/o Expert Design Priors / w/o RAG**：当前论文叙事里不再强调 RAG，且 Env_002 结果显示强模型在简单步态任务上可以不依赖专家先验。它可以作为附录讨论，不建议作为主贡献消融。
- **w/o Best Archive**：更像工程安全机制，不适合单独包装成创新点。

## 7. 和相关工作的区分

### 已有工作

- **Potential-based reward shaping** 是经典理论工具，不应包装成新公式。
- **Text2Reward** 已经做了基于 LLM 的 dense reward code 生成。
- **Eureka** 已经做了 LLM reward code 的进化式优化。
- **RF-Agent** 将奖励设计建模为序贯决策过程，并用搜索算法管理 reward design。
- **RDA** 使用视觉轨迹理解做语义失败诊断和 reward code 修订。

### DERES 应强调的差异

1. **失败奖励再利用**：失败候选不是直接丢弃，而是转化为下一轮奖励修复证据。
2. **证据驱动的局部修复**：组件统计、历史代码、best code 和外部分数共同构成 Reward Evidence System。
3. **层级演化策略**：L1/L2/L3 将奖励修改限制在尺度、数学形态、骨架三个层级，降低无约束 LLM 迭代震荡。
4. **低质量初始奖励救援证据**：Env_001 的 5 seed 结果显示 0/5 初始 solved，但最终 5/5 solved。

## 8. 论文标题建议

首选：

**DERES: Diagnosis-Guided Reward Function Self-Evolution with Large Language Models**

强调实验现象：

**From Failed Rewards to Solved Policies: Diagnosis-Guided Reward Self-Evolution with Large Language Models**

更贴强化学习方向：

**Structured Reward Function Self-Evolution for Reinforcement Learning via Diagnostic Language Agents**
