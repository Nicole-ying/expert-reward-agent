# 实验复现与消融审计

## 一、主实验的历史配置

归档主实验由 `run_paper_v4.sh` 调用 `configs/env001_paper_v4.yaml`。该配置继承 `configs/env001_deepseek_rag.yaml`，并明确启用 structured feedback、structured reflection、expert RAG 和 reward memory。

| 项目 | paper-v4 值 |
|---|---:|
| 环境 | LunarLander-v3 |
| 算法 | Stable-Baselines3 PPO |
| 独立搜索 lineage | 5（seed 0–4） |
| 每条 lineage 最大 reward evaluations | 10 |
| 每个 candidate 训练步数 | 1,000,000 |
| native evaluation episodes | 20 |
| target score | 200 |
| PPO vector environments | 4 |
| `n_steps` / `batch_size` | 1024 / 64 |
| `gamma` / `gae_lambda` | 0.999 / 0.98 |

## 二、历史提交中已有的对照配置

`cafceeb9` 同时保留了以下配置，但“文件存在”不等于“已形成可直接写进论文的新实验结果”。重跑时必须保持 seed、训练预算、evaluation episodes 和停止规则一致，并单独核对输出。

| 配置 | 操作变量 | 可回答的问题 |
|---|---|---|
| `env001_ablation_score_only.yaml` | 将反馈压缩为 score-only | 结构化训练证据是否优于单一分数 |
| `env001_ablation_unconstrained_reflection.yaml` | 使用 unconstrained reflection | 受限 L1/L2/L3 编辑是否提高搜索稳定性 |
| `env001_ablation_no_rag.yaml` | 关闭 expert RAG | 检索知识对该历史实现的影响 |
| `env001_ablation_no_memory.yaml` | 关闭 reward memory | 历史版本曾预留该条件；当前论文不把 memory 声称为已被单独证明的性能来源 |
| `env001_baseline_unconstrained_sequential.yaml` | 顺序但无受限反思 | 与完整 CREATE 的历史基线候选 |

论文当前的核心机制证据应围绕 **structured evidence** 与 **bounded editing** 展开。Memory 是 agent 持久状态的组成部分，但在没有可靠独立结果时，不写“memory 必然带来性能提升”，也无需主动讨论负面消融。

## 三、Baseline 边界

- 初始奖励 `R₀` 的 native score 是搜索起点，可作为“初始化性能”报告，但它不是独立方法 baseline。
- 真正的搜索 baseline 应在相同 LLM/任务输入和相同 PPO 预算下，移除训练证据驱动的闭环修复，例如独立生成或无反馈顺序生成。
- 官方环境 reward 训练 PPO 只能作为任务参考上界/参考条件，不能替代 reward-search baseline。

## 四、本论文不纳入的后续探索

Subagent investigator、额外 Component delta、Formula switching guide 和 paper-v4 之后新增的 prompt/config 不属于这份冻结方法。若未来需要研究它们，应新建实验前缀与单独论文版本，不覆盖 `paper_v4` 结果，也不回写为本次投稿的方法组件。

## 五、重跑必须保留的证据

- 每轮 reward 源码与 validation report
- 固定 PPO 配置、seed、训练预算和 native evaluation 结果
- component statistics、training feedback 和训练轨迹
- Reflection Agent 的 System/User Prompt、响应和工具调用记录
- reward memory
- Best Archive 及其首次达标轮次

任何新结果替换论文数字时，摘要、正文、表格、图、图注和结论必须同步更新。
