# Env_001 Baseline 与消融实验方案

## 研究问题

1. DERES 的顺序自进化是否优于一次生成和等预算独立采样？
2. 结构化、局部且可验证的干预是否优于无约束顺序改写？
3. 专家 RAG、跨轮 Reward Memory 和组件级反馈分别提供了多少增益？
4. 生成奖励训练出的策略与官方环境奖励训练参考相比处于什么水平？

## 公平预算

- 训练 seed：0–4。
- 每个奖励候选：PPO 1M timestep。
- 每轮固定评估：20 局，seed 10000–10019。
- 比较搜索方法时：每 seed 最多训练 5 个候选。
- 最终 best：独立固定 seed 评估 100 局。
- 已完成的10轮主实验可继续用于长程搜索曲线。但它早于最近的target/route/restart公共修复，不能直接作为新消融的唯一控制组。消融主表使用当前冻结代码重新运行 `DERES-Control K=5`，避免代码版本混杂。

官方奖励 PPO 只训练一个候选，不参与 Success@K 的搜索预算比较；它是训练参考，不是奖励搜索算法。

## 匹配控制组：DERES-Control K=5

它不是新baseline，而是与以下消融完全相同代码版本、seed和候选预算的完整方法控制组。

```bash
nohup conda run --no-capture-output -n eure python -u -m pipeline.run_multi_seed_experiment --config configs/env001_deres_control_k5.yaml --start-seed 0 --num-seeds 5 > logs/control_lander_deres_k5_v1.log 2>&1 &
```

## Baseline

### Official-Reward PPO

不安装生成奖励 wrapper，使用环境原始奖励，PPO配置与主实验一致。

```bash
nohup conda run --no-capture-output -n eure python -u scripts/run_official_reward_baseline.py --config configs/env001_deepseek_rag.yaml --prefix baseline_lander_official_ppo_v1 --start-seed 0 --num-seeds 5 --total-timesteps 1000000 --eval-episodes 20 > logs/baseline_lander_official_ppo_v1.log 2>&1 &
```

### LLM-Once

优先直接读取主实验五个 seed 的 Iter1，避免重复训练。它与 DERES 使用相同初始生成器、RAG和PPO配置，只是不进行后续搜索。

如需独立复跑：

```bash
nohup conda run --no-capture-output -n eure python -u scripts/run_independent_reward_baseline.py --config configs/env001_deepseek_rag.yaml --prefix baseline_lander_llm_once_v1 --candidates 1 --start-seed 0 --num-seeds 5 --total-timesteps 1000000 --eval-episodes 20 > logs/baseline_lander_llm_once_v1.log 2>&1 &
```

### Independent Multi-Sample

每个 seed 独立生成5个候选。候选之间不共享奖励、分数、反馈或memory，训练seed保持一致。

```bash
nohup conda run --no-capture-output -n eure python -u scripts/run_independent_reward_baseline.py --config configs/env001_deepseek_rag.yaml --prefix baseline_lander_independent_k5_v1 --candidates 5 --start-seed 0 --num-seeds 5 --total-timesteps 1000000 --eval-episodes 20 > logs/baseline_lander_independent_k5_v1.log 2>&1 &
```

### Unconstrained Sequential

保留顺序反馈、RAG、memory和best，但取消Level 1/2/3、结构化诊断字段与单组件局部干预约束。

```bash
nohup conda run --no-capture-output -n eure python -u -m pipeline.run_multi_seed_experiment --config configs/env001_baseline_unconstrained_sequential.yaml --start-seed 0 --num-seeds 5 > logs/baseline_lander_unconstrained_seq_v1.log 2>&1 &
```

## 核心消融

### w/o Expert RAG

同时关闭初始专家上下文、Reflection route提示和知识工具。环境事实、反馈、memory及结构化干预保持不变。

```bash
nohup conda run --no-capture-output -n eure python -u -m pipeline.run_multi_seed_experiment --config configs/env001_ablation_no_rag.yaml --start-seed 0 --num-seeds 5 > logs/ablation_lander_no_rag_v1.log 2>&1 &
```

### w/o Reward Memory

LLM不接收跨轮历史表，restart也不接收已尝试结构摘要；外层best保留，以免同时消融安全归档机制。

```bash
nohup conda run --no-capture-output -n eure python -u -m pipeline.run_multi_seed_experiment --config configs/env001_ablation_no_memory.yaml --start-seed 0 --num-seeds 5 > logs/ablation_lander_no_memory_v1.log 2>&1 &
```

### Score-Only Feedback（辅助消融）

Reflection只看到外部分数、episode长度和终止分布；移除组件构成。Memory保留分数历史，但删除组件统计列。

```bash
nohup conda run --no-capture-output -n eure python -u -m pipeline.run_multi_seed_experiment --config configs/env001_ablation_score_only.yaml --start-seed 0 --num-seeds 5 > logs/ablation_lander_score_only_v1.log 2>&1 &
```

## 报告指标

- Best return @ K，K=1…5。
- Success@K，解决阈值200。
- 首次达到200所需候选数和累计PPO timestep。
- Reward rescue rate：Iter1未解决但K轮内解决的seed比例。
- 5-seed最终best均值、bootstrap 95% CI和所有seed散点。
- LLM调用数、无效代码数、重复候选数和wall-clock。

核心论文表格优先放 DERES、LLM-Once、Independent-K5、Unconstrained Sequential 和 Official PPO。消融图放 w/o RAG、w/o Memory；Score-Only可在版面不足时移到附录。
