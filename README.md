# expert_eureka_env001_bridge_v9_direct_generator

当前版本是 config-driven iterative reward agent：

```text
iter_01:
  Environment Analyzer LLM
  → RAG compressor
  → Reward Generator LLM
  → reward_v1.py
  → PPO training
  → training_feedback.md / training_summary.json
  → reward_memory.md

iter_02+:
  previous training_feedback.md + reward_memory.md + matched expert cards
  → iteration_context.md with Iteration Control Decision
  → Reward Revision LLM
  → reward_vN.py
  → PPO training
  → update reward_memory.md
  → update best reward / early stop if needed
```

## 关键机制

- 目标分数：Env_001 的 solved threshold 是 `target_score: 200.0`。
- 最终结果使用 best reward，不使用最后一轮 reward。
- 每轮训练后自动更新 compact memory。
- 每轮 revision 前自动写入 `Iteration Control Decision`。
- 已解决后如果新 reward 掉回 target 以下，停止并保留 best。
- 已解决后如果 revision 生成 identical reward，停止并保留 best。
- 未解决阶段如果 revision 生成 identical reward，会自动追加 no-op retry instruction 并重试；仍 identical 则停止，避免白跑训练。
- 连续无明显提升会 early stop，避免 10 轮里后半段空转。

## 主要配置

```yaml
iteration:
  total_rounds: 10
  experiment_prefix: exp_iter
  experiment_root: runs/env_001/experiments
  memory_path: runs/env_001/memory/reward_memory.md
  reset_memory_at_start: true
  cards_top_k: 4
  stop_on_invalid_reward: true
  use_mock_llm: false

  target_score: 200.0
  save_best_artifacts: true
  stop_after_solved_drop: true
  stop_when_solved_and_identical: true
  no_improvement_patience_after_solved: 2
  no_improvement_patience_unsolved: 3
  min_meaningful_improvement: 5.0
  retry_identical_when_unsolved: true
  max_identical_revision_retries: 2
```

`cards_top_k` 不是 reward 组件数，而是每轮最多放入 `iteration_context.md` 的 failure / reward-misalignment 专家卡片数量。

## 启动实验

建议直接使用 Python orchestrator，因为它支持 seed、rounds、mock 等完整参数：

```bash
export DEEPSEEK_API_KEY="你的key"

python -m pipeline.run_iterative_experiment \
  --config configs/env001_deepseek_rag.yaml \
  --prefix exp_iter \
  --seed 0
```

临时覆盖轮数：

```bash
python -m pipeline.run_iterative_experiment \
  --config configs/env001_deepseek_rag.yaml \
  --prefix exp10 \
  --rounds 10 \
  --seed 0
```

mock 流程测试：

```bash
python -m pipeline.run_iterative_experiment \
  --config configs/env001_deepseek_rag.yaml \
  --prefix mock_iter \
  --seed 0 \
  --mock
```

跑 seed0-5：

```bash
for s in 0 1 2 3 4 5; do
  python -m pipeline.run_iterative_experiment \
    --config configs/env001_deepseek_rag.yaml \
    --prefix exp10 \
    --rounds 10 \
    --seed "$s"
done
```

## 输出目录

每个 seed 独立保存：

```text
runs/env_001/experiments/seed_<SEED>/<PREFIX>/iter_01/
runs/env_001/experiments/seed_<SEED>/<PREFIX>/iter_02/
...
runs/env_001/experiments/seed_<SEED>/<PREFIX>/best/
runs/env_001/experiments/seed_<SEED>/<PREFIX>/experiment_summary.md
```

训练输出：

```text
runs/env_001/training_runs/experiments/seed_<SEED>/<PREFIX>/iter_XX/training/
```

每个 seed 的 memory：

```text
runs/env_001/memory/seed_<SEED>/reward_memory.md
```

## 主要查看文件

```text
runs/env_001/memory/seed_<SEED>/reward_memory.md
runs/env_001/experiments/seed_<SEED>/<PREFIX>/experiment_summary.md
runs/env_001/experiments/seed_<SEED>/<PREFIX>/best/best_reward.py
runs/env_001/experiments/seed_<SEED>/<PREFIX>/best/best_summary.md
runs/env_001/experiments/seed_<SEED>/<PREFIX>/iter_XX/iteration_context.md
runs/env_001/experiments/seed_<SEED>/<PREFIX>/iter_XX/generation/reward_vX.py
runs/env_001/training_runs/experiments/seed_<SEED>/<PREFIX>/iter_XX/training/training_feedback.md
```

## 方法借鉴

本项目借鉴 `frde-hdrc` 的方法，而不是照搬其内容：

- 用历史 reward 和分数指导下一轮，不每轮从零生成；
- 根据 score trend、episode length、组件结构判断 tune / add / delete / rebuild；
- 只记录 compact memory，而不是全量日志；
- 当前项目额外加入 RAG：每轮只给命中的 failure / reward-misalignment 专家卡片；
- 达到目标后保护 best reward，而不是继续盲目迭代。

## 单独生成迭代上下文

```bash
python -m pipeline.run_04_build_iteration_context \
  --train-run-dir runs/env_001/training_runs/experiments/seed_0/exp10/iter_01/training \
  --memory runs/env_001/memory/seed_0/reward_memory.md \
  --cards knowledge_base/iteration/reward_misalignment_cards.md \
  --top-k 4 \
  --out runs/env_001/experiments/seed_0/exp10/iter_02/iteration_context.md
```

## 单独更新 memory

```bash
python -m pipeline.run_06_update_reward_memory \
  --iter 1 \
  --train-run-dir runs/env_001/training_runs/experiments/seed_0/exp10/iter_01/training \
  --memory runs/env_001/memory/seed_0/reward_memory.md \
  --target-score 200 \
  --best-score 210 \
  --best-iter 1 \
  --decision target_solved_new_best
```
