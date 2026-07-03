# Env_003 五 Seed 长实验运行指南

## 固定配置

- prefix: `exp_deres_cartpole_5seed`
- seeds: `0, 1, 2, 3, 4`
- reward-search rounds: `10`
- PPO timesteps per reward candidate: `100000`
- evaluation episodes per iteration: `20`
- solved threshold: `475`
- environment: `eure`

PPO参数来自给定的CartPole-v1配置：

```yaml
n_envs: 8
n_steps: 32
batch_size: 256
gae_lambda: 0.8
gamma: 0.98
n_epochs: 20
ent_coef: 0.0
learning_rate: lin_0.001
clip_range: lin_0.2
```

`lin_`参数已在训练包装器中转换为从初始值线性衰减到0的SB3 schedule。

## 启动前检查

在Git Bash中进入仓库，然后激活环境：

```bash
cd /c/Users/Administrator/Downloads/expert_eureka_env001_bridge_v9_direct_generator/expert_eureka_env001_bridge_v9_direct_generator
source /c/ProgramData/miniconda3/etc/profile.d/conda.sh
conda activate eure
python -c "import sys; print(sys.executable)"
```

输出应指向 `C:\ProgramData\miniconda3\envs\eure\python.exe`。

配置DeepSeek API Key并检查变量存在：

```bash
export DEEPSEEK_API_KEY="你的API Key"
python -c "import os; print(bool(os.environ.get('DEEPSEEK_API_KEY')))"
```

应输出 `True`。不要把API Key写入配置文件或日志。

## nohup后台启动

```bash
mkdir -p runs/env_003/logs
nohup bash scripts/run_env003_5seed.sh > runs/env_003/logs/exp_deres_cartpole_5seed.log 2>&1 &
echo $! > runs/env_003/logs/exp_deres_cartpole_5seed.pid
cat runs/env_003/logs/exp_deres_cartpole_5seed.pid
```

五个seed按顺序运行。单个seed可能在达到475分后根据停止规则提前结束，不一定执行满10轮。

## 查看进度

```bash
tail -f runs/env_003/logs/exp_deres_cartpole_5seed.log
```

查看已生成的seed和迭代目录：

```bash
find runs/env_003/exp_deres_cartpole_5seed -maxdepth 2 -type d | sort
```

检查进程：

```bash
ps -p "$(cat runs/env_003/logs/exp_deres_cartpole_5seed.pid)" -f
```

## 结果路径

```text
runs/env_003/exp_deres_cartpole_5seed/
  seed_0/
  seed_1/
  seed_2/
  seed_3/
  seed_4/
```

每个seed包含：

- `iter_XX/generation/`：奖励代码和LLM记录；
- `iter_XX/training/`：模型、评估结果和反馈；
- `memory/reward_memory.md`：搜索轨迹；
- `best/`：历史最佳奖励与训练结果；
- `experiment_summary.md`：最终汇总。

## 正常停止与异常处理

需要主动停止时，先读取PID并发送普通终止信号：

```bash
kill "$(cat runs/env_003/logs/exp_deres_cartpole_5seed.pid)"
```

不要同时启动第二个相同prefix的任务。异常退出后先检查日志和已经完成的seed，再决定是否使用单seed `--resume-from`，避免覆盖已有结果。
