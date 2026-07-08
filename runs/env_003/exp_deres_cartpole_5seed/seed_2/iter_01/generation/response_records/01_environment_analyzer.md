# Response Record

# Env_003 环境理解卡片

## 1. 任务目标
在一个有限长度的轨道上，通过向左或向右对移动底座施加恒定大小的水平推力，使底座上未经驱动的摆杆尽可能长时间保持直立，同时底座不能超出轨道边界。任务本质是生存任务，目标是最大化存活时间步数，避免因杆倾倒或底座出界而终止。

## 2. 任务类型选择
selected_route_id: survival_balance_task
confidence: high
reason: 任务要求控制底座以维持不稳定杆的平衡，同时约束底座位置范围，没有明确的目标点或成功状态，唯一正向指标是存活步数，符合生存平衡类任务特征。

## 3. 观察空间 observation_space
- type: Box
- shape: [4]
- dtype: float32
- obs[0]: 底座水平位置，范围 [-4.8, 4.8]，超出 ±2.4 即终止，故有效范围为 (-2.4, 2.4)
- obs[1]: 底座水平速度，无界（但受动力学约束）
- obs[2]: 杆与竖直方向的夹角（弧度），范围 [-0.41887903, 0.41887903]，但终止阈值为绝对值大于0.20943951
- obs[3]: 杆的角速度，无界

## 4. 动作空间 action_space
- type: Discrete，2 个动作
- action 0: 向轨道负方向施加固定推力（push_negative_direction）
- action 1: 向轨道正方向施加固定推力（push_positive_direction）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无（不存在成功终止条件）
- failure-like termination: 
  - 底座位置绝对值超过 2.4（出界）
  - 杆角度绝对值超过 0.20943951 弧度（倾倒）
- ambiguous termination: 无
- truncation: 达到 500 步时截断（不代表成功或失败，只是时间限制）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 字典为空，{}）
- forbidden_or_uncertain_info_fields: 所有虚构字段（如 “success”, “failure”, “termination_reason” 等）均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs：4 维观测（底座位置、速度、杆角、角速度）
- action：当前时刻执行的动作（0 或 1）
- next_obs：下一时刻观测（4 维）
- info：当前为空字典 {}，无任何可用字段

禁止使用：
- original_reward：已强制屏蔽，不可使用、不可逆向工程
- training_progress：未明确授权，不可用
- 未声明的 info 字段
- 未声明的 obs 切片（仅上述 4 个元素可用）

## 7. 可用于奖励函数的信号
- 底座位置 (obs[0] / next_obs[0])
- 底座速度 (obs[1] / next_obs[1])
- 杆角度 (obs[2] / next_obs[2])，可计算距直立的偏差
- 杆角速度 (obs[3] / next_obs[3])
- 动作 (action)，可用于惩罚频繁换向等
- 存活步数隐含在 termination/truncation 中，但不可直接用原始奖励，可通过自定义奖励鼓励存活（如每一步给+1）

## 8. 不确定或不可用的信号
- original_reward：完全屏蔽，禁止使用
- info 内任何字段：info 恒为空，故所有假定字段均不存
- 终止原因细节：环境只返回 terminated 布尔值，无法从 info 中获取是出界还是倾倒，不能直接作为奖励函数输入源，只能通过 obs/next_obs 特征自行推断
