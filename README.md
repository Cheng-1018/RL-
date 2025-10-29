# 强化学习（Reinforcement Learning）学习笔记

本仓库包含强化学习的学习笔记和代码实现，涵盖从基础理论到高级算法的完整学习路径。

## 📚 目录

### 理论篇

1. **[马尔可夫决策过程](note/1.马尔可夫决策过程.md)**
   - 马尔可夫性质
   - 马尔可夫奖励过程
   - 马尔可夫决策过程
   - 价值函数与贝尔曼方程
   - 策略迭代与价值迭代

2. **[表格型方法](note/2.表格型方法.md)**
   - 蒙特卡洛方法
   - 时序差分学习
   - Q-Learning算法
   - Sarsa算法
   - 免模型预测与控制

3. **[策略梯度](note/3.策略梯度.md)**
   - 策略梯度定理
   - REINFORCE算法


4. **[PPO算法](note/4.PPO算法.md)**
   - PPO-Clip与PPO-Penalty

5. **[DQN介绍](note/5.DQN.md)**
   - 状态价值函数
   - 动作价值函数
   - 目标网络
   - 探索
   - 经验回放
6. **[DQN进阶技巧](note/6.DQN.2.md)**
   - DDQN
   - dueling DQN
   - PER
   - TD(N)
   - 噪声网络
   - 分布式Q函数
   - 彩虹
7. **[连续动作DQN](note/7.DQN.3.md)**
   - 对动作采样
   - 梯度上升
   - 设计网络架构
   
### 实践篇

- **[Q-Learning实现](code/QLearning.ipynb)** - 基于Gymnasium的Q-Learning算法完整实现
- **[代码示例](code/)** - 各种强化学习算法的Python实现

## 🛠️ 环境配置

```bash
# 克隆仓库
git clone https://github.com/Cheng-1018/RL-.git
cd RL-

# 安装依赖
pip install -r requirements.txt
```

### 依赖包
- `gymnasium` - 强化学习环境
- `matplotlib` - 数据可视化
- `seaborn` - 统计图形
- `numpy` - 数值计算
