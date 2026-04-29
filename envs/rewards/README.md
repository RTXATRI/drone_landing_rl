# envs/rewards 目录说明

通用奖励计算器已经退役。奖励函数现在由 `curriculum/strategies/` 下的具体课程策略独立实现。

`reward_functions.py` 只保留 `RewardCalculator` 兼容壳；如果遗留代码继续调用它，会立即抛错，避免静默训练出无意义结果。

新增或修改奖励时，请直接在对应课程策略中维护公式、常量、注释和 reward CSV 字段。
