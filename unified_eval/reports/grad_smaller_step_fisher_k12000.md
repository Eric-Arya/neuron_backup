# Grad Fisher 小步 endpoint 最小验证（k=12k）

## 目的

检验沿已有 Fisher 方向缩短一次更新，能否缓解大步线性化失真和重复生成。这里的“小步”是从原模型一次性到达缩小后的 endpoint，不是多步迭代，也不重新计算梯度。

按照当前数据协议，只使用 HarmBench-47 raw tuning 选择配置，随后直接在固定 HarmBench-200 raw test 上测试；不再生成或使用 HarmBench-150 confirmation split。

## 设置

- 模型：Meta-Llama-3-8B-Instruct，FP32；
- Grad：positive-only、tail scope、`k=12000`；
- 基础方向：`c=.24` 和 `c=.48`，floor 0、cap .75、damping 1；
- 候选半径：`t=.2,.3,.4,.5,.6`；
- 安全评估：raw prompt、greedy decoding、128 new tokens；
- 已有真实数据 benchmark 被复用，batch size 为 16；
- finalist 规则：先最小化 HB47 ASR；若安全相同，再选择 repetition 更少者。

## HB47 line search

| 方向 | t | ASR | 攻击成功数 | repetition |
|---|---:|---:|---:|---:|
| c=.24 | .2 | 48.94% | 23/47 | 8/47 |
| c=.24 | .3 | 42.55% | 20/47 | 13/47 |
| c=.24 | .4 | 38.30% | 18/47 | 13/47 |
| c=.24 | .5 | 29.79% | 14/47 | 15/47 |
| c=.24 | .6 | 27.66% | 13/47 | 18/47 |
| c=.48 | .2 | 46.81% | 22/47 | 12/47 |
| c=.48 | .3 | 42.55% | 20/47 | 13/47 |
| c=.48 | .4 | 27.66% | 13/47 | 14/47 |
| c=.48 | .5 | 23.40% | 11/47 | 14/47 |
| **c=.48** | **.6** | **21.28%** | **10/47** | **15/47** |

`c=.48,t=.6` 是这 10 个点中的明确 finalist：ASR 最低，而且 repetition 并非最高。它在两个指标上都优于 `c=.24,t=.6`。

## HB200 test

| 设置 | ASR | repetition | 空回答 |
|---|---:|---:|---:|
| c=.24, t=1（已有结果） | 7.0% (14/200) | 94/200 | 0 |
| c=.48, t=1（已有结果） | 1.5% (3/200) | 110/200 | 0 |
| **c=.48, t=.6** | **10.0% (20/200)** | **82/200** | **0** |

相对 `c=.48,t=1`，缩短到 `t=.6` 使 repetition 从 110 降至 82（减少 28），但 ASR 从 1.5% 上升至 10.0%。相对较均衡的 `c=.24,t=1`，它使 repetition 减少 12，但 ASR 恶化 3 个百分点。

## 与真实 target 和 KL 的对应

这些数值复用同方向、同 endpoint 的 256 条 first-cue target 与 256 条 WikiText validation context 验证结果：

| 设置 | 真实 target gain | 真实 sequence KL |
|---|---:|---:|
| c=.48, t=.6 | 0.5532 | 0.2871 |
| c=.48, t=1 | 0.3824 | 0.8067 |
| c=.24, t=1 | 0.4871 | 0.3653 |

`c=.48,t=.6` 相对其完整 endpoint 将真实 KL 降低约 64.4%，同时真实 first-cue target gain 反而更高。这符合“大步越过 target objective 最佳区域”的诊断。然而，它的安全性明显更弱：first-cue teacher-forced target objective 不能单独预测 HarmBench 安全行为。

## 结论

最低成本方案验证了两个不同结论：

1. 缩小同一 Fisher 方向确实能显著降低 KL 和重复生成，因此更小 delta 是有效的退化控制旋钮。
2. 单纯缩短冻结方向不能同时保持原大步的安全性；在本次候选中，没有点在相同安全水平下支配 `c=.24` 或 `c=.48` 的完整 endpoint。

因此 `c=.48,t=.6` 可以作为较温和的 safety--degeneration trade-off 候选，但不是对原方法的全面改进。下一项最有信息量的实验应是 midpoint/Heun 或逐步重算梯度：保持较小局部步长，同时允许方向随当前位置改变。尚未运行 IFEval/MATH，所以本报告不宣称 repetition/KL 的改善已经转化为通用或数学能力提升。
