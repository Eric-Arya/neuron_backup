# Smaller-step Fisher：改善一次大步线性化的方案

## 核心判断

使用更小的 $\Delta\alpha$ 可以改善 $L_{\mathrm{tgt}}$ 的一阶近似，但必须区分两种情况：

1. 把同一个大步机械地拆成多个小步，中间不重新计算梯度：最终 $\alpha$ 完全相同，不能解决线性化误差。
2. 每走一个小步都重新计算梯度并修正方向：这才是真正的 iterative re-linearization。

当前 `k=12k, c=.24/.48` 的验证结果表明，一阶近似的点误差在 $t\le .01$ 时低于 5%，按 paired-bootstrap 95% 区间完全位于 `[0.8, 1.25]` 的标准，有效半径约为 $t\le .02$。由于完整方向的最大 delta 为 0.75，$t=.02$ 时单神经元最大 delta 只有

$$
0.75\times0.02=0.015.
$$

这种单步编辑可能不足以产生强安全效果。因此推荐的方法不是把最终编辑永久限制在一个极小 delta，而是通过多个重新线性化的小步累积编辑。

实验同时表明：完整 directional Fisher 从 $t=.02$ 到 $1$ 都能准确预测真实 WikiText KL，主要快速失效的是 $L_{\mathrm{tgt}}$ 的起点梯度。因此最具性价比的策略是频繁更新 $g$、低频更新 $F$。

## 1. 最低成本方案：沿现有方向做真实 line search

暂时不改变神经元方向，只沿现有方向搜索最终编辑比例：

$$
\alpha(t)=\mathbf1+t\Delta\alpha.
$$

已有目标路径结果为：

| 设置 | 真实 $L_{\mathrm{tgt}}$ 最佳测试点 | 目标增益 | 真实 WikiText KL |
|---|---:|---:|---:|
| `c=.24` | $t=.4$ | 0.5802 | 0.0577 |
| `c=.48` | $t=.4$ | 0.5887 | 0.1277 |
| `c=.24`, endpoint | $t=1$ | 0.4871 | 0.3653 |
| `c=.48`, endpoint | $t=1$ | 0.3824 | 0.8067 |

因此 $t=.4$ 同时获得更高的真实 target objective 和显著更低的通用分布 KL。但其安全效果未知，不能仅根据 teacher-forced target objective 决定。

### 最小实验

保持已有 `c=.24/.48` 方向不变，测试：

$$
t\in\{.2,.3,.4,.5,.6\}.
$$

协议：

- Grad direction 保持 positive-only；
- SNCorpus、HarmBench 和安全评估均使用 raw format；
- 先在 HarmBench-47 tuning split 筛选；
- 再在 disjoint HarmBench-150 confirmation split 确认；
- 同时记录 repetition、真实 $L_{\mathrm{tgt}}$ 和真实 WikiText KL；
- 不接触 frozen HarmBench、IFEval 或 MATH，直到选出明确 finalist；
- 不运行 BeaverTrail、Beaver score、GSM8K 或 MMLU。

这一步不能证明一阶近似在 $t=.4$ 仍然准确。此时真实目标增益仅为线性预测的约 24%--32%。它的作用是绕过不准确的线性增益预测，直接用真实目标和开发集行为选择步长。

## 2. 推荐主方法：迭代式投影自然梯度

令第 $m$ 步的 multiplier 为 $\alpha_m$，计算当前点的目标梯度：

$$
g_m=\nabla_\alpha L_{\mathrm{tgt}}(\alpha_m).
$$

使用 diagonal Fisher 作为计算便宜的预条件器：

$$
d_m=(F_{\mathrm{diag}}+\lambda I)^{-1}g_m.
$$

更新采用投影：

$$
\alpha_{m+1}
=
\operatorname{clip}
\left(
\alpha_m+\eta_m d_m,
1,
1.75
\right).
$$

投影到 `[1, 1.75]` 有两个作用：

- 最终模型始终满足 positive-only 约束；
- 如果某个已放大的神经元在新位置梯度变负，可以将其向 1 拉回，而不是继续错误放大。

每一步通过 backtracking line search 选择 $\eta_m$，要求：

1. held-out 实际 $L_{\mathrm{tgt}}$ 提高；
2. 当前步或累计的真实 WikiText KL 不超过预算；
3. HarmBench tuning repetition 不显著恶化；
4. 如果没有候选步长满足条件，则停止。

伪代码：

```text
alpha = 1
for step in 1 ... M:
    recompute g at alpha
    direction = (F_diag + damping)^(-1) g

    for eta in [1, .5, .25, .125, ...]:
        candidate = clip(alpha + eta * direction, 1, 1.75)
        evaluate actual L_tgt and general KL
        if L_tgt improves and KL is feasible:
            accept candidate
            break

    stop if no candidate improves L_tgt
```

### Fisher 更新频率

第一版应固定现有 2,048-context diagonal Fisher，只在每一步重算 $g_m$。原因是：

- 完整 directional Fisher 在现有两个方向上直到 $t=1$ 都保持准确；
- 当前实验确认主要失效来自目标梯度，而不是 Fisher 二阶模型；
- 重算 256-example target gradient 远便宜于重算 2,048-context Fisher。

只有在以下情况之一出现时才重算 Fisher：

- 当前 directional Fisher 对真实 step KL 的误差超过预注册阈值；
- 累计 base-model KL 超过一个阶段预算；
- 新梯度方向与先前方向的 cosine similarity 明显下降；
- 每隔固定的若干步进行一次低频刷新。

### KL 步长控制

当前 diagonal Fisher 在两个方向上将真实 KL 低估约 32%--37%，但误差接近稳定的乘法偏差。第一版可使用保守校准：

$$
\widehat D_{\mathrm{gen}}
=
1.5\cdot\frac12\sum_jF_{jj}\Delta\alpha_j^2.
$$

在现有两个 endpoint 上，`1.5×` diagonal 分别为 0.3460 和 0.8270，对应真实 KL 0.3653 和 0.8067，误差约为 -5.3% 和 +2.5%。

这个 `1.5×` 因子只能作为 pilot 的保守工程校准，不能直接宣称普适。正式实验应在 calibration split 上确定因子，并在独立 confirmation split 或更多方向上验证。对于每个新方向，也可以直接估计

$$
d_m^\top Fd_m=\mathbb E[(s^\top d_m)^2],
$$

以很低的存储成本获得 full-directional step curvature。

## 3. 更省梯度次数：midpoint / Heun natural gradient

如果不希望执行很多小步，可以使用 predictor--corrector，只增加一次梯度计算。

### Midpoint 更新

先在起点计算：

$$
d_0=(F+\lambda I)^{-1}g_0.
$$

走到预测中点：

$$
\alpha_{1/2}
=
\operatorname{clip}
\left(
\alpha_0+\frac{\eta}{2}d_0,
1,
1.75
\right).
$$

在中点重新计算：

$$
g_{1/2}=\nabla L_{\mathrm{tgt}}(\alpha_{1/2}),
\qquad
d_{1/2}=(F+\lambda I)^{-1}g_{1/2}.
$$

最终更新为：

$$
\alpha_1
=
\operatorname{clip}
\left(
\alpha_0+\eta d_{1/2},
1,
1.75
\right).
$$

### Heun 更新

也可以先构造 trial endpoint，在那里计算 $g_{\mathrm{trial}}$，然后使用平均方向：

$$
d_{\mathrm{Heun}}
=
\frac12(F+\lambda I)^{-1}
\left(g_0+g_{\mathrm{trial}}\right).
$$

Midpoint/Heun 是优先级很高的 pilot：实现简单，只比当前闭式方法多一遍 target gradient，却能利用路径中部信息修正起点切线。

## 4. 改善目标函数的饱和

当前目标是单个 first-cue span 的 mean log probability：

$$
L_{\mathrm{tgt}}=\log p(\text{cue}).
$$

它最大只能达到 0。当前未编辑均值为 -0.8995，因此最大可能增益只有 0.8995，但线性模型对 `.24/.48` 分别预测 4.56 和 6.16，必然在大步下严重失真。

可以改为 refusal-vs-unsafe 的 log-odds 或 contrastive margin：

$$
L_{\mathrm{margin}}
=
\log P(\mathcal R\mid x)
-
\log P(\mathcal U\mid x),
$$

其中 $\mathcal R$ 是 refusal cue 集合，$\mathcal U$ 是 unsafe/on-policy continuation 集合。另一种形式是：

$$
L_{\mathrm{contrast}}
=
\log p_\alpha(y_{\mathrm{safe}}\mid x)
-
\log p_\alpha(y_{\mathrm{unsafe}}\mid x).
$$

潜在优势：

- 减轻单一 cue 概率接近 1 后的上界饱和；
- 直接优化安全回答相对有害回答的优势；
- 降低 `illegal`、`I'm just` 等偶然 cue 对梯度的控制；
- 可能比单 cue teacher forcing 更接近自由生成安全行为。

改变目标不能单独保证任意大步变得线性，因此仍应与 iterative re-linearization 或 line search 配合。

## 5. 刷新 on-policy target

当前 first-cue 目标及其 response prefix 来自未编辑模型。随着 $\alpha$ 改变，这些 prefix 会逐渐变成 off-policy，固定目标的梯度可能不再描述编辑模型真实生成时遇到的状态。

可以采用两级更新：

1. 每一步在固定 target 上重新计算 $g_m$，纯粹修复有限步 Taylor 误差；
2. 每若干步用当前模型重新生成 SNCorpus raw responses、提取新的 first refusal cue 和 prefix，再刷新目标集合。

这两种更新应在实验中分开消融，以区分：

- gradient rotation；
- target/prefix distribution shift；
- 新 on-policy 样本选择造成的变化。

刷新 target 时仍需过滤重复响应，并保留独立 held-out target set，防止每一步只拟合当前训练生成。

## 6. 低维非线性子空间优化

另一条路线是不直接在 12,000 个坐标上做非线性优化，而是收集多个局部自然梯度方向：

$$
d_0=F^{-1}g(\alpha_0),
\quad
d_1=F^{-1}g(\alpha_1),
\quad
d_2=F^{-1}g(\alpha_2).
$$

然后只在这些方向张成的低维空间中搜索：

$$
\Delta\alpha
=
\beta_0d_0+
\beta_1d_1+
\beta_2d_2.
$$

这样只需优化 2--5 个 $\beta$ 系数，可以直接使用真实 $L_{\mathrm{tgt}}$ 和真实 KL，而不依赖全局线性目标。它相当于用多个局部切线拟合一条弯曲路径。

最终仍需将 $\alpha$ 投影到 `[1, 1.75]`，保持 positive-only 与 cap 约束。

## 7. 建议的实验顺序

### 阶段 A：现有方向的低成本 line search

1. 对 `.24/.48` 生成 $t=.2,.3,.4,.5,.6$ scale artifacts。
2. 在 HarmBench-47 raw tuning split 上测 ASR 和 repetition。
3. 对可行点在 HarmBench-150 raw confirmation split 上确认。
4. 同时记录真实 target objective 和 WikiText KL。
5. 如果 $t=.3$--`.5` 能保留足够安全性，则先把它作为更温和 endpoint。

### 阶段 B：Midpoint/Heun pilot

1. 固定当前 diagonal Fisher。
2. 使用起点梯度生成 predictor。
3. 在一个中点或 trial endpoint 重算梯度。
4. 生成 midpoint 与 Heun 两种 corrected direction。
5. 用与阶段 A 相同的开发协议比较实际 target gain、KL、ASR 和 repetition。

### 阶段 C：迭代式 projected natural gradient

1. 先固定 target corpus 和 Fisher，只在每一步重算 $g$。
2. 使用小步、backtracking 和 `[1,1.75]` 投影。
3. 记录每一步的实际目标、预测目标、真实 KL 和 safety development 指标。
4. 对比机械拆步控制：相同累计步长但始终使用 $g_0$。
5. 只有 iterative 版本优于机械拆步时，才能把收益归因于 re-linearization。

### 阶段 D：目标刷新和 contrastive objective

分别增加 on-policy target refresh 和 refusal-vs-unsafe objective，做正交消融。不要同时改变梯度更新、目标定义和 Fisher，否则无法识别改善来源。

## 8. 必须包含的控制组

- 原始一次大步 `.24/.48`；
- 同一方向的最佳固定 $t$；
- 把同一大步机械拆成 $M$ 步、但不更新梯度；
- 每步重算梯度的 iterative 方法；
- midpoint 或 Heun；
- 如果改变目标，保留原 first-cue objective 的同算法控制。

关键比较是：

> 相同最终 KL 或相同安全水平下，重新线性化是否提高真实 target objective、降低 repetition，并保留更多 IFEval/MATH 能力。

## 9. 判定标准

推荐预先确定：

- 每一步实际 target gain 必须为正；
- 每一步实际/预测 target gain 位于预设 calibration band；
- 每步与累计 KL 均不得超过预算；
- confirmation ASR 必须满足预设 feasibility threshold；
- repetition 不能通过退化换取表面安全；
- 最终方法应在相近安全或相近真实 KL 下与一次大步比较能力。

应分别报告：

- 单步最大可靠半径；
- 累计步数与总 KL；
- 梯度方向随步数的 cosine similarity；
- 被投影回 1 或截断到 1.75 的神经元数量；
- 固定 target 与 refreshed target 的差异。

## 优先级结论

1. **立即执行**：现有 `.24/.48` 方向的 $t=.2$--`.6` HarmBench development line search。
2. **首个算法改进**：midpoint/Heun natural gradient，只增加一次梯度计算。
3. **正式方法候选**：固定 Fisher、每步重算 $g$ 的 iterative projected natural gradient。
4. **后续增强**：周期性刷新 on-policy target，并测试 refusal-vs-unsafe contrastive objective。

最重要的原则是：

> 更小的 delta 只有在中途重新计算梯度、修正方向或提前停止时，才能真正改善一次大步线性化；把相同 endpoint 机械拆成小步不会改变结果。
