# 验证目标梯度与 Fisher 近似的实验方案

## 核心问题

最有说服力的检验不是观察最终 ASR 是否改善，而是直接比较“近似预测值”和“有限编辑后的真实函数值”。需要区分三个问题：

1. Taylor 近似本身是否成立；
2. 梯度和 Fisher 的有限样本估计是否稳定、能否泛化；
3. 即使数学近似成立，$L_{\mathrm{tgt}}$ 和 $D_{\mathrm{gen}}$ 是否真的能代理安全与通用能力。

在 $\alpha=\mathbf 1$ 处，$\nabla L_{\mathrm{tgt}}$ 是精确导数，$F=\nabla^2D_{\mathrm{gen}}$ 也是精确等式。因此真正需要实验检验的是：它们能否外推到 `c=0.22` 和 `c=0.48` 对应的有限编辑。

## 1. 检验 $L_{\mathrm{tgt}}$ 的一阶近似

对一个实际编辑方向 $d=\Delta\alpha$，定义路径

$$
\alpha(t)=\mathbf 1+td,\qquad t\in[0,1].
$$

一阶近似预测为

$$
\Delta L_{\mathrm{lin}}(t)=t g^\top d,
\qquad
g=\nabla_\alpha L_{\mathrm{tgt}}(\mathbf 1),
$$

真实变化为

$$
\Delta L_{\mathrm{true}}(t)
=L_{\mathrm{tgt}}(\mathbf 1+td)-L_{\mathrm{tgt}}(\mathbf 1).
$$

### 1.1 测试方向与编辑半径

至少测试以下实际方向：

- Fisher 12k，`c=0.22`；
- Fisher 12k，`c=0.48`；
- Fisher 12k，`c=0.24`；
- Direct 12k，`s=0.3`；
- Direct 8k，`s=0.6`；
- gentle-12k sweep 中的其他方向，用于检验候选设置的排序。

建议使用

$$
t\in\{0,0.02,0.05,0.1,0.2,0.4,0.6,0.8,1.0\}.
$$

其中 $t=1$ 对应真实部署编辑，不能只报告很小 $t$ 下的结果。

### 1.2 主要指标

#### 数值校准

定义

$$
\rho_L(t)=
\frac{\Delta L_{\mathrm{true}}(t)}{t g^\top d}.
$$

理想情况下 $\rho_L(t)$ 接近 1。应重点报告 `c=0.22` 和 `c=0.48` 在 $t=1$ 时的比值及置信区间。

#### 方向判断

检查

$$
\operatorname{sign}(\Delta L_{\mathrm{true}})
=
\operatorname{sign}(g^\top d).
$$

如果符号不一致，一阶梯度不能支持该编辑方向。对于 $g^\top d$ 接近零的方向，应报告绝对误差而不是不稳定的相对误差。

#### 候选设置排序

在多个方向 $d_i$ 上比较

$$
g^\top d_i
\quad\text{和}\quad
L(\mathbf 1+d_i)-L(\mathbf 1)
$$

的 Spearman 相关、成对排序准确率和 top-$k$ 一致性。由于方法实际使用梯度来选择和分配神经元，能否选对候选设置比单点拟合更重要。

### 1.3 路径梯度诊断

计算路径上的方向导数

$$
s(t)=\nabla L_{\mathrm{tgt}}(\mathbf 1+td)^\top d.
$$

它满足精确恒等式

$$
L(\mathbf 1+d)-L(\mathbf 1)=\int_0^1s(t)\,dt.
$$

一阶近似等价于假设 $s(t)\approx s(0)=g^\top d$。因此应绘制 $s(t)/s(0)$：

- 曲线接近水平线：一阶近似合理；
- 曲线随 $t$ 快速下降：初始梯度高估收益；
- 曲线变号：该方向已经离开局部有效区域；
- `c=0.22` 平稳而 `c=0.48` 不平稳：说明小编辑支持理论，大编辑主要是经验工作点。

可使用 8 点或 16 点 Gauss--Legendre 积分计算路径积分，并验证积分结果与真实目标变化的 completeness。

### 1.4 单神经元排名检验

从不同层和排名区间分层抽样神经元，例如 top 100、101--1k、1k--4k、4k--12k 以及未入选神经元。对每个神经元比较

$$
L(\mathbf 1+\delta e_j)-L(\mathbf 1)
\quad\text{和}\quad
\delta g_j,
$$

其中 $\delta$ 取多个大小。报告梯度排名与真实单神经元效应的 Spearman 相关、符号一致率和 top-$k$ precision。这检验梯度神经元排名本身是否具有局部因果意义。

### 1.5 数据拆分与误差解耦

使用两组 SNCorpus raw 数据：

- 原 256 个 safe on-policy first-cue 样本：检验当前经验目标上的纯 Taylor 误差；
- 新的、互不重叠的 safe on-policy first-cue 样本：检验样本外泛化。

在验证集上重新计算 $g_{\mathrm{val}}$，然后分别检查：

1. 用 $g_{\mathrm{val}}$ 预测 $L_{\mathrm{val}}$：Taylor 近似误差；
2. 比较 $g_{\mathrm{train}}$ 和 $g_{\mathrm{val}}$：梯度估计和排序的泛化误差。

不能只用 $g_{\mathrm{train}}$ 预测验证集目标，否则 Taylor 误差和数据泛化误差会混在一起。用于数值验证的梯度应以 FP32 重新累计，不能直接把 FP16 保存误差计入近似误差。

## 2. 检验 $D_{\mathrm{gen}}$ 的 Fisher 近似

在 held-out WikiText raw contexts 上计算真实扰动代价：

$$
D_{\mathrm{true}}(td)
=
\mathbb E_x\left[
\mathrm{KL}\left(
p_{\mathbf 1}(\cdot\mid x)
\,\Vert\,
p_{\mathbf 1+td}(\cdot\mid x)
\right)
\right].
$$

Fisher 二阶预测为

$$
D_{\mathrm{quad}}(td)
=\frac12t^2d^\top Fd.
$$

沿与目标函数实验相同的 $t$ 网格直接比较二者。prefix 应从未编辑模型采样，并在相同 prefix 上计算未编辑模型和编辑模型之间的完整词表 KL；当前 `evaluate_actual_kls` 的计算方式适合这一检验。

### 2.1 分开检验 Fisher 二阶近似和对角近似

当前方法实际使用 diagonal Fisher，因此存在两层近似：

$$
D
\approx\frac12d^\top Fd
\approx\frac12\sum_jF_{jj}d_j^2.
$$

验证 Hessian 等于 Fisher 并不能自动验证 diagonal Fisher。对于任意给定方向，不必构造 12k×12k 的完整矩阵，可以直接估计完整方向曲率：

$$
d^\top Fd=\mathbb E[(s^\top d)^2].
$$

因此对每个实际方向同时计算

$$
q_{\mathrm{full-dir}}
=\frac12\mathbb E[(s^\top d)^2]
$$

和

$$
q_{\mathrm{diag}}
=\frac12\sum_jF_{jj}d_j^2.
$$

据此将误差拆成：

- $q_{\mathrm{diag}}/q_{\mathrm{full-dir}}$：删除非对角项造成的误差；
- $D_{\mathrm{true}}(td)/(t^2q_{\mathrm{full-dir}})$：有限步长造成的 Taylor 误差；
- 不同 context/probe seed 下的变化：Monte Carlo 估计误差。

方向曲率可以直接从 token-score Rademacher probe 的梯度向量得到：对每个 probe 向量 $v$ 累计 $(v^\top d)^2$。这保留交叉坐标项，而不需要保存完整 Fisher 矩阵。

### 2.2 局部曲率 sanity check

对很小的 $\varepsilon$ 计算

$$
h_{\mathrm{emp}}(d)
=
\frac{
D(\mathbf 1+\varepsilon d)
+D(\mathbf 1-\varepsilon d)
}{\varepsilon^2}.
$$

理论上应满足

$$
h_{\mathrm{emp}}(d)\approx d^\top Fd.
$$

同时定义方向非对称性

$$
A(t)=
\frac{D(td)-D(-td)}{D(td)+D(-td)}.
$$

二阶 Fisher 预测是对称的；明显的 $A(t)$ 表示三阶及更高阶项不可忽略。负方向只作为局部数值诊断，不作为最终 Grad 编辑设置。

### 2.3 需要报告的图和统计量

对每个方向绘制三条曲线：

- 真实 KL：$D_{\mathrm{true}}(td)$；
- 完整方向 Fisher：$\frac12t^2d^\top Fd$；
- diagonal Fisher：$\frac12t^2\sum_jF_{jj}d_j^2$。

另外绘制

$$
\frac{D_{\mathrm{true}}(td)}{\frac12t^2d^\top Fd}
$$

随 $t$ 的变化，从而直接确定 Fisher 的 trust region。所有置信区间应以 context 为重采样单位，而不是把同一 context 内的 token 当作独立样本。

还应使用独立的 Fisher context/probe seed 重复方向曲率估计，或至少对 per-context 方向曲率做 bootstrap，以判断 2,048 contexts 和 4 probes 是否已经足够稳定。

## 3. 检验代理目标的行为有效性

即使上述近似完全准确，也不代表代理目标一定能预测最终行为：

- first-cue teacher-forced $L_{\mathrm{tgt}}$ 可能不预测自由生成时的拒绝；
- WikiText 平均 KL 可能不预测 MATH、IFEval 或 BBH 的能力损失。

因此应进行独立的次级分析：

1. 对每个方向和 $t$ 记录真实 $\Delta L_{\mathrm{tgt}}$ 和真实 WikiText KL；
2. 在 HarmBench-47 raw tuning 上选择配置，随后直接在 HarmBench-200 raw test 上测 ASR 和 repetition；不使用 HarmBench-150 confirmation；
3. 在开发集上测 IFEval 和 MATH；
4. 分析真实 $\Delta L_{\mathrm{tgt}}$ 与 ASR 的相关性；
5. 在相近真实 KL 下比较 capability。

这检验的是 proxy validity，不应与 Taylor/Fisher approximation validity 混为一谈。已有 actual-KL matched controller 的安全和 IFEval 表现仍然不同，只能说明相同平均 KL 不保证相同行为，不能单独证明 Fisher 二阶展开错误。

## 4. 最小可行实验

第一阶段快速验证：

- 方向：`c=0.22`、`c=0.48`、Direct 12k `s=0.3`；
- $t=\{0,0.05,0.1,0.2,0.4,0.6,0.8,1\}$；
- $L_{\mathrm{tgt}}$：64--128 个新的 SNCorpus raw first-cue 样本；
- $D_{\mathrm{gen}}$：现有 256 个 held-out WikiText raw contexts；
- Fisher：同时计算 diagonal 和 full directional quadratic form；
- 统计：以 example/context 为单位做 paired bootstrap 置信区间。

在正式运行新的路径目标工作负载前，应先用从真实数据中抽取的最小集合 benchmark batch size，并将最优配置写入 runner。已有 Fisher batch-size benchmark 可以复用，但需要先确认新增的 full-directional 统计没有改变显存和吞吐最优点。

如果第一阶段显示 `c=0.22` 在 $t=1$ 仍可靠，再扩展到完整 gentle-12k sweep，检验跨候选方向的排序。

## 5. 判定方式

不要只给出“合理/不合理”的二元结论，而应分别报告最大有效半径：

$$
t_L^*=\max\{t:\text{一阶目标近似仍满足预注册误差标准}\},
$$

$$
t_F^*=\max\{t:\text{Fisher 二阶近似仍满足预注册误差标准}\}.
$$

可以预注册如下实用标准：

- 很小 $t$ 下的方向导数或局部曲率误差不超过 5%；
- 部署半径下预测和真实变化符号一致；
- 部署半径下校准比率的 paired-bootstrap 95% 区间位于 $[0.8,1.25]$；
- 跨候选方向的 Spearman 相关大于 0.8；
- 近似误差小于候选设置之间需要区分的效应量。

这些阈值应在查看完整结果前确定。最终最有价值的结论形式是：

> `c=0.22` 是否位于两个近似的有效域内，以及 `c=0.48` 是否已经超出其中一个近似的 trust region。
