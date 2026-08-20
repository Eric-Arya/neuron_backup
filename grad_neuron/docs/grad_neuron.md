## §1 设定、记号与命题 0（焊入的精确性）

**设定.** 取 decoder-only Transformer 第 $l$ 层 MLP，隐藏维 $H$，FFN 中间维 $d:=d_{\text{ff}}$。在 token 位置 $t$：

$$
h_t^{(l+1)} \;=\; h_t^{(l)} + \sum_{j=1}^{d} a_j^{(t)}\, W_j,\qquad W_j := W_{\text{down}}^{(l)}[:,j]\in\mathbb{R}^{H},
$$

其中 $a_j^{(t)}$ 是该神经元的标量激活（其具体形式，SwiGLU/GELU 等，对下文无影响）。

**干预算子.** 引入参数向量 $\alpha\in\mathbb{R}^{d}$（每层一个，初值 $\alpha=\mathbb{1}$），将第 $j$ 个神经元的写入改为 $\alpha_j\, a_j^{(t)} W_j$。记扰动后模型分布为 $p_\alpha$，$\Delta\alpha := \alpha-\mathbb{1}$。

**命题 0（焊入精确性）.** 干预后的模型与一个标准模型功能等价：把 $W_{\text{down}}[:,j]$ 替换为 $\alpha_j W_{\text{down}}[:,j]$ 即可，无任何近似。

**证明.** $\sum_j \alpha_j a_j^{(t)} W_j = \sum_j a_j^{(t)}(\alpha_j W_j)$，代入即得。$\blacksquare$

推论：该方法的零推理开销是定理而非工程结论——因为干预从一开始就被定义在权重列缩放的参数化里。

---

## §2 定理 1（GradAct 恒等式）：归因分数 = 精确偏导

**定理 1.** 设 $\mathcal{L}$ 为任意关于前向计算可微的标量目标。则对第 $l$ 层任一神经元 $j$：

$$
g_j \;=\ \frac{\partial \mathcal{L}}{\partial \alpha_j} \;=\; \sum_{t} a_j^{(t)}\cdot W_j^{\top}\,\nabla_{h_t^{(l+1)}}\mathcal{L}.
$$

进一步，在 $\alpha=\mathbb{1}$ 处，$W_j^{\top}\nabla_{h_t^{(l+1)}}\mathcal{L} = \dfrac{\partial \mathcal{L}}{\partial a_j^{(t)}}$，于是

$$
g_j \;=\ \left.\frac{\partial \mathcal{L}}{\partial \alpha_j}\right|_{\alpha=\mathbb{1}} \;=\; \sum_t \underbrace{a_j^{(t)}}_{\text{激活}}\cdot\underbrace{\frac{\partial \mathcal{L}}{\partial a_j^{(t)}}}_{\text{梯度}}.
$$


即"激活 × 梯度（逐位置求和）"不是启发式，而是 $\mathcal{L}$ 对干预参数 $\alpha_j$ 的**精确偏导数**。

**证明.** $h_t^{(l+1)}$ 对 $\alpha_j$ 的依赖只经过加性项 $\alpha_j a_j^{(t)}W_j$（$a_j^{(t)}$ 在干预点之前计算，不依赖 $\alpha$），故

$$
\frac{\partial h_t^{(l+1)}}{\partial \alpha_j} = a_j^{(t)}W_j \in \mathbb{R}^{H},
$$

对下游一切模块为常量。由链式法则，

$$
\frac{\partial \mathcal{L}}{\partial \alpha_j}
= \sum_t \left(\frac{\partial h_t^{(l+1)}}{\partial \alpha_j}\right)^{\!\top} \nabla_{h_t^{(l+1)}}\mathcal{L}
= \sum_t a_j^{(t)} W_j^{\top}\nabla_{h_t^{(l+1)}}\mathcal{L}.
$$

第二断言：$a_j^{(t)}$ 进入计算图同样只经过 $\alpha_j a_j^{(t)}W_j$，故 $\partial\mathcal{L}/\partial a_j^{(t)} = \alpha_j W_j^{\top}\nabla_{h_t^{(l+1)}}\mathcal{L}$，在 $\alpha_j=1$ 处取值即得。$\blacksquare$

---

## §3 Llama-3 “I cannot” pilot

For a harmful request $q_i$, format the normal chat prompt and prefill the assistant
with the single token `I`. The scalar refusal score is

$$
s_i(\alpha)=\log p_\alpha(\texttt{" cannot"}\mid
\operatorname{chat}(q_i)+\texttt{"I"}).
$$

`" cannot"` is one token (ID 4250) for the local Meta-Llama-3-8B-Instruct
tokenizer. The per-example neuron signal and dataset-level detection strength are

$$
g_{i,l,j}=\frac{\partial s_i}{\partial\alpha_{l,j}},\qquad
\bar g_{l,j}=\frac{1}{N}\sum_i g_{i,l,j},\qquad
D_{l,j}=|\bar g_{l,j}|.
$$

A positive $\bar g$ supports the refusal pivot; a negative $\bar g$ suppresses it.
$D$ is the primary strength ranking. Mean absolute gradient, standard deviation,
sign consistency, and $|\bar g|/\mathrm{std}(g)$ are retained to diagnose signals
that are large but unstable across requests.

The pilot selects only stored generations whose response begins exactly with
`I cannot`. Scaling is global across all token positions, so the extracted
gradient retains the exact weight-column interpretation of §1–2.

Run with:

```bash
bash scripts/extract_i_cannot_gradients.sh
```

---

## §4 Standard Integrated Gradients pilot

Standard IG moves every neuron-scale coordinate jointly along one straight path. For baseline
$b=0.9$ and the actual model $\alpha=\mathbf 1$,

$$
\alpha(\tau)=b\mathbf 1+\tau(1-b)\mathbf 1,\qquad
\operatorname{IG}_{l,j}=(1-b)\int_0^1
\frac{\partial s(\alpha(\tau))}{\partial\alpha_{l,j}}\,d\tau.
$$

The score $s$ and the 194-example selection are unchanged from §3. The implementation uses
16-point Gauss--Legendre quadrature and FP32 model computation. FP32 is required here: the BF16
pilot produced poor completeness because the small pathwise score differences were quantized.

Completeness is checked per example:

$$
\sum_{l,j}\operatorname{IG}_{l,j}\approx s(\mathbf 1)-s(b\mathbf 1).
$$

Run with:

```bash
bash scripts/extract_integrated_gradients.sh
```
