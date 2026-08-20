# Fixed top-50 neuron sweep

## What was evaluated

The fixed global intervention set was increased from 25 to 50 neurons. Strengths `s={0.5,0.75,1.0}` were evaluated on the same HarmBehavior-100 and GSM8K-100 examples as the top-25 sweep. Scaling remains `positive=1+s` and `negative=max(0,1-s)`.

## Results

| Strength | K=25 ASR ↓ | K=50 ASR ↓ | K=25 GSM8K ↑ | K=50 GSM8K ↑ |
|---:|---:|---:|---:|---:|
| 0.50 | 30% | **24%** | 67% | 66% |
| 0.75 | **22%** | 27% | **64%** | 58% |
| 1.00 | **19%** | 20% | **62%** | 57% |

At `s=0.5`, K=50 trades one additional GSM8K point for six ASR points relative to K=25. The paired K=50-versus-K=25 differences are not individually significant (ASR exact `p=0.146`; GSM8K `p=1.0`), but this condition adds a useful intermediate operating point: 24% ASR and 66% GSM8K.

At `s=0.75` and `s=1`, K=50 is dominated by K=25: it has both higher ASR and lower GSM8K accuracy. Against baseline, `K=50,s=1` lowers ASR by 14 points (paired exact `p=0.0094`) while lowering GSM8K by 11 points (unadjusted `p=0.0347`).

Conclusion: increasing the intervention set is useful only with a weaker intervention. The best choices currently are `K=50,s=0.5` when capability preservation matters and `K=25,s=1` when minimizing ASR matters.

Artifacts are under `results/tradeoff/fixed_top50_strength_sweep/`. The matched comparison is in `k25_vs_k50_paired_comparison.csv`.
