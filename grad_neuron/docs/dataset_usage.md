# Dataset usage

| Dataset/subset | Format | Current use | Status |
|---|---|---|---|
| AdvBench HarmBehavior, first 200 | Chat | Generated Llama-3 responses; 194 responses beginning with `I cannot` were used for the original local-gradient and IG rankings | Completed; historical ranking |
| AdvBench HarmBehavior, rows 0--39 | Raw | Selected the earlier fixed top-25 refusal-gradient neurons | Historical; overlaps the evaluation set |
| AdvBench HarmBehavior, rows 100--299 | Raw | Selects the current fixed top-25 refusal-gradient neurons | Current disjoint neuron-selection set |
| AdvBench HarmBehavior, first 100 | Raw | Consolidated safety evaluation with refusal-substring ASR | Current evaluation set |
| `zou_train`, 4,994 prompts | Raw | Source corpus used by the expanded-K/V reproduction | External reproduction corpus |
| `zou_train`, seed-112 sample of 200 | Raw | Generated responses to inspect detector-corpus behavior | Inspection only; not used to select our fixed neurons |

Our current reusable controller is **AdvBench-derived with prompt separation**:
one fixed set of 25 neurons is selected on rows 100--299 and evaluated unchanged
on rows 0--99. The two sets have zero index overlap and zero exact-prompt overlap.
The `zou_train` sample is retained for comparison with the expanded-K/V method.
