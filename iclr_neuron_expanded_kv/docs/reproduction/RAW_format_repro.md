# Chat-format detection + deactivation-rate benchmark (GPU 1)

- Preflight: 19/19 detector/evaluator tests passed; only physical GPU 1 is assigned.
- Detection: chat, `zou_train` 200/200 (0 failures), top attn/FFN=1000/2000; retained `[7790, 7790, 14784, 14784, 4960]` (`fwd_up,fwd_down,q,k,v`; total 50,108).
- Rate 0: GSM8K chat 0-shot `67/100` (67%); safety chat ASR `2/100` (2%).
- Rate 0.0002 (262 channels): GSM8K chat 0-shot `61/100` (61%); safety chat ASR `2/100` (2%).
- Rate 0.0004 (524 channels): GSM8K chat 0-shot `64/100` (64%); safety chat ASR `6/100` (6%).
- Rate 0.0008 (1,049 channels): GSM8K chat 0-shot `14/100` (14%; paired exact McNemar vs rate 0: `p=3.11e-15`); safety chat ASR `6/100` (6%).
- Raw safety, rate 0.0002 (chat-discovered neurons; 262 channels): ASR `53/100` (53%).
- Raw safety, rate 0.0004 (chat-discovered neurons; 524 channels): ASR `49/100` (49%).
