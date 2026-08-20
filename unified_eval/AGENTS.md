This repo is for evaluating methods on editting safety related neurons in LLM. My research idea is 'Grad', the other two, IA3 posttrain and sn-tune, are from previous papers.
## benchmark selection
1. in your experiment, don't run on beavertrail for now. i find the dataset to be problematic after running some on it; so forget about beavertrail.
2. use raw format in sn-tune neuron selection, grad neuron selection and safety eval
3. don't run beaver score, gsm8k, mmlu ever again
## research question
1. i want to prove that there is a trade-off between safety alignment and general capability, like math.
## experiment details
1. in Grad, when editting neuron activation, default --grad-direction positive-only
2. for IA3, only use and report SNCorpus raw SFT IA3; other IA3 training data or formats are out of scope
3. commit and push any Markdown reports and figures that are created or updated
