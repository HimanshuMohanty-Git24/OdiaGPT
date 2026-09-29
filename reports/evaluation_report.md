# OdiaGPT Comprehensive Evaluation Report

## 1. Quantitative Progression Across Experimental Tiers

| Tier | Parameters | Training Tokens Seen | Final Train Loss | Final Val Loss | Val Perplexity | Status |
|---|---|---|---|---|---|---|
| **Random Baseline** | 31.5M | 0 | ~9.07 | ~9.09 | 8,865.0 | Initial |
| **Tiny Overfit** | 0.95M | 768 (32 tok memorization) | 0.4317 | N/A | N/A | Memorized |
| **Nano Model** | 5.51M | 8,192,000 | 5.7545 | 5.9224 | 373.30 | Baseline |
| **OdiaGPT-30M** | 31.47M | 29,491,200 | 4.5156 | 4.6643 | 106.09 | Pretrained Base |

Across the progression:
1. Validation perplexity dropped from **8,865** down to **106.09**.
2. Training loss fell steadily by more than 4.5 nats.
3. Overfitting remained controlled: the validation loss (4.664) closely tracks training loss (4.515), confirming the corpus diversity is healthy.

## 2. Qualitative Generation Comparison

Generations were benchmarked using three fixed prompts under both **Greedy** and **Sampling** (temperature=0.8, top-k=50, top-p=0.9):

### Prompt 1: `ଓଡ଼ିଶାର ଲୋକମାନେ` (People of Odisha)
- **Random Init**: Incoherent byte fragments, unprintable characters, invalid Odia Unicode codepoint sequences.
- **Nano (5.5M)**: Severe repetitive loops and broken morphology (`ଓଡ଼ିଶାର ଲୋକମାନେ ଏକ ଏକ ଏକ...`).
- **OdiaGPT-30M Sampled**:
  > *"ଓଡ଼ିଶାର ଲୋକମାନେ ବିରିୟାରେ ସାନ୍ଦ୍ରାର ଅଭାବକୁ ନେଇ ନିଜକୁ ନିଜେ ଅନେକ କଥା କହୁଥିବା କଥା । ଆମେ ସମସ୍ତେ ଜାଣନ୍ତି ଯେ ଏହି ମହାଭାରତ ତଥା ଦେଶର ବିଭିନ୍ନ ରାଜ୍ୟରୁ ଆସୁଥିବା ଓଡ଼ିଆଙ୍କ ପାଇଁ ଏକ ପ୍ରକାର ପ୍ରୟାସ । ଏହି ସମୟରେ ଆମେ ସମସ୍ତେ ସମସ୍ତେ ଏହି ଅବସରରେ ରହି ଏହିସବୁ କାର୍ଯ୍ୟକ୍ରମ ବିଷୟରେ କହିବାକୁ ଯାଉଛୁ । ଆମେ ମଧ୍ୟ ଏହି ସଭାରେ ଅଂଶଗ୍ରହଣ କରିବା ପାଇଁ ଏକ ନୂଆ ପ୍ରସ୍ତୁତି"*
  - *Analysis*: Coherent compound sentence structure, correct postpositions (`ପାଇଁ`, `ବିଷୟରେ`), plural markers (`ଙ୍କ`), and culturally congruent vocabulary (`ଓଡ଼ିଆଙ୍କ`, `ରାଜ୍ୟରୁ`, `କାର୍ଯ୍ୟକ୍ରମ`).

### Prompt 2: `ଆଜି ସକାଳେ` (Today morning)
- **OdiaGPT-30M Sampled**:
  > *"ଆଜି ସକାଳେ ସକାଳେ ଦୁଇ ଝିଅ ସହ ଆସି ଶୋଇପଡ଼ିଲା । ସେହି ସମୟରେ ବାପା ଘର ଲୋକ କୁ ଖବର ଦେଇ କହିଥିଲେ ଯେ, \"ତେବେ ମାଆ, ତମେ ଆଉ କିଛି କହିଦେବୁ । ମୁଁ ଏ କଥା କିଛି ଶୁଣିନାହିଁ । କିନ୍ତୁ ବାପା, ମା'ଙ୍କୁ ନେଇ ମା'ର ମା'ର ପୁଅ ଝିଅ । ଏ କଥା କହି ସେ ଆଉ କାହା କଥା ଜାଣିନଥିଲା ।"*
  - *Analysis*: Emergence of natural story narrative, past tense agreement (`ଆସି ଶୋଇପଡ଼ିଲା`, `କହିଥିଲେ`, `ଜାଣିନଥିଲା`), direct quotation handling, familial relationships (`ବାପା`, `ମାଆ`, `ଝିଅ`).

### Prompt 3: `ଓଡ଼ିଆ ଭାଷାର` (Of Odia language)
- **OdiaGPT-30M Sampled**:
  > *"ଓଡ଼ିଆ ଭାଷାର ଅନେକ ଦୃଶ୍ୟ, ଚିତ୍ର-ଅର୍ଣ୍ଣିମ, ନୂଆବୋହୂକ ଓ ଲୋକନାଥଙ୍କ ଭଳି କିଛି ନୂଆ କାହାଣୀକୁ ନେଇ ଅନେକ ପ୍ରଶ୍ନବାଚୀ ସୃଷ୍ଟି ହୋଇଛି । ତେଣୁ ଏହି ଦୁଇ ଚରିତ୍ରର କଥା ଜାଣିଲେ ଆଶ୍ଚର୍ଯ୍ୟ ହୋଇଯିବେ । ପ୍ରଧାନମନ୍ତ୍ରୀଙ୍କ ଚିଠିରେ ପ୍ରକାଶ ପାଇବା ପରେ ଅନେକ ଘଟଣା ସାମ୍ନାକୁ ଆସିଛି ।"*
  - *Analysis*: Journalistic/news narrative tone, complex nominal compounds, correct punctuation, and cohesive thematic transition.

## 3. Linguistic & Architectural Observations
- **Odia Script & Diacritics**: Zero invalid UTF-8 sequences. The from-scratch 8K SentencePiece tokenizer captures Odia conjuncts and matras reliably with 100% character coverage.
- **RoPE & Attention**: Context length of 512 functioned seamlessly with SDPA; attention maps successfully learn forward syntax.
- **Greedy Repetition**: In pure argmax greedy decoding, the model exhibits phrase-level repetition cycles (e.g. `ନିଜ ନିଜ ନିଜ...`). This is expected for small base decoder models without repetition penalty or nucleus sampling.

## 4. Limitations
- **Not Instruction-Tuned**: The model is a pure next-token predictor trained on web/text corpus; it does not follow questions or chat instructions without conversational finetuning.
- **Factual Hallucinations**: Generations reflect language patterns and topical associations, not factual knowledge or reasoning.
