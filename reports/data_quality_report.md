# Initial Odia data quality report

This report covers a small prefix of the verified Odia stream. It is not a representative random sample.

- Dataset: `ai4bharat/sangraha`, configuration `verified`, split `ori`
- Fixed dataset revision: `8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8`
- Documents inspected: 893
- Documents retained: 886
- Documents rejected: 7
- Exact duplicate texts rejected: 0
- Duplicate document IDs rejected: 0
- Raw documents already NFC: 893
- Cleaned characters: 1,500,127
- Cleaned length mean / median: 1693.1 / 1089.0
- Odia letter/mark ratio mean / median: 0.9925 / 1.0000
- Opening-line prefixes seen in at least three documents: 0 (detection only)
- Train / validation documents: 877 / 9

## Source types among inspected documents

- `pdf`: 126
- `web`: 767

## Rejection reasons

- `few_odia_characters`: 2
- `too_long_for_smoke`: 5

## Short raw / clean comparisons

Only the first 70 characters of each version are shown.

| Example | Raw prefix | Clean prefix |
|---|---|---|
| 1 | କନକ ବ୍ୟୁରୋ : ଅନୁଗୁଳ ହଣ୍ଡପା ଷ୍ଟେସନ ନିକଟରେ ଆଜମେର-ପୁରୀ ଏକ୍ସପ୍ରେସ୍ ଇଞ୍ଜିନ୍ | କନକ ବ୍ୟୁରୋ : ଅନୁଗୁଳ ହଣ୍ଡପା ଷ୍ଟେସନ ନିକଟରେ ଆଜମେର-ପୁରୀ ଏକ୍ସପ୍ରେସ୍ ଇଞ୍ଜିନ୍ |
| 2 | ବାରିପଦାଃ କୂଅ ଭିତରେ ଖସି ପଡ଼ିଲା ଛୁଆ ହାତୀ । ମୟୂରଭଞ୍ଜ ଜିଲ୍ଲା ରାସଗୋବିନ୍ଦପୁର | ବାରିପଦାଃ କୂଅ ଭିତରେ ଖସି ପଡ଼ିଲା ଛୁଆ ହାତୀ । ମୟୂରଭଞ୍ଜ ଜିଲ୍ଲା ରାସଗୋବିନ୍ଦପୁର |
| 3 | ମୁମ୍ବାଇ ୦୮। ୦୭ଃ ବଲିଉଡର ଭେଟେରାନ ଅଭିନେତ୍ରୀ ନୀତୁ କପୁର ଆଜି ତାଙ୍କର ୬୪ ତମ ଜନ | ମୁମ୍ବାଇ ୦୮। ୦୭ଃ ବଲିଉଡର ଭେଟେରାନ ଅଭିନେତ୍ରୀ ନୀତୁ କପୁର ଆଜି ତାଙ୍କର ୬୪ ତମ ଜନ |

## Unicode observations

NFC normalization was applied. Odia combining marks, punctuation, digits, and code-mixed text were preserved by the cleaning rules. The initial 200-row inspection found no unwanted controls, but that does not establish their absence from the full dataset.

Document splitting occurred before token packing. Exact normalized text duplicates were removed before the split.
