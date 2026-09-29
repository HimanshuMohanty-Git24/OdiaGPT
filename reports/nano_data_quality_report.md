# Nano Odia data quality report

This is a prefix of the fixed verified/ori stream; it is not a random sample.
- Dataset: ai4bharat/sangraha, verified/ori
- Revision: 8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8
- Documents inspected: 12,297
- Retained: 12,199
- Rejected: 98
- Cleaned characters: 20,004,188
- Train / validation documents: 12,069 / 130
- Length mean / median: 1639.8 / 1118.0
- Odia ratio mean / median: 0.9906 / 1.0000
- Raw NFC documents: 12,293

## Rejection reasons

- few_odia_characters: 24
- too_long_for_nano: 74

## Source types

- pdf: 1,859
- web: 10,438

Exact duplicate text removal and document-level splitting precede token packing.
Existing smoke validation IDs remain in validation for this corpus.
