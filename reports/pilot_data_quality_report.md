# Pilot Odia data quality report

This is a prefix of the fixed verified/ori stream; it is not a random sample.
- Dataset: ai4bharat/sangraha, verified/ori
- Revision: 8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8
- Documents inspected: 68,579
- Retained: 67,983
- Rejected: 596
- Cleaned characters: 110,001,302
- Train / validation documents: 67,330 / 653
- Length mean / median: 1618.1 / 1103.0
- Odia ratio mean / median: 0.9926 / 1.0000
- Raw NFC documents: 68,553
- Exact boilerplate lines removed: 9,005
- Documents affected by boilerplate removal: 9,001
- Boilerplate list SHA256: 91ba65e2f8c0bcfcb6552010cace261b1bef88d8fc3fd432dceddbe72e0d346d

## Rejection reasons

- duplicate_doc_id: 1
- few_odia_characters: 159
- too_long_for_pilot: 436

## Source types

- pdf: 10,184
- speech: 3
- web: 58,392

Exact duplicate text removal and document-level splitting precede token packing.
Retained existing nano validation IDs remain in validation for this corpus.
