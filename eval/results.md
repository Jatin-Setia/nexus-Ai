Retrieval evaluation - 30 questions, top 6 chunks, 6 PDFs

| Method | Hit@1 | Hit@3 | Hit@6 | MRR |
|---|---|---|---|---|
| Vector only | 63.3% | 76.7% | 83.3% | 0.711 |
| BM25 only (keyword) | 76.7% | 86.7% | 96.7% | 0.826 |
| Hybrid (vector + BM25 + RRF) | 70.0% | 90.0% | 93.3% | 0.789 |

By question type (Hit@6):
| Method | Keyword-style | Paraphrased |
|---|---|---|
| Vector only | 77.8% (n=18) | 91.7% (n=12) |
| BM25 only (keyword) | 94.4% (n=18) | 100.0% (n=12) |
| Hybrid (vector + BM25 + RRF) | 94.4% (n=18) | 91.7% (n=12) |

Questions with no correct chunk in the top 6:
- Vector only: O05, C01, C02, C04, C05
- BM25 only (keyword): C02
- Hybrid (vector + BM25 + RRF): O05, C04
