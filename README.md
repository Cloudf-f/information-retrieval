# Information Retrieval

## Dataset: Zalo Ai 2021

## Follow train:
- Preprocessing data
- Split train, dev set
- Use bm25 get top_100 document
- Create data for training DPR from top 100 document bm25
- Use the first 50 documents as negative
- Use the remaining 50 documents as hard negative