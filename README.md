# Information Retrieval
Use Dense Passage Retrieval (DPR) for information retrieval.

Gradient Cached Dense Passage Retrieval ([GC-DPR](https://github.com/luyug/GC-DPR)) - is an extension of the original DPR library.
With GC-DPR , you can reproduce the state-of-the-art open Q&A system trained on 8 x 32GB V100 GPUs with a single 11 GB GPU.

## Dataset: Zalo AI 2021

## Follow train:
- Preprocessing data
- Split train, dev set
- Use bm25 get top_100 document
- Create data for training DPR from top 100 document bm25
- Use the first 50 documents as negative
- Use the remaining 50 documents as hard negative
- Train DPR, get top k document for a query
