import argparse
import os
import csv
import glob
import json
import gzip
import logging
import pickle
import time
from typing import List, Tuple, Dict, Iterator

import numpy as np
import torch
from torch import Tensor as T
from torch import nn
from tqdm import tqdm
from dpr.data.qa_validation import calculate_matches
from dpr.models import init_biencoder_components
from dpr.options import add_encoder_params, setup_args_gpu, print_args, set_encoder_params_from_state, \
    add_tokenizer_params, add_cuda_params
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint
from dpr.indexer.faiss_indexers import DenseIndexer, DenseHNSWFlatIndexer, DenseFlatIndexer

import re
def preprocess(text):
    try:
        text = re.sub("\t"," ",text)
        text = re.sub("\n"," ",text)
        text = re.sub("^Điều [0-9][0-9]*?. "," ",text).strip()
        text = re.sub("^[0-9][0-9]*?. "," ",text).strip()
    except:
        return ""
    return text

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

class DenseRetriever(object):
    """
    Does passage retrieving over the provided index and question encoder
    """
    def __init__(self, question_encoder: nn.Module, batch_size: int, tensorizer: Tensorizer, index: DenseIndexer):
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.index = index

    def generate_question_vectors(self, questions: List[str]) -> T:
        n = len(questions)
        bsz = self.batch_size
        query_vectors = []

        self.question_encoder.eval()

        with torch.no_grad():
            for j, batch_start in enumerate(range(0, n, bsz)):

                batch_token_tensors = [self.tensorizer.text_to_tensor(q) for q in
                                       questions[batch_start:batch_start + bsz]]

                q_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
                q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
                q_attn_mask = self.tensorizer.get_attn_mask(q_ids_batch)
                _, out, _ = self.question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

                query_vectors.extend(out.cpu().split(1, dim=0))

                # if len(query_vectors) % 100 == 0:
                #     logger.info('Encoded queries %d', len(query_vectors))

        query_tensor = torch.cat(query_vectors, dim=0)

        # logger.info('Total encoded queries tensor %s', query_tensor.size())

        assert query_tensor.size(0) == len(questions)
        return query_tensor

    def get_top_docs(self, query_vectors: np.array, top_docs: int = 100) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        time0 = time.time()
        results = self.index.search_knn(query_vectors, top_docs)
        # logger.info('index search time: %f sec.', time.time() - time0)
        return results


parser = argparse.ArgumentParser()
add_encoder_params(parser)
add_tokenizer_params(parser)
add_cuda_params(parser)

parser.add_argument('--ctx_file', type=str, default="src/Reranker/legal_data/legal-docs.tsv",
                        help="All passages file in the tsv format: id \\t passage_text \\t title")
parser.add_argument('--encoded_ctx_file', type=str, default="legal_corpus_index/dpr/legal.50_0.pkl",
                        help='Glob path to encoded passages (from generate_dense_embeddings tool)')
parser.add_argument('--match', type=str, default='string', choices=['regex', 'string'],
                        help="Answer matching logic type")
parser.add_argument('--n-docs', type=int, default=100, help="Amount of top docs to return")
parser.add_argument('--validation_workers', type=int, default=16,
                        help="Number of parallel processes to validate results")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size for question encoder forward pass")
parser.add_argument('--index_buffer', type=int, default=100000,
                        help="Temporal memory data buffer size (in samples) for indexer")
parser.add_argument("--hnsw_index", action='store_true', help='If enabled, use inference time efficient HNSW index')
parser.add_argument("--save_or_load_index", action='store_true', help='If enabled, save index')

parser.add_argument("--encode_q_and_save", action='store_true')
parser.add_argument("--re_encode_q", action='store_true')
parser.add_argument("--q_encoding_path")

args = parser.parse_args()
args.model_file = "dpr/hard_neg_50/dpr_biencoder.5.40"
assert args.model_file, 'Please specify --model_file checkpoint to init model weights'
setup_args_gpu(args)
print_args(args)

saved_state = load_states_from_checkpoint(args.model_file)
set_encoder_params_from_state(saved_state.encoder_params, args)

tensorizer, encoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)

encoder = encoder.question_model

encoder, _ = setup_for_distributed_mode(encoder, None, args.device, args.n_gpu,
                                            args.local_rank,
                                            args.fp16)
encoder.eval()

# load weights from the model file
model_to_load = get_model_obj(encoder)
logger.info('Loading saved model state ...')

prefix_len = len('question_model.')
question_encoder_state = {key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if
                              key.startswith('question_model.')}
model_to_load.load_state_dict(question_encoder_state)
vector_size = model_to_load.get_out_size()
logger.info('Encoder vector_size=%d', vector_size)

if args.hnsw_index:
    index = DenseHNSWFlatIndexer(vector_size, args.index_buffer)
else:
    index = DenseFlatIndexer(vector_size, args.index_buffer)

retriever = DenseRetriever(encoder, args.batch_size, tensorizer, index)

# index all passages
ctx_files_pattern = args.encoded_ctx_file
input_paths = glob.glob(ctx_files_pattern)

index_path = "_".join(input_paths[0].split("_")[:-1])
if args.save_or_load_index and (os.path.exists(index_path) or os.path.exists(index_path + ".index.dpr")):
    retriever.index.deserialize_from(index_path)
else:
    logger.info('Reading all passages data from files: %s', input_paths)
    retriever.index.index_data(input_paths)
    if args.save_or_load_index:
        retriever.index.serialize(index_path)
# get top k results

with open("data_preprocessed/test_data.json","r",encoding="utf-8") as fr:
    dev_data = json.load(fr)
true_lb = 0
total = 0
i = 0
for item in tqdm(dev_data["items"]):
    i += 1
    question = preprocess(item["question"])
    relevant = []
    for article in item["relevant_articles"]:
        relevant.append(article["law_id"]+"_"+article["article_id"])
    questions_tensor = retriever.generate_question_vectors([question])
    top_ids_and_scores = retriever.get_top_docs(questions_tensor.numpy(), args.n_docs)
    top_ids = top_ids_and_scores[0][0]
    # if i == 640: 
    #     print("question is: ",question)
    #     print("article relevant: ", relevant)
    #     print("top docs retrieval: ")
    #     print(top_ids)
    for id_ in relevant:
        if(id_ in top_ids):
            true_lb += 1
        total += 1

# print(true_lb)
# print(total)
print(true_lb/total)


