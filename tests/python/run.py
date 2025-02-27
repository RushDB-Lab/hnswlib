import hnswlib
import os
import gc
import numpy as np
import pandas as pd
from time import time
from utils.io import fvecs_read, ivecs_read
from utils.preprocess import normalize
from utils.beam_size import beam_size_gen
from utils.memory import get_memory_usage
from settings import ROUND, TOPK, datasets, degrees

def find_EFS(index, query, NQ, gt, TOPK):
    EFS = []
    W = beam_size_gen(TOPK)  # beam size generator
    prev_recall = 0
    while True:
        EF = next(W)
        EFS.append(EF)
        total_time = 0
        results = []
        index.set_ef(EF)
        for i in range(NQ):
            t1 = time()
            # 使用 hnswlib 的 knn_query 方法，返回 labels 和 distances
            labels, _ = index.knn_query(query[i:i+1], k=TOPK)
            t2 = time()
            results.append(labels[0])  # labels[0] 是最近邻的索引列表
            total_time += t2 - t1

        total_num = NQ * TOPK
        total_correct = 0
        for i in range(NQ):
            res_set = set(results[i])
            for j in range(TOPK):
                if gt[i][j] in res_set:
                    total_correct += 1

        qps = NQ / total_time
        recall = total_correct / total_num * 100

        if recall > 99.8 or (recall - prev_recall) < 0.05 or qps < 10:
            break
        prev_recall = recall
    return EFS

if __name__ == "__main__":
    for DATASET in datasets.keys():
        DISTANCE = datasets[DATASET]

        base = fvecs_read(f"./data/{DATASET}/{DATASET}_base.fvecs")
        query = fvecs_read(f"./data/{DATASET}/{DATASET}_query.fvecs")
        gt = ivecs_read(f"./data/{DATASET}/{DATASET}_groundtruth.ivecs")

        NQ, D = query.shape
        N = base.shape[0]

        # 根据距离度量设置 space 参数并归一化数据
        if DISTANCE == "angular":
            query = normalize(query)
            space = 'cosine'
        else:
            space = 'l2'

        for DEGREE in degrees[DATASET]:
            m1 = get_memory_usage()

            # 加载 HNSW 索引
            index_path = f"./data/{DATASET}/hnsw_{DEGREE}.index"
            index = hnswlib.Index(space=space, dim=D)
            index.load_index(index_path, max_elements=N)

            m2 = get_memory_usage()
            MEMORY = m2 - m1

            EFS = find_EFS(index, query, NQ, gt, TOPK)

            ALL_QPS = []
            ALL_RECALL = []

            for _ in range(ROUND):
                QPS = []
                RECALL = []
                for EF in EFS:
                    total_time = 0
                    results = []
                    index.set_ef(EF)
                    for i in range(NQ):
                        t1 = time()
                        labels, _ = index.knn_query(query[i:i+1], k=TOPK)
                        t2 = time()
                        results.append(labels[0])
                        total_time += t2 - t1

                    total_num = NQ * TOPK
                    total_correct = 0
                    for i in range(NQ):
                        res_set = set(results[i])
                        for j in range(TOPK):
                            if gt[i][j] in res_set:
                                total_correct += 1

                    qps = NQ / total_time
                    recall = total_correct / total_num * 100
                    QPS.append(qps)
                    RECALL.append(recall)

                ALL_QPS.append(QPS)
                ALL_RECALL.append(RECALL)

            ALL_QPS = np.average(np.array(ALL_QPS), axis=0)
            ALL_RECALL = np.average(np.array(ALL_RECALL), axis=0)

            df = pd.DataFrame(
                {
                    "QPS": ALL_QPS,
                    "Recall": ALL_RECALL,
                    "EFS": EFS,
                    "Method": f"hnsw{DEGREE}",
                    "Memory": MEMORY,
                }
            )

            res_dir = f"./results/{DATASET}/hnsw/"
            try:
                os.makedirs(res_dir)
            except OSError as e:
                print(e)
            df.to_csv(res_dir + f"hnsw{DEGREE}_{TOPK}.csv", index=False)

            del index
            del df
            gc.collect()