from utils.io import fvecs_read, ivecs_read
from utils.preprocess import normalize
import hnswlib
from time import time
from settings import EF, datasets, degrees, iter

if __name__ == "__main__":
    for DATASET in datasets.keys():
        DISTANCE = datasets[DATASET]
        ITER = iter[DATASET]

        # 读取数据
        base = fvecs_read(f"./data/{DATASET}/{DATASET}_base.fvecs")
        query = fvecs_read(f"./data/{DATASET}/{DATASET}_query.fvecs")
        gt = ivecs_read(f"./data/{DATASET}/{DATASET}_groundtruth.ivecs")

        N, D = base.shape

        # 根据距离类型处理数据
        if DISTANCE == "angular":
            base = normalize(base)
            query = normalize(query)
            space = 'cosine'  # angular 距离对应余弦距离
        else:
            space = 'l2'  # 默认使用欧几里得距离

        for DEGREE in degrees[DATASET]:
            # 初始化 HNSW 索引
            p = hnswlib.Index(space=space, dim=D)
            p.init_index(max_elements=N, ef_construction=EF, M=DEGREE)

            # 计时并构建索引
            t1 = time()
            p.add_items(base)  # HNSW 在添加数据时自动构建索引
            t2 = time()
            print(f"The construction time for {DATASET}{DEGREE} is {t2-t1}")

            # 保存索引
            index_path = f"./data/{DATASET}/hnsw_{DEGREE}.index"
            p.save_index(index_path)