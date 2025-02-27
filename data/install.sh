wget -P data/sift https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/sift1M/sift_base.fvecs
wget -P data/sift https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/sift1M/sift_query.fvecs
wget -P data/sift https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/sift1M/sift_groundtruth.ivecs

wget -P data/glove https://nlp.stanford.edu/data/glove.840B.300d.zip
unzip data/glove/glove.840B.300d.zip -d data/glove/

wget -P data/gist http://ann-benchmarks.com/gist-960-euclidean.hdf5

wget -P data/deep http://ann-benchmarks.com/deep-image-96-angular.hdf5