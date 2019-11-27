## UAGA
This is the code for *[Unsupervised Adversarial Graph Alignment with Graph Embedding](https://arxiv.org/pdf/1907.00544.pdf)*.

Part of iUAGA's code will be updated later.

## Dependencies
* Python 2/3 with [NumPy](http://www.numpy.org/)/[SciPy](https://www.scipy.org/)
* [PyTorch](http://pytorch.org/)
* [Faiss](https://github.com/facebookresearch/faiss) (recommended) for fast nearest neighbor search (CPU or GPU).

## Datasets
We used three public datasets in the paper: 
* [Last.fm](http://lfs.aminer.cn/lab-datasets/multi-sns/lastfm.tar.gz)
* [Flickr](http://lfs.aminer.cn/lab-datasets/multi-sns/livejournal.tar.gz)
* [MySpace](http://lfs.aminer.cn/lab-datasets/multi-sns/myspace.tar.gz)

You can learn more about these datasets from [here](https://www.aminer.cn/cosnet).

### Data format
We utilized *DeepWalk* to learn the source and target graph embeddings in this work, so the format of input data followed the output of deepwalk. You can learn more about the detail from [here](https://github.com/phanein/deepwalk).
