import os
import io
import re
import sys
import torch
import pickle
import random
import inspect
import argparse
import subprocess
import numpy as np
from torch import optim
from logging import getLogger

from .logger import create_logger
from .dictionary import Dictionary


MAIN_DUMP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'dumped')

logger = getLogger()


# load Faiss if available (dramatically accelerates the nearest neighbor search)
try:
    import faiss
    FAISS_AVAILABLE = True
    if not hasattr(faiss, 'StandardGpuResources'):
        sys.stderr.write("Impossible to import Faiss-GPU. "
                         "Switching to FAISS-CPU, "
                         "this will be slower.\n\n")

except ImportError:
    sys.stderr.write("Impossible to import Faiss library!! "
                     "Switching to standard nearest neighbors search implementation, "
                     "this will be significantly slower.\n\n")
    FAISS_AVAILABLE = False


def initialize_exp(params):
    """
    Initialize experiment.
    """
    # initialization
    if getattr(params, 'seed', -1) >= 0:
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
        if params.cuda:
            torch.cuda.manual_seed(params.seed)

    # dump parameters
    params.exp_path = get_exp_path(params)
    with io.open(os.path.join(params.exp_path, 'params.pkl'), 'wb') as f:
        pickle.dump(params, f)

    # create logger
    logger = create_logger(os.path.join(params.exp_path, 'train.log'), vb=params.verbose)
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v in sorted(dict(vars(params)).items())))
    logger.info('The experiment will be stored in %s' % params.exp_path)
    return logger


def get_idf(europarl, src_graph, tgt_graph, n_idf):
    """
    Compute IDF values.
    """
    idf = {src_graph: {}, tgt_graph: {}}
    k = 0
    for graph in idf:
        start_idx = 200000 + k * n_idf
        end_idx = 200000 + (k + 1) * n_idf
        for sent in europarl[graph][start_idx:end_idx]:
            for node in set(sent):
                idf[graph][node] = idf[graph].get(node, 0) + 1
        n_doc = len(europarl[graph][start_idx:end_idx])
        for node in idf[graph]:
            idf[graph][node] = max(1, np.log10(n_doc / (idf[graph][node])))
        k += 1
    return idf


def get_nn_avg_dist(emb, query, knn):
    """
    Compute the average distance of the `knn` nearest neighbors
    for a given set of embeddings and queries.
    Use Faiss if available.
    """
    if FAISS_AVAILABLE:
        emb = emb.cpu().numpy()
        query = query.cpu().numpy()
        if hasattr(faiss, 'StandardGpuResources'):
            # gpu mode
            res = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = 0
            index = faiss.GpuIndexFlatIP(res, emb.shape[1], config)
        else:
            # cpu mode
            index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        distances, _ = index.search(query, knn)
        return distances.mean(1)
    else:
        bs = 1024
        all_distances = []
        emb = emb.transpose(0, 1).contiguous()
        for i in range(0, query.shape[0], bs):
            distances = query[i:i + bs].mm(emb)
            best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
            all_distances.append(best_distances.mean(1).cpu())
        all_distances = torch.cat(all_distances)
        return all_distances.numpy()


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in ['off', 'false', '0']:
        return False
    if s.lower() in ['on', 'true', '1']:
        return True
    raise argparse.ArgumentTypeError("invalid value for a boolean flag (0 or 1)")


def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params


def get_exp_path(params):
    """
    Create a directory to store the experiment.
    """
    # create the main dump path if it does not exist
    exp_folder = MAIN_DUMP_PATH if params.exp_path == '' else params.exp_path
    if not os.path.exists(exp_folder):
        subprocess.Popen("mkdir %s" % exp_folder, shell=True).wait()
    assert params.exp_name != ''
    exp_folder = os.path.join(exp_folder, params.exp_name)
    if not os.path.exists(exp_folder):
        subprocess.Popen("mkdir %s" % exp_folder, shell=True).wait()
    if params.exp_id == '':
        chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
        while True:
            exp_id = ''.join(random.choice(chars) for _ in range(10))
            exp_path = os.path.join(exp_folder, exp_id)
            if not os.path.isdir(exp_path):
                break
    else:
        exp_path = os.path.join(exp_folder, params.exp_id)
        assert not os.path.isdir(exp_path), exp_path
    # create the dump folder
    if not os.path.isdir(exp_path):
        subprocess.Popen("mkdir %s" % exp_path, shell=True).wait()
    return exp_path


def clip_parameters(model, clip):
    """
    Clip model weights.
    """
    if clip > 0:
        for x in model.parameters():
            x.data.clamp_(-clip, clip)


def read_txt_embeddings(params, source, full_vocab):
    """
    Reload pretrained embeddings from a text file.
    """
    node2id = {}
    vectors = []

    # load pretrained embeddings
    graph = params.src_graph if source else params.tgt_graph
    emb_path = params.src_emb if source else params.tgt_emb
    _emb_dim_file = params.emb_dim
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0:
                split = line.split()
                assert len(split) == 2
                assert _emb_dim_file == int(split[1])
            else:
                node, vect = line.rstrip().split(' ', 1)
                vect = np.fromstring(vect, dtype=float, sep=' ')
                if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                    vect[0] = 0.01
                node2id[node] = len(node2id)
                vectors.append(vect)

    assert len(node2id) == len(vectors)
    logger.info("Loaded %i pre-trained node embeddings." % len(vectors))

    # compute new vocabulary / embeddings
    id2node = {v: k for k, v in node2id.items()}
    dico = Dictionary(id2node, node2id, graph)
    # embeddings = np.concatenate(vectors, 0)
    embeddings = np.mat(vectors)
    embeddings = torch.from_numpy(embeddings).float()
    embeddings = embeddings.cuda() if (params.cuda and not full_vocab) else embeddings

    assert embeddings.size() == (len(dico), params.emb_dim)
    return dico, embeddings


def load_embeddings(params, source, full_node=True):
    """
    Reload pretrained embeddings.
    """
    assert type(source) is bool and type(full_node) is bool
    emb_path = params.src_emb if source else params.tgt_emb
    return read_txt_embeddings(params, source, full_node)


def normalize_embeddings(emb, types, mean=None):
    """
    Normalize embeddings by their norms / recenter them.
    """
    for t in types.split(','):
        if t == '':
            continue
        if t == 'center':
            if mean is None:
                mean = emb.mean(0, keepdim=True)
            emb.sub_(mean.expand_as(emb))
        elif t == 'renorm':
            emb.div_(emb.norm(2, 1, keepdim=True).expand_as(emb))
        else:
            raise Exception('Unknown normalization type: "%s"' % t)
    return mean.cpu() if mean is not None else None


def export_embeddings(src_emb, tgt_emb, params):
    """
    Export embeddings to a text.
    """
    assert params.export is "txt"

    # text file
    src_path = os.path.join(params.exp_path, 'vectors-%s.txt' % params.src_graph)
    tgt_path = os.path.join(params.exp_path, 'vectors-%s.txt' % params.tgt_graph)
    # source embeddings
    logger.info('Writing source embeddings to %s ...' % src_path)
    with io.open(src_path, 'w', encoding='utf-8') as f:
        f.write(u"%i %i\n" % src_emb.size())
        for i in range(len(params.src_dico)):
            f.write(u"%s %s\n" % (params.src_dico[i], " ".join('%.5f' % x for x in src_emb[i])))
    # target embeddings
    logger.info('Writing target embeddings to %s ...' % tgt_path)
    with io.open(tgt_path, 'w', encoding='utf-8') as f:
        f.write(u"%i %i\n" % tgt_emb.size())
        for i in range(len(params.tgt_dico)):
            f.write(u"%s %s\n" % (params.tgt_dico[i], " ".join('%.5f' % x for x in tgt_emb[i])))
