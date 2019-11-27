import os
import io
from logging import getLogger
import numpy as np
import torch

from ..utils import get_nn_avg_dist


logger = getLogger()



def load_dictionary(path, node2id1, node2id2):
    """
    Return a torch tensor of size (n, 2) where n is the size of the
    loader dictionary.
    """
    assert os.path.isfile(path)

    pairs = []
    not_found = 0
    not_found1 = 0
    not_found2 = 0

    with io.open(path, 'r', encoding='utf-8') as f:
        for _, line in enumerate(f):
            assert line == line.lower()
            node1, node2 = line.rstrip().split()
            if node1 in node2id1 and node2 in node2id2:
                pairs.append((node1, node2))
            else:
                not_found += 1
                not_found1 += int(node1 not in node2id1)
                not_found2 += int(node2 not in node2id2)

    logger.info("Found %i pairs of nodes in the dictionary (%i unique). "
                "%i other pairs contained at least one unknown node "
                "(%i in graph1, %i in graph2)"
                % (len(pairs), len(set([x for x, _ in pairs])),
                   not_found, not_found1, not_found2))

    pairs = sorted(pairs, key=lambda x: node2id1[x[0]])
    dico = torch.LongTensor(len(pairs), 2)
    for i, (node1, node2) in enumerate(pairs):
        dico[i, 0] = node2id1[node1]
        dico[i, 1] = node2id2[node2]

    return dico


def get_node_matching_accuracy(graph1, node2id1, emb1, graph2, node2id2, emb2, method, dico_eval):
    """
    Given source and target node embeddings, and a dictionary,
    evaluate the matching accuracy using the precision@k.
    """
    path = dico_eval
    dico = load_dictionary(path, node2id1, node2id2)
    dico = dico.cuda() if emb1.is_cuda else dico

    assert dico[:, 0].max() < emb1.size(0)
    assert dico[:, 1].max() < emb2.size(0)

    # normalize node embeddings
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)

    # nearest neighbors
    if method == 'nn':
        query = emb1[dico[:, 0]]
        scores = query.mm(emb2.transpose(0, 1))

    # CGSS dissimilarity measure
    elif method.startswith('cgss_knn_'):
        # average distances to k nearest neighbors
        knn = method[len('cgss_knn_'):]
        assert knn.isdigit()
        knn = int(knn)
        average_dist1 = get_nn_avg_dist(emb2, emb1, knn)
        average_dist2 = get_nn_avg_dist(emb1, emb2, knn)
        average_dist1 = torch.from_numpy(average_dist1).type_as(emb1)
        average_dist2 = torch.from_numpy(average_dist2).type_as(emb2)
        # queries / scores
        query = emb1[dico[:, 0]]
        scores = query.mm(emb2.transpose(0, 1))
        scores.mul_(2)
        scores.sub_(average_dist1[dico[:, 0]][:, None])
        scores.sub_(average_dist2[None, :])

    else:
        raise Exception('Unknown method: "%s"' % method)

    results = []
    top_matches = scores.topk(10, 1, True)[1]
    for k in [1, 5, 10]:
        top_k_matches = top_matches[:, :k]
        _matching = (top_k_matches == dico[:, 1][:, None].expand_as(top_k_matches)).sum(1)
        # allow for multiple possible translations
        matching = {}
        for i, src_id in enumerate(dico[:, 0].cpu().numpy()):
            matching[src_id] = min(matching.get(src_id, 0) + _matching[i], 1)
        # evaluate precision@k
        precision_at_k = 100 * np.mean(list(matching.values()))
        logger.info("%i source nodes - %s - Precision at k = %i: %f" %
                    (len(matching), method, k, precision_at_k))
        results.append(('precision_at_%i' % k, precision_at_k))

    return results
