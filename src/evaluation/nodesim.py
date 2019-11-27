import os
import io
from logging import getLogger
import numpy as np
import torch
from scipy.stats import spearmanr


logger = getLogger()


def get_node_pairs(path):
    """
    Return a list of (node1, node2, score) tuples from a node similarity file.
    """
    assert os.path.isfile(path)
    node_pairs = []
    with io.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            line = line.split()
            # ignore phrases, only consider nodes
            if len(line) != 3:
                assert len(line) > 3
                continue
            node_pairs.append((line[0], line[1], float(line[2])))
    return node_pairs


def get_node_id(node, node2id):
    """
    Get a node ID.
    """
    node_id = node2id.get(node)
    if node_id is None:
        node_id = node2id.get(node.capitalize())
    if node_id is None:
        node_id = node2id.get(node.title())
    return node_id


def get_spearman_rho(node2id1, embeddings1, path,
                     node2id2=None, embeddings2=None):
    """
    Compute simple-graph or cross-graph node similarity score.
    """
    assert not ((node2id2 is None) ^ (embeddings2 is None))
    node2id2 = node2id1 if node2id2 is None else node2id2
    embeddings2 = embeddings1 if embeddings2 is None else embeddings2
    assert len(node2id1) == embeddings1.shape[0]
    assert len(node2id2) == embeddings2.shape[0]
    node_pairs = get_node_pairs(path)
    not_found = 0
    pred = []
    gold = []
    for node1, node2, similarity in node_pairs:
        id1 = get_node_id(node1, node2id1)
        id2 = get_node_id(node2, node2id2)
        if id1 is None or id2 is None:
            not_found += 1
            continue
        u = embeddings1[id1]
        v = embeddings2[id2]
        score = u.dot(v) / (np.linalg.norm(u) * np.linalg.norm(v))
        gold.append(similarity)
        pred.append(score)
    return spearmanr(gold, pred).correlation, len(gold), not_found


def get_nodesim_scores(graph, node2id, embeddings):
    """
    Return simple-graph node similarity scores.
    """
    scores = {}
    separator = "=" * (30 + 1 + 10 + 1 + 13 + 1 + 12)
    pattern = "%30s %10s %13s %12s"
    logger.info(separator)
    logger.info(pattern % ("Dataset", "Found", "Not found", "Rho"))
    logger.info(separator)

    for filename in list(os.listdir(dirpath)):
        if filename.startswith('%s_' % (graph.upper())):
            filepath = os.path.join(dirpath, filename)
            print(filepath)
            coeff, found, not_found = get_spearman_rho(node2id, embeddings, filepath)
            logger.info(pattern % (filename[:-4], str(found), str(not_found), "%.4f" % coeff))
            scores[filename[:-4]] = coeff
    logger.info(separator)

    return scores



def get_crossgraph_nodesim_scores(graph1, node2id1, embeddings1,
                                    graph2, node2id2, embeddings2, lower=True):
    """
    Return cross-graph node similarity scores.
    """
    f1 = os.path.join(SEMEVAL17_EVAL_PATH, '%s-%s-SEMEVAL17.txt' % (graph1, graph2))
    f2 = os.path.join(SEMEVAL17_EVAL_PATH, '%s-%s-SEMEVAL17.txt' % (graph2, graph1))

    if os.path.exists(f1):
        coeff, found, not_found = get_spearman_rho(
            node2id1, embeddings1, f1,
            node2id2, embeddings2
        )
    elif os.path.exists(f2):
        coeff, found, not_found = get_spearman_rho(
            node2id2, embeddings2, f2,
            node2id1, embeddings1
        )

    scores = {}
    separator = "=" * (30 + 1 + 10 + 1 + 13 + 1 + 12)
    pattern = "%30s %10s %13s %12s"
    logger.info(separator)
    logger.info(pattern % ("Dataset", "Found", "Not found", "Rho"))
    logger.info(separator)

    task_name = '%s_%s_SEMEVAL17' % (graph1.upper(), graph2.upper())
    logger.info(pattern % (task_name, str(found), str(not_found), "%.4f" % coeff))
    scores[task_name] = coeff
    if not scores:
        return None
    logger.info(separator)

    return scores


def get_two_subgraph_scores(node2id1, embeddings1,node2id2, embeddings2, lower=True):
    """
    Return cross-graph subgraph similarity scores.
    """
    assert len(node2id1) == embeddings1.shape[0]
    assert len(node2id2) == embeddings2.shape[0]
    top1_count = 0.
    top5_count = 0.
    top10_count = 0.
    all_count = 0.
    test_count = 0
    for i in node2id1:
        id1 = node2id1[i]
        u = embeddings1[id1]
        top10 = []
        for j in node2id2:
            id2 = node2id2[j]
            v = embeddings2[id2]
            score = u.dot(v) / (np.linalg.norm(u) * np.linalg.norm(v))
            #for top10
            if len(top10) != 10:
                top10.append([score,j])
                top10.sort()
            else:
                if top10[0][0] < score:
                    top10[0] = [score,j]
                    top10.sort() 
        for num in range(10):
            if (num > 8) and (top10[num][1] == i):
                top1_count += 1
                top5_count += 1
                top10_count += 1
            elif (num < 9) and (num >= 5) and (top10[num][1] == i):
                top5_count += 1
                top10_count += 1
            elif (num < 5) and (num >= 0) and (top10[num][1] == i):
                top10_count += 1
            else:
                pass
        all_count += 1
        test_count += 1
        if test_count % 1000 == 0:
            print('all number is:%f'%(all_count))
            print('top1: %.5f, top5: %.5f, top10: %.5f'%(top1_count/all_count,top5_count/all_count,top10_count/all_count))
    return (top1_count/all_count,top5_count/all_count,top10_count/all_count)


def get_two_graph_scores(node2id1, embeddings1,node2id2, embeddings2, lower=True):
    """
    Return cross-graph graph similarity scores.
    """

    assert len(node2id1) == embeddings1.shape[0]
    assert len(node2id2) == embeddings2.shape[0]
    top1_count = 0.
    top5_count = 0.
    top10_count = 0.
    all_count = 0.
    test_count = 0
    
    dic_src = {}
    dic_tgt = {}
    with open('./data/lastfm.nodes','r') as f1:
        src_context = f1.readlines()
    for src_i in src_context:
        src_sp = src_i.split('\t')
        dic_src[src_sp[0]] = src_sp[1][:-1]
    
    with open('./data/flickr.nodes','r') as f2:
        tgt_context = f2.readlines()
    for tgt_i in tgt_context:
        tgt_sp = tgt_i.split('\t')
        dic_tgt[tgt_sp[0]] = tgt_sp[1][:-1]
    
    dic_map = {}
    with open('./data/flickr-lastfm.map.raw','r') as f3:
        map_context = f3.readlines()
    for map_i in map_context:
        map_sp = map_i.split(' ')
        dic_map[map_sp[1][:-1]] = map_sp[0]
        
    
    for i in node2id1:
        id1 = node2id1[i]
        u = embeddings1[id1]

        top10 = []
        
        for j in node2id2:
            id2 = node2id2[j]
            v = embeddings2[id2]
            score = u.dot(v) / (np.linalg.norm(u) * np.linalg.norm(v))
            if len(top10) != 10:
                top10.append([score,j])
                top10.sort()
            else:
                if top10[0][0] < score:
                    top10[0] = [score,j]
                    top10.sort() 
        for num in range(10):
            if (i in dic_src) and (dic_src[i] in dic_map):
                print(dic_src[i],dic_tgt[top10[num][1]],dic_map[dic_src[i]])
                if (num > 8) and (dic_tgt[top10[num][1]] == dic_map[dic_src[i]]):
                    top1_count += 1
                    top5_count += 1
                    top10_count += 1
                elif (num < 9) and (num >= 5) and (dic_tgt[top10[num][1]] == dic_map[dic_src[i]]):
                    top5_count += 1
                    top10_count += 1
                elif (num < 5) and (num >= 0) and (dic_tgt[top10[num][1]] == dic_map[dic_src[i]]):
                    top10_count += 1
                else:
                    pass
        all_count += 1
        test_count += 1
        if test_count % 1000 == 0:
            print('all number is:%f'%(all_count))
            print('top1: %.5f, top5: %.5f, top10: %.5f'%(top1_count/all_count,top5_count/all_count,top10_count/all_count))
    return (top1_count/all_count,top5_count/all_count,top10_count/all_count)
