from logging import getLogger
from copy import deepcopy
import numpy as np
from torch.autograd import Variable
from torch import Tensor as torch_tensor
from . import get_nodesim_scores, get_crossgraph_nodesim_scores ,get_two_subgraph_scores,get_two_graph_scores
from . import get_node_matching_accuracy
from ..dico_builder import get_candidates, build_dictionary
from src.utils import get_idf


logger = getLogger()


class Evaluator(object):

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.src_emb = trainer.src_emb
        self.tgt_emb = trainer.tgt_emb
        self.src_dico = trainer.src_dico
        self.tgt_dico = trainer.tgt_dico
        self.mapping = trainer.mapping
        self.discriminator = trainer.discriminator
        self.params = trainer.params

    def simplegraph_nodesim(self, to_log):
        """
        Evaluation on simple-graph node similarity.
        """
        src_ws_scores = get_nodesim_scores(
            self.src_dico.graph, self.src_dico.node2id,
            self.mapping(self.src_emb.weight).data.cpu().numpy()
        )
        tgt_ws_scores = get_nodesim_scores(
            self.tgt_dico.graph, self.tgt_dico.node2id,
            self.tgt_emb.weight.data.cpu().numpy()
        ) if self.params.tgt_graph else None
        if src_ws_scores is not None:
            src_ws_simplegrpah_scores = np.mean(list(src_ws_scores.values()))
            logger.info("simplegrpah source node similarity score average: %.5f" % src_ws_simplegrpah_scores)
            to_log['src_ws_simplegrpah_scores'] = src_ws_simplegrpah_scores
            to_log.update({'src_' + k: v for k, v in src_ws_scores.items()})
        if tgt_ws_scores is not None:
            tgt_ws_simplegrpah_scores = np.mean(list(tgt_ws_scores.values()))
            logger.info("simplegrpah target node similarity score average: %.5f" % tgt_ws_simplegrpah_scores)
            to_log['tgt_ws_simplegrpah_scores'] = tgt_ws_simplegrpah_scores
            to_log.update({'tgt_' + k: v for k, v in tgt_ws_scores.items()})
        if src_ws_scores is not None and tgt_ws_scores is not None:
            ws_simplegrpah_scores = (src_ws_simplegrpah_scores + tgt_ws_simplegrpah_scores) / 2
            logger.info("simplegrpah node similarity score average: %.5f" % ws_simplegrpah_scores)
            to_log['ws_simplegrpah_scores'] = ws_simplegrpah_scores


    def crossgraph_nodesim(self, to_log):
        """
        Evaluation on cross-graph node similarity.
        """
        src_emb = self.mapping(self.src_emb.weight).data.cpu().numpy()
        tgt_emb = self.tgt_emb.weight.data.cpu().numpy()
        # cross-graph nodesim evaluation
        src_tgt_ws_scores = get_crossgraph_nodesim_scores(
            self.src_dico.graph, self.src_dico.node2id, src_emb,
            self.tgt_dico.graph, self.tgt_dico.node2id, tgt_emb,
        )
        if src_tgt_ws_scores is None:
            return
        ws_crossgraph_scores = np.mean(list(src_tgt_ws_scores.values()))
        logger.info("Cross-graph node similarity score average: %.5f" % ws_crossgraph_scores)
        to_log['ws_crossgraph_scores'] = ws_crossgraph_scores
        to_log.update({'src_tgt_' + k: v for k, v in src_tgt_ws_scores.items()})

    def graph_cosine_accuracy(self, to_log):
        """
        Evaluation on two subgraph node accuracy.
        """
        src_emb = self.mapping(self.src_emb.weight).data.cpu().numpy()
        tgt_emb = self.tgt_emb.weight.data.cpu().numpy()
        # cross-graph nodesim evaluation
        accuracy1,accuracy5,accuracy10 = get_two_subgraph_scores(
            self.src_dico.node2id, src_emb,
            self.tgt_dico.node2id, tgt_emb,
        )       
        if accuracy1 is None:
            return
        logger.info("synthetic data accuracy1: %.5f,accuracy5: %.5f,accuracy10: %.5f" % (accuracy1,accuracy5,accuracy10))
        to_log['synthetic data accuracy'] = (accuracy1,accuracy5,accuracy10)

    def node_matcing(self, to_log):
        """
        Evaluation on node matching.
        """
        # mapped node embeddings
        src_emb = self.mapping(self.src_emb.weight).data
        tgt_emb = self.tgt_emb.weight.data

        for method in ['nn', 'cgss_knn_10']:
            results = get_node_matching_accuracy(
                self.src_dico.graph, self.src_dico.node2id, src_emb,
                self.tgt_dico.graph, self.tgt_dico.node2id, tgt_emb,
                method=method,
                dico_eval=self.params.dico_eval
            )
            to_log.update([('%s-%s' % (k, method), v) for k, v in results])

    def dist_mean_cosine(self, to_log):
        """
        Mean-cosine model selection criterion.
        """
        # get normalized embeddings
        src_emb = self.mapping(self.src_emb.weight).data
        tgt_emb = self.tgt_emb.weight.data
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)

        # build dictionary
        for dico_method in ['nn', 'cgss_knn_10']:
            dico_build = 'S2T'
            dico_max_size = 10000
            # temp params / dictionary generation
            _params = deepcopy(self.params)
            _params.dico_method = dico_method
            _params.dico_build = dico_build
            _params.dico_threshold = 0
            _params.dico_min_size = 0
            _params.dico_max_size = dico_max_size
            s2t_candidates = get_candidates(src_emb, tgt_emb, _params)
            t2s_candidates = get_candidates(tgt_emb, src_emb, _params)
            dico = build_dictionary(src_emb, tgt_emb, _params, s2t_candidates, t2s_candidates)
            # mean cosine
            if dico is None:
                mean_cosine = -1e9
            else:
                mean_cosine = (src_emb[dico[:dico_max_size, 0]] * tgt_emb[dico[:dico_max_size, 1]]).sum(1).mean()
            mean_cosine = mean_cosine.item() if isinstance(mean_cosine, torch_tensor) else mean_cosine
            logger.info("Mean cosine (%s method, %s build, %i max size): %.5f"
                        % (dico_method, _params.dico_build, dico_max_size, mean_cosine))
            to_log['mean_cosine-%s-%s-%i' % (dico_method, _params.dico_build, dico_max_size)] = mean_cosine

    def all_eval(self, to_log):
        """
        Run all evaluations.
        """
        self.simplegraph_nodesim(to_log)
        self.crossgraph_nodesim(to_log)
        self.node_matcing(to_log)
        self.dist_mean_cosine(to_log)
        
        
    def graph_eval(self, to_log):
        """
        Run all evaluations.
        """
        self.dist_mean_cosine(to_log)
        self.graph_cosine_accuracy(to_log)

    def eval_dis(self, to_log):
        """
        Evaluate discriminator predictions and accuracy.
        """
        bs = 128
        src_preds = []
        tgt_preds = []

        self.discriminator.eval()

        for i in range(0, self.src_emb.num_embeddings, bs):
            emb = Variable(self.src_emb.weight[i:i + bs].data, volatile=True)
            preds = self.discriminator(self.mapping(emb))
            src_preds.extend(preds.data.cpu().tolist())

        for i in range(0, self.tgt_emb.num_embeddings, bs):
            emb = Variable(self.tgt_emb.weight[i:i + bs].data, volatile=True)
            preds = self.discriminator(emb)
            tgt_preds.extend(preds.data.cpu().tolist())

        src_pred = np.mean(src_preds)
        tgt_pred = np.mean(tgt_preds)
        logger.info("Discriminator source / target predictions: %.5f / %.5f"
                    % (src_pred, tgt_pred))

        src_accu = np.mean([x >= 0.5 for x in src_preds])
        tgt_accu = np.mean([x < 0.5 for x in tgt_preds])
        dis_accu = ((src_accu * self.src_emb.num_embeddings + tgt_accu * self.tgt_emb.num_embeddings) /
                    (self.src_emb.num_embeddings + self.tgt_emb.num_embeddings))
        logger.info("Discriminator source / target / global accuracy: %.5f / %.5f / %.5f"
                    % (src_accu, tgt_accu, dis_accu))

        to_log['dis_accu'] = dis_accu
        to_log['dis_src_pred'] = src_pred
        to_log['dis_tgt_pred'] = tgt_pred
