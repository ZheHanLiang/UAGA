from logging import getLogger


logger = getLogger()


class Dictionary(object):

    def __init__(self, id2node, node2id, graph):
        assert len(id2node) == len(node2id)
        self.id2node = id2node
        self.node2id = node2id
        self.graph = graph
        self.check_valid()

    def __len__(self):
        """
        Returns the number of nodes in the dictionary.
        """
        return len(self.id2node)

    def __getitem__(self, i):
        """
        Returns the node of the specified index.
        """
        return self.id2node[i]

    def __contains__(self, w):
        """
        Returns whether a node is in the dictionary.
        """
        return w in self.node2id

    def __eq__(self, y):
        """
        Compare the dictionary with another one.
        """
        self.check_valid()
        y.check_valid()
        if len(self.id2node) != len(y):
            return False
        return self.graph == y.graph and all(self.id2node[i] == y[i] for i in range(len(y)))

    def check_valid(self):
        """
        Check that the dictionary is valid.
        """
        assert len(self.id2node) == len(self.node2id)
        for i in range(len(self.id2node)):
            assert self.node2id[self.id2node[i]] == i

    def index(self, node):
        """
        Returns the index of the specified node.
        """
        return self.node2id[node]

