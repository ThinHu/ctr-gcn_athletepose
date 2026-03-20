import numpy as np

# --- 1. Graph Definition (COCO 17-Joint Format) ---

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    return np.dot(A, Dn)

def get_spatial_graph(num_node, self_link, inward, outward):
    I = np.eye(num_node)
    In = np.zeros((num_node, num_node))
    Out = np.zeros((num_node, num_node))
    for i, j in inward:
        In[j, i] = 1
    for i, j in outward:
        Out[j, i] = 1
    A = np.stack((I, normalize_digraph(In), normalize_digraph(Out)))
    return A

class GraphCOCO:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = 17
        self.self_link = [(i, i) for i in range(self.num_node)]
        # COCO inward edges (towards center)
        self.inward = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6), 
                       (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0), 
                       (1, 0), (2, 0), (3, 1), (4, 2)]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode == 'spatial':
            return get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError()