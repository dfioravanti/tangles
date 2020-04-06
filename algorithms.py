import numpy as np
import networkx as nx
import scipy.sparse as sp


class FindCuts(object):
    """ Finds partitions of input graph in three steps: 
    1. Coarsening phase
    2. Partitioning phase
    3. Refinement (projection) phase
    """
    def __init__(self, A, merge_fn, partition_fn):
        """ merge_fn is used to coarsen, partition_fn is used to partition
        merge_fn(A): takes as input an adjacency matrix A, returns a set of edges as adjacency
        partition_fn(A, w): takes as input an adjacency matrix A and nodeweights w, returns a partition
                            in form of an indicator vector.
        """
        self.A = A
        self.V = [{i} for i in range(A.shape[0])]
        self.merge_fn = merge_fn
        self.partition_fn = partition_fn
    
    def _coarsen(self, A, V):
        """ Coarsens graph with adjacency matrix A and nodes V by merging edges in edges, which is given
        as adjacency matrix."""
        edges_to_merge = self.merge_fn(A=A, node_weights=np.array([len(v) for v in V]))
        G_edges = nx.from_numpy_array(edges_to_merge)
        connected_components = list(nx.connected_components(G_edges))
        V_coarse = [self._merge_nodes(V, component) for component in connected_components]
        n_coarse = len(V_coarse)
        A_coarse = np.zeros((n_coarse, n_coarse))
        # TODO: vectorize
        for i in range(n_coarse):
            for j in range(i+1, n_coarse):
                for v in connected_components[i]:
                    for w in connected_components[j]:
                        A_coarse[i, j] += A[v, w] 
        A_coarse += A_coarse.T
        return A_coarse, V_coarse
    
    def _refine(self, V_coarse, x_coarse):
        """ Takes partition of coarse graph and refines it to partition of input graph."""
        x = np.zeros(self.A.shape[0])
        for i,v in enumerate(V_coarse):
            x[list(v)]= x_coarse[i]
        return x
    
    def _merge_nodes(self, V, component):
        """ Merge the sets of V that are indexed in component."""
        merged_node = set({})
        for idx in component:
            merged_node = merged_node.union(V[idx])
        return merged_node
        
    def __call__(self, N_max=100, verbose=False):
        """ Coarsen graph until it contains at most N_max nodes, then partition, then project it back. """
        A = self.A
        V = self.V
        if verbose:
            print('Start coarsening')
        for _ in range(50):
            if verbose:
                print(f'{len(V)} nodes in graph')
            if len(V) < N_max:
                if verbose:
                    print('Start partitioning')
                x_coarse = self.partition_fn(A=A, w=np.array([len(v) for v in V]))
                if verbose:
                    print('Start refining')
                x = self._refine(V_coarse=V, x_coarse=x_coarse)
                return x
            A, V = self._coarsen(A, V)
        if verbose:
            print('Maximal number of merging iterations exceeded.')
        return
   

def compute_spectral_wcut(A, w=None, K=2, max_iter=50):
    """ Solves weighted ratio cut as a relaxed trace maximization problem with Yu and Shi postprocessing."""
    N = A.shape[0]
    if w is None:
        w = np.ones(N)
    L = np.diag(np.sum(A, axis=1)) - A
    
    # Solve eigenvalue problem
    w_inv_sqrt = w**(-1/2)
    s, V = sp.linalg.eigsh(np.eye(L.shape[0]) - w_inv_sqrt.reshape(-1, 1) * L * w_inv_sqrt, k=K, which='LA')
    s, V = s[::-1], V[:,::-1]
    Z = w_inv_sqrt.reshape(-1, 1) * V
#     assert np.isclose((np.diag(1/w) @ (np.diag(w)-L)) @ Z, Z @ np.diag(s)).all(), 'Eigendecomposition failed'
    
    # Normalize solution
    X_tilde = 1 / np.linalg.norm(Z, axis=1, keepdims=True) * Z
#     assert np.isclose(np.linalg.norm(X_tilde, axis=1), np.ones(X_tilde.shape[0])).all(), 'Normalization failed'
    
    # Initialize R
    R = np.zeros((K, K))
    R[:, 0] = X_tilde[np.random.randint(N), :]
    c = np.zeros(N)
    for k in range(1, K):
        c += np.abs(X_tilde @ R[:, k-1])
        R[:, k] = X_tilde[np.argmin(c), :]
    
    # Update X and R iteratively until convergence
    psi_old = 0
    for _ in range(max_iter):
        # Update X
        arg_maxs = np.argmax(X_tilde @ R, axis=1)
        X = np.zeros_like(X_tilde)
        X[np.arange(N), arg_maxs] = 1
        
        # Update R
        U, omega, U_tilde = np.linalg.svd(X.T @ X_tilde)
#         assert np.isclose((U * omega) @ U_tilde, X.T @ X_tilde).all(), 'SVD failed'
        psi_new = omega.sum()
#         print(np.abs(psi_old-psi_new))
        # Check for convergence
        if np.isclose(psi_new, psi_old):
#             print(f'Converged after {_} steps')
            break
        else:
            psi_old = psi_new
        R = (U @ U_tilde).T
    if K==2:
        X = X[:, 0]
    return X

def temperature_merging(A, node_weights=None, temperature=.6):
    """ Sample each edge with probability proportional to edge weight, scaled by the temperature.
    Returns new edges in form of adjacency matrix"""
    edges_idx = np.triu_indices_from(A, k=1)
    edge_probs = (A[edges_idx] / A[edges_idx].sum())**temperature
    sampled_edges = np.random.binomial(n=1, p=edge_probs)
    is_sampled = (sampled_edges == 1)
    sampled_edges_idx = (edges_idx[0][is_sampled], edges_idx[1][is_sampled])
    edges_to_merge = np.zeros_like(A)
    edges_to_merge[sampled_edges_idx] = 1
    return edges_to_merge + edges_to_merge.T

def _marked_merging(A, criterion_fn, node_weights=None):
    """ Method of paper 'Weighted Graph Cuts without Eigenvectors: A Multilevel Approach' """
    if node_weights is None:
        node_weights = np.ones(A.shape[0])
    unmarked_vertices = list(np.arange(A.shape[0]))
    edges_to_merge = np.zeros_like(A)
    while len(unmarked_vertices)>0:
        v = np.random.choice(unmarked_vertices)
        neighbors_v = np.where(A[v, :] != 0)
        unmarked_neighbors = np.intersect1d(unmarked_vertices, neighbors_v)
        if len(unmarked_neighbors) == 0:
            unmarked_vertices.remove(v)
        else:
            w_idx = criterion_fn(edge_weights=A[v, unmarked_neighbors], 
                                 neighbor_weights=node_weights[unmarked_neighbors],
                                 v_weight=node_weights[v])
            w = unmarked_neighbors[w_idx]
            edges_to_merge[v, w] = 1
            unmarked_vertices.remove(v)
            unmarked_vertices.remove(w)
    return edges_to_merge + edges_to_merge.T

def _random_matching_criterion(edge_weights, neighbor_weights, v_weight):
    """ Compute index of neighbor for the random matching criterion for node v and its neighbors."""
    return np.random.randint(len(edge_weights))

def _heavy_edge_criterion(edge_weights, neighbor_weights, v_weight):
    """ Compute index of neighbor for heavy edge matching criterion for node v and its neighbors."""
    return np.argmax(edge_weights)

def _max_cut_criterion(edge_weights, neighbor_weights, v_weight):
    """ Compute index of neighbor for max-cut criterion for node v and its neighbors."""
    return np.argmax((1 / v_weight + 1 / neighbor_weights) * edge_weights)

def random_matching_merging(A, node_weights=None):
    """ Compute random matching. """
    return _marked_merging(A=A, criterion_fn=_random_matching_criterion, node_weights=node_weights)

def heavy_edge_merging(A, node_weights=None):
    """ Compute heavy edge matching. """
    return _marked_merging(A=A, criterion_fn=_heavy_edge_criterion, node_weights=node_weights)

def max_cut_merging(A, node_weights=None):
    """ Compute max-cut matching. """
    return _marked_merging(A=A, criterion_fn=_max_cut_criterion, node_weights=node_weights)