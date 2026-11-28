"""Dense unnormalized spectral scaling and matrix Chebyshev recurrence."""
import numpy as np
import torch


def scaled_laplacian(adjacency):
    graph = np.array(adjacency,dtype=np.float64,copy=True)
    if graph.ndim!=2 or not len(graph) or graph.shape[0]!=graph.shape[1] or not np.isfinite(graph).all() or np.any(graph<0):
        raise ValueError('Graph requires a finite nonnegative square adjacency')
    laplacian = np.diag(graph.sum(1))-graph
    maximum = float(np.linalg.eigvals(laplacian).real.max())
    scaled = 2*laplacian/maximum-np.eye(len(graph)) if maximum>1e-12 else -np.eye(len(graph))
    return torch.tensor(scaled,dtype=torch.float32)


def chebyshev_basis(base,order):
    if type(order) is not int or order<1:
        raise ValueError('Chebyshev order must be a positive integer')
    terms = [torch.eye(len(base),device=base.device,dtype=base.dtype)]
    if order>1:
        terms.append(base)
    for _ in range(2,order):
        terms.append(2*base@terms[-1]-terms[-2])
    return torch.stack(terms)
