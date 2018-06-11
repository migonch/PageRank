import pandas as pd
import numpy as np

import networkx as nx
from scipy import sparse, stats
from scipy.optimize import minimize_scalar
import random
from sklearn.preprocessing import normalize


class PageRank():
    def __init__(self, data=None):
        if type(data) == nx.classes.digraph.DiGraph:
            self.graph = data
            self.size = nx.number_of_nodes(self.graph)
            self.adj_matrix = nx.adjacency_matrix(self.graph, 
                                                  nodelist=range(1, self.size + 1))
        elif type(data) == sparse.csr.csr_matrix or sparse.csc.csc_matrix:
            self.adj_matrix = data
            self.size = data.shape[0]
            self.graph = nx.DiGraph(incoming_graph_data=self.adj_matrix)
            nx.relabel_nodes(self.graph, dict(zip(range(self.size), range(1, self.size + 1))), 
                             copy=False)
        else:
            raise TypeError('Type of data is unsuitable')
            

    def set_custom_attributes(self):
        out_degrees = np.array(list(self.graph.out_degree(range(1, self.size + 1))))[:, 1]
        self.null_column_indeces = np.zeros(self.size)
        self.null_column_indeces[np.where(out_degrees == 0)] = 1
        self.P_matrix = normalize(self.adj_matrix, norm='l1').T
        self.dataframe_of_ranks = pd.DataFrame(index=range(1, self.size + 1))
        self.dataframe_of_ranks.index.name = 'node_id'
        return self
        
    
    def power_method_iteration(self, x, alpha=0.85):
        res = alpha * self.P_matrix.dot(x)
        res += alpha / self.size * np.dot(self.null_column_indeces, x)
        res += (1 - alpha) / self.size * np.sum(x)
        return res
    
    
    def power_method(self, x0, n_iter=200, alpha=0.85, return_residuals=False):
        assert len(x0) == self.size and abs(np.linalg.norm(x0, ord=1) - 1.0) < 1e-7
        x_cur = x0
        residuals = []
        for i in range(n_iter):
            x_next = self.power_method_iteration(x_cur, alpha=alpha)
            residuals.append(np.linalg.norm(x_next - x_cur, ord=1))
            x_cur = x_next
        self.pm_ranks = x_cur
        self.dataframe_of_ranks['Power Method'] = self.pm_ranks 
        if not return_residuals: residuals = None
        return x_cur, residuals
    
    
    def mcmc(self, n_iter=int(1e6), alpha=0.85):
        self.mcmc_ranks = np.zeros(self.size)
        self.mcmc_deviations = []
        list_of_nodes = list(self.graph.nodes())
        rv = stats.bernoulli(1 - alpha)
        # sequence of 0 and 1
        # 1's numbers = numbers of steps with teleport
        teleports = rv.rvs(size=n_iter)
        # start to surf
        node = random.choice(list_of_nodes)
        for i in range(len(teleports)):
            if teleports[i] or list(self.graph.successors(node)) == []:
                node = random.choice(list_of_nodes)
            else:
                node = random.choice(list(self.graph.successors(node)))
            self.mcmc_ranks[node - 1] += 1
            if (i + 1) % 100 == 0:
                self.mcmc_deviations.append(np.linalg.norm(self.mcmc_ranks / (i + 1) - self.pm_ranks))
        self.mcmc_ranks /= n_iter
        self.dataframe_of_ranks['MCMC'] = self.mcmc_ranks
        return self
    
    
    def frank_wolfe_iteration(self, x, k, step='line_search', alpha=0.85):
        Ax = alpha * (self.P_matrix.dot(x) + np.dot(self.null_column_indeces, x) / self.size) 
        Ax += (1 - alpha) / self.size - x # Ax
        grad = alpha * self.P_matrix.T.dot(Ax) - Ax # A^TAx
        y = np.zeros(self.size)
        y[grad.argmin()] = 1 # y_k
        Ay = alpha * (self.P_matrix.dot(y) + 
                      self.null_column_indeces[grad.argmin()] / self.size) 
        Ay += (1 - alpha) / self.size - y
        if step == 'line_search':
            gamma = minimize_scalar(lambda gam: np.linalg.norm((1 - gam) * Ax + gam * Ay) ** 2 / 2, 
                                    bounds=(0, 1)).x
        elif step == 'nonadaptive':
            gamma = 2 / (k + 2)
        else:
            raise ValueError("step must be 'line_search' or 'nonadaptive'")
        return x + gamma * (y - x), np.linalg.norm(Ax) ** 2 / 2 # x_{k+1}
    
    
    def frank_wolfe(self, x0, n_iter, step='line_search', alpha=0.85, 
                    return_targets=False, return_residuals=False):
        assert len(x0) == self.size and abs(np.linalg.norm(x0, ord=1) - 1.0) < 1e-7
        x_cur = x0
        targets = []
        residuals = []
        for i in range(n_iter):
            x_next, target_cur = self.frank_wolfe_iteration(x_cur, i, step=step, alpha=alpha)
            targets.append(target_cur)
            residuals.append(np.linalg.norm(x_next - x_cur, ord=1))
            x_cur = x_next
        self.fw_ranks = x_cur
        self.dataframe_of_ranks['Frank-Wolfe'] = self.fw_ranks
        if not return_targets: targets = None
        if not return_residuals: residuals = None
        return x_cur, targets, residuals
    
    
    def show_ranks(self, n_first=5, sort_by='Power Method'):
        return self.dataframe_of_ranks.sort_values(by=sort_by, 
                                                   ascending=False).head(n_first)