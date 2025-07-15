import numpy as np
from dataclasses import dataclass
import openjij as oj
import jijmodeling as jm
import ommx_openjij_adapter as oj_ad
import pandas as pd

@dataclass
class OptimizerConfig:
    """
    Configuration for the optimizer.
    """
    mode: str = "新規営業"  # Available modes: "新規営業", "有料移行サポート", "離脱防止サポート"
    K: int = 5
    L: int = 5
    lambda_d: float = 1
    lambda_c: float = 1

    def __init__(self, mode: str = "新規営業", K: int = 5, L: int = 5, lambda_d: float = 1, lambda_c: float = 1):
        self.mode = mode
        self.K = K
        self.L = L
        self.lambda_d = lambda_d
        self.lambda_c = lambda_c

    def __post_init__(self):
        """
        Validate the configuration parameters.
        """
        if self.mode not in ["新規営業", "有料移行サポート", "離脱防止サポート"]:
            raise ValueError("Invalid mode. Choose from '新規営業', '有料移行サポート', '離脱防止サポート'.")
        if self.K <= 0:
            raise ValueError("K must be a positive integer.")
        if self.L <= 0:
            raise ValueError("L must be a positive integer.")
        if self.lambda_d < 0:
            raise ValueError("lambda_d must be non-negative.")
        if self.lambda_c < 0:
            raise ValueError("lambda_c must be non-negative.")
        
class Optimizer:
    """
    Optimizer class to handle the optimization logic based on the configuration.
    """
    def __init__(self, archive_vecs: np.ndarray, target_vec: np.ndarray, config: OptimizerConfig):
        self.archive_vecs = archive_vecs
        self.target_vec = target_vec
        self.config = config
        self.__post_init__()

    def __post_init__(self):
        """
        Validate the input vectors and configuration.
        """
        if not isinstance(self.archive_vecs, np.ndarray):
            raise ValueError("archive_vecs must be a numpy array.")
        if not isinstance(self.target_vec, np.ndarray):
            raise ValueError("target_vec must be a numpy array.")
        if self.archive_vecs.ndim != 2:
            raise ValueError("archive_vecs must be a 2D numpy array.")
        if self.target_vec.ndim != 1:
            raise ValueError("target_vec must be a 1D numpy array.")
        if self.archive_vecs.shape[1] != self.target_vec.shape[0]:
            raise ValueError("The number of features in archive_vecs must match target_vec.")
        
    def calculate_similarity(self):
        """
        Calculate the 2-norm (Euclidean distance) similarity between the target vector and each vector in the archive.
        Returns:
            similarities (np.ndarray): Array of negative Euclidean distances (similarity).
        """
        # 2ノルム距離（Euclidean distance）
        pq_distances = np.linalg.norm(self.archive_vecs - self.target_vec, axis=1)
        # 類似度として負の距離を使う（距離が小さいほど類似度が高い）
        pq_similarities = -pq_distances

        # archive_vecs同士の2ノルム距離行列
        # (N, D) - (N, D) の全組み合わせ
        N = self.archive_vecs.shape[0]
        pp_distances = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                pp_distances[i, j] = np.linalg.norm(self.archive_vecs[i] - self.archive_vecs[j])
        pp_similarities = -pp_distances

        return pq_similarities, pp_similarities

    def create_qubo(self, pq_similarities, pp_similarities):
        """
        Create a QUBO (Quadratic Unconstrained Binary Optimization) model based on the configuration.
        This is a placeholder for the actual QUBO creation logic.
        """
        a = jm.Placeholder("a", shape=(self.archive_vecs.shape[0],))
        s_target = jm.Placeholder("s_target", shape=(self.archive_vecs.shape[0],))
        s_archive = jm.Placeholder("s_archive", shape=(self.archive_vecs.shape[0], self.archive_vecs.shape[0]))
        lambda_d = jm.Placeholder("lambda_d", shape=())
        L = jm.Placeholder("L", shape=())
        x = jm.BinaryVar("x", shape=(self.archive_vecs.shape[0],))
        p = jm.Element("p", belong_to=(0, pp_similarities.shape[0] - 1))
        p1 = jm.Element("p1", belong_to=(0, pp_similarities.shape[0] - 1))
        p2 = jm.Element("p2", belong_to=(0, pp_similarities.shape[0] - 1))

        problem = jm.Problem("SalesSupportOptimization")
        problem += - jm.sum(p, a[p] * s_target[p] * x[p]) + lambda_d * jm.sum([p1,p2], s_archive[p1,p2] * x[p1] * x[p2])
        problem += jm.Constraint("num_selected", jm.sum(p, x[p]) == L)

        instance_data = {
            "a": np.ones(self.archive_vecs.shape[0]).tolist(),
            "s_target": pq_similarities.tolist(),
            "s_archive": pp_similarities.tolist(),
            "lambda_d": float(self.config.lambda_d),
            "L": int(self.config.L)
        }
        interpreter = jm.Interpreter(instance_data)
        instance = interpreter.eval_problem(problem)
        qubo, _ = instance.to_qubo(uniform_penalty_weight=self.config.lambda_c)
        adapter = oj_ad.OMMXOpenJijSAAdapter(instance)

        return qubo, adapter

    def optimize(self):
        """
        Perform the optimization based on the current configuration.
        """
        pq_similarities, pp_similarities = self.calculate_similarity()
        qubo, adapter = self.create_qubo(pq_similarities, pp_similarities)
        sampler = oj.SASampler()
        res = sampler.sample_qubo(Q=qubo, num_reads=100, num_sweeps=1000)
        sampleset = adapter.decode_to_sampleset(res)
        best_sample = sampleset.best_feasible_unrelaxed

        return best_sample        
