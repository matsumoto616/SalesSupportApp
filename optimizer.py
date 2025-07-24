import numpy as np
from dataclasses import dataclass
import openjij as oj
import jijmodeling as jm
import ommx_openjij_adapter as oj_ad
import pandas as pd
import pulp

@dataclass
class OptimizerConfig:
    """
    Configuration for the optimizer.
    """

    def __init__(self, mode: str = "多様性考慮（2次計画）", K: int = 5, L: int = 5, lambda_d: float = 1, lambda_c: float = 1):
        self.mode = mode
        self.K = K
        self.L = L
        self.lambda_d = lambda_d
        self.lambda_c = lambda_c

    def __post_init__(self):
        """
        Validate the configuration parameters.
        """
        if self.mode not in ["多様性考慮（2次計画）", "多様性非考慮（線形計画）"]:
            raise ValueError("Invalid mode. Choose from '多様性考慮（2次計画）', '多様性非考慮（線形計画）'")
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
        QUBO行列生成
        """
        Nassets = self.archive_vecs.shape[0]
        QUBO = np.zeros((Nassets, Nassets))
        # ペアワイズ相関（多様性）
        for i in range(Nassets):
            for j in range(Nassets):
                if i != j:
                    QUBO[i][j] += self.config.lambda_d * pp_similarities[i][j]
        # スコア（類似度）
        for i in range(Nassets):
            QUBO[i][i] += -pq_similarities[i]
        # 制約条件
        for i in range(Nassets):
            for j in range(Nassets):
                QUBO[i][j] += self.config.lambda_c
                if i == j:
                    QUBO[i][j] += -self.config.lambda_c * 2 * self.config.L
        return QUBO

    def create_linear_program(self, pq_similarities, pp_similarities): 
        """
        Create a linear program based on the configuration.
        This is a placeholder for the actual linear program creation logic.
        """
        model = pulp.LpProblem("SalesSupportOptimization", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("x", range(self.archive_vecs.shape[0]), cat='Binary')
        model += pulp.lpSum(pq_similarities[i] * x[i] for i in range(self.archive_vecs.shape[0]))
        model += pulp.lpSum(x[i] for i in range(self.archive_vecs.shape[0])) == self.config.L
        
        return model

    def optimize(self):
        """
        Perform the optimization based on the current configuration.
        """
        pq_similarities, pp_similarities = self.calculate_similarity()

        if self.config.mode == "多様性考慮（2次計画）":
            # # まず線形計画で実行可能解を取得
            # model_lp = self.create_linear_program(pq_similarities, pp_similarities)
            # model_lp.solve()
            # x_vars_lp = [v for v in model_lp.variables() if v.name.startswith("x_")]
            # x_vars_lp.sort(key=lambda v: int(v.name.split('_')[1]))
            # initial_x = [int(v.varValue == 1) for v in x_vars_lp]

            # 2次計画の最適化（線形計画の解を初期値に）
            QUBO = self.create_qubo(pq_similarities, pp_similarities)
            sampler = oj.SQASampler()
            # initial_state = {i: v for i, v in enumerate(initial_x)}
            # res = sampler.sample_qubo(Q=QUBO, num_reads=1000, num_sweeps=1000, initial_state=initial_state)
            res = sampler.sample_qubo(Q=QUBO, num_reads=100, num_sweeps=1000)
            best_sample = res.first
            nonzero_keys = [k if isinstance(k, int) else k[0] for k, v in best_sample.sample.items() if v > 0.5]
            return nonzero_keys
        else:
            # 線形計画の最適化
            model = self.create_linear_program(pq_similarities, pp_similarities)
            model.solve()
            x_vars = [v for v in model.variables() if v.name.startswith("x_")]
            x_vars.sort(key=lambda v: int(v.name.split('_')[1]))
            selected_indices = [i for i, v in enumerate(x_vars) if v.varValue == 1]
            return selected_indices
