import numpy as np
from dataclasses import dataclass

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
        Calculate the cosine similarity between the target vector and each vector in the archive.
        Returns:
            similarities (np.ndarray): Array of cosine similarities.
        """
        # 正規化（ゼロ割防止のため微小値を加算）
        archive_norm = np.linalg.norm(self.archive_vecs, axis=1) + 1e-10
        target_norm = np.linalg.norm(self.target_vec) + 1e-10
        # コサイン類似度計算
        pq_similarities = np.dot(self.archive_vecs, self.target_vec) / (archive_norm * target_norm)

        # archive_vecs同士のコサイン類似度行列を計算
        # (N, D) @ (D, N) = (N, N)
        dot_products = np.dot(self.archive_vecs, self.archive_vecs.T)
        norm_matrix = np.outer(archive_norm, archive_norm)
        pp_similarities = dot_products / (norm_matrix + 1e-10)
                
        return pq_similarities, pp_similarities

    def create_qubo(self):
        """
        Create a QUBO (Quadratic Unconstrained Binary Optimization) model based on the configuration.
        This is a placeholder for the actual QUBO creation logic.
        """
        print(f"Creating QUBO with mode: {self.config.mode}, K: {self.config.K}, L: {self.config.L}, "
              f"lambda_d: {self.config.lambda_d}, lambda_c: {self.config.lambda_c}")
        # Actual QUBO creation logic would go here

    def optimize(self):
        """
        Perform the optimization based on the current configuration.
        """
        # Placeholder for optimization logic
        print(f"Optimizing with mode: {self.config.mode}, K: {self.config.K}, L: {self.config.L}, "
              f"lambda_d: {self.config.lambda_d}, lambda_c: {self.config.lambda_c}")
        # Actual optimization logic would go here