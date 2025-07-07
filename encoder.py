import pandas as pd
import numpy as np
import hashlib

def simple_text_embedder(text, dim=10):
    """
    テキストをハッシュ化し、dim次元のベクトルに変換（簡易実装）
    """
    h = hashlib.sha256(text.encode('utf-8')).digest()
    arr = np.frombuffer(h, dtype=np.uint8)[:dim]
    return arr.astype(float) / 255.0

def simple_image_embedder(filename, dim=10):
    """
    画像ファイル名をハッシュ化し、dim次元のベクトルに変換（簡易実装）
    """
    if not filename or filename == 'nan':
        return np.zeros(dim)
    h = hashlib.sha256(filename.encode('utf-8')).digest()
    arr = np.frombuffer(h, dtype=np.uint8)[:dim]
    return arr.astype(float) / 255.0

class Encoder:
    """
    企業情報CSVから特徴量ベクトルを生成するクラス。
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        # 必要なカラムが存在するかチェック
        # required_columns = ['企業名', '業種', '資本金(百万円)', '従業員数', '備考', '画像']
        required_columns = ['企業名', '業種', '資本金(百万円)', '従業員数', '備考']
        for col in required_columns:            
            if col not in self.df.columns:
                raise ValueError(f"DataFrameに必要なカラム '{col}' が存在しません。")

    def encode(self, text_embedder=simple_text_embedder, image_embedder=simple_text_embedder):
        """
        企業情報をベクトル化して返す。
        テキスト・画像カラムもエンコード可能。
        Args:
            text_embedder (callable): テキスト→ベクトル変換関数（例: lambda x: np.zeros(10)）
            image_embedder (callable): 画像ファイル名→ベクトル変換関数
        Returns:
            np.ndarray: shape=(企業数, 特徴量数)
        """
        # 数値特徴量
        capital = self.df['資本金(百万円)'].fillna(0).astype(float).values.reshape(-1, 1)
        employees = self.df['従業員数'].fillna(0).astype(float).values.reshape(-1, 1)

        # 業種をone-hotエンコーディング
        # industry_onehot = pd.get_dummies(self.df['業種'], prefix='業種')

        # 企業名・備考（テキスト）
        if text_embedder is not None:
            name_vecs = np.vstack([text_embedder(str(x)) for x in self.df['企業名']])
            industry_vecs = np.vstack([text_embedder(str(x)) for x in self.df['業種']])
            note_vecs = np.vstack([text_embedder(str(x)) for x in self.df['備考']])
        else:
            # デフォルト: ゼロベクトル
            name_vecs = np.zeros((len(self.df), 10))
            note_vecs = np.zeros((len(self.df), 10))

        # 画像（画像ファイル名→ベクトル）
        # if image_embedder is not None:
        #     image_vecs = np.vstack([image_embedder(str(x)) if pd.notnull(x) else np.zeros(10) for x in self.df['画像']])
        # else:
        #     image_vecs = np.zeros((len(self.df), 10))

        # ベクトル結合
        features = np.hstack([
            capital,
            employees,
            industry_vecs,
            name_vecs,
            note_vecs,
            # image_vecs
        ])
        return features