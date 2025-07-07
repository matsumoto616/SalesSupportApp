import os
from dotenv import load_dotenv  # .envファイルから環境変数を読み込む
import streamlit as st  # Streamlit本体
import pandas as pd
from optimizer import OptimizerConfig, Optimizer
from encoder import Encoder

# Streamlitのページ設定（ワイド画面をデフォルトに）
st.set_page_config(layout="wide")

# .envファイルから環境変数をロード
load_dotenv()
version = os.getenv("VERSION", "unknown")  # バージョン番号を取得

# サイドバーの設定
sidebar = st.sidebar
sidebar.markdown(
    f"""
# 営業サポートアプリ
### バージョン: {version}
"""
)
sidebar.divider()

# サイドバーの各種入力項目
mode = sidebar.radio("モード選択", ["新規営業", "有料移行サポート", "離脱防止サポート"])
K = sidebar.number_input("レコメンド数", min_value=1, max_value=10, value=5, step=1)
L = sidebar.number_input("参照過去事例数", min_value=1, max_value=10, value=5, step=1)
log_lambda_d = sidebar.number_input("多様性重要度　log(λ_d)", min_value=-5, max_value=5, value=0, step=1)
log_lamnda_c = sidebar.number_input("制約条件重要度　log(λ_c)", min_value=-5, max_value=5, value=0, step=1)
button_pushed = sidebar.button("実行")

# 対象顧客の入力
st.subheader("対象顧客情報")
st.markdown(
    """
    以下の情報を入力してください。これらの情報は、過去の事例と比較してレコメンドを行うために使用されます。
    """
)
# 対象顧客の情報を入力するためのテキスト入力フィールド
customer_info = {
    "業種": st.text_input("業種", placeholder="例: IT, 製造業"),
    "従業員数": st.number_input("従業員数", min_value=1, max_value=100000, value=1000, step=1),
    "資本金": st.number_input("資本金 (万円)", min_value=0, max_value=10000000, value=10000, step=1000),
    "業種": st.text_input("業種", placeholder="例: IT, 製造"),
}
# 入力された情報をDataFrameに変換
customer_df = pd.DataFrame([customer_info])

# 検索ボタン
button_pushed = st.button("過去事例検索")

# 過去事例の表示
st.subheader("過去事例")
df_archive = pd.read_csv("./db/companies_archive.csv")
st.dataframe(df_archive, use_container_width=True, hide_index=True)

# レコメンド
if button_pushed:
    # 過去事例のベクトルを取得
    archive_vecs = df_archive.drop(columns=["id"]).values
    # 対象顧客のベクトルを取得
    target_vec = customer_df.drop(columns=["業種"]).values.flatten()

    # 入力値を取得
    lambda_d = 10 ** log_lambda_d
    lambda_c = 10 ** log_lamnda_c

    # OptimizerConfigのインスタンスを作成
    config = OptimizerConfig(mode=mode, K=K, L=L, lambda_d=lambda_d, lambda_c=lambda_c)

    # Optimizerのインスタンスを作成
    optimizer = Optimizer(archive_vecs, target_vec, config)

    # QUBOを作成
    optimizer.create_qubo()

    # 最適化を実行
    optimizer.optimize()
