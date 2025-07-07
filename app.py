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
version = os.getenv("VERSION", "demo")  # バージョン番号を取得

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

# 対象顧客の入力
st.subheader("対象顧客情報")
st.markdown(
    """
    以下の情報を入力してください。これらの情報は、過去の事例と比較してレコメンドを行うために使用されます。
    """
)
# 対象顧客の情報を入力するためのテキスト入力フィールド
target_info = {
    "企業名": st.text_input("企業名", placeholder="例: 株式会社サンプル"),
    "従業員数": st.number_input("従業員数", min_value=1, max_value=100000, value=1000, step=1),
    "資本金(百万円)": st.number_input("資本金(百万円)", min_value=0, max_value=10000, value=100, step=1),
    "業種": st.text_input("業種", placeholder="例: IT, 製造"),
    "備考": st.text_area("備考", placeholder="例: 特記事項やニーズなど"),
}
# 入力された情報をDataFrameに変換
target_df = pd.DataFrame([target_info])

# 検索ボタン
button_pushed = st.button("過去事例検索")

# 過去事例の表示
st.subheader("過去事例")
archive_df = pd.read_csv("./db/companies_archive.csv")
st.dataframe(archive_df, use_container_width=True, hide_index=True)

# レコメンド
if button_pushed:
    # 過去事例のベクトルを取得
    archive_encoder = Encoder(archive_df)
    archive_vecs = archive_encoder.encode()

    # 対象顧客のベクトルを取得
    target_encoder = Encoder(target_df)
    target_vec = target_encoder.encode().flatten()

    # 入力値を取得
    lambda_d = 10 ** log_lambda_d
    lambda_c = 10 ** log_lamnda_c

    # OptimizerConfigのインスタンスを作成
    config = OptimizerConfig(mode=mode, K=K, L=L, lambda_d=lambda_d, lambda_c=lambda_c)

    # Optimizerのインスタンスを作成
    optimizer = Optimizer(archive_vecs, target_vec, config)

    # 最適化を実行
    best_sample = optimizer.optimize()
    x_value: dict[tuple[int, int], float] = best_sample.extract_decision_variables("x")
    # インデックスがint型の場合に変換
    nonzero_keys = [k if isinstance(k, int) else k[0] for k, v in x_value.items() if v > 0.5]

    # 選ばれた行だけ表示
    selected_df = archive_df.loc[nonzero_keys]
    st.subheader("類似する過去事例")
    st.dataframe(selected_df, use_container_width=True, hide_index=True)
    print("最適化結果:", nonzero_keys)