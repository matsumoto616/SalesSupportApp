import os
from dotenv import load_dotenv  # .envファイルから環境変数を読み込む
import streamlit as st  # Streamlit本体
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from optimizer import OptimizerConfig, Optimizer
from encoder import Encoder
from data import CompanyDataset
import gdown

def encode_dataset(encoder, dataset, device="cpu", batch_size=32):
    encoder.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_vecs = []
    with torch.no_grad():
        for batch in dataloader:
            # batch: (input_ids, attn_mask, numeric)
            input_ids, attn_mask, numeric = batch
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            numeric = numeric.to(device)
            z, _, = encoder(input_ids, attn_mask, numeric)
            all_vecs.append(z.cpu())
    all_vecs = torch.cat(all_vecs, dim=0)
    return all_vecs.numpy()

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
mode = sidebar.radio("モード選択", ["多様性考慮（2次計画）", "多様性非考慮（線形計画）"])
# K = sidebar.number_input("レコメンド数", min_value=1, max_value=10, value=5, step=1)
L = sidebar.number_input("参照過去事例数", min_value=1, max_value=10, value=5, step=1)
log_lambda_d = sidebar.number_input("多様性重要度　log(λ_d)　※2次計画時のみ", min_value=-5, max_value=5, value=-1, step=1)
log_lamnda_c = sidebar.number_input("制約条件重要度　log(λ_c)　※2次計画時のみ", min_value=0, max_value=5, value=2, step=1)

# 対象顧客の入力
st.subheader("対象顧客情報")
st.markdown(
    """
    以下の情報を入力してください。これらの情報は、過去の事例と比較してレコメンドを行うために使用されます。
    """
)
# 対象顧客の情報を入力するためのテキスト入力フィールド
target_info = {
    "企業名": st.text_input("企業名", value="株式会社アクティブ"),
    "従業員数": st.number_input("従業員数", min_value=1, max_value=10000, value=120, step=1),
    "資本金(百万円)": st.number_input("資本金(百万円)", min_value=0, max_value=10000, value=500, step=1),
    "業種": st.text_input("業種", value="IT"),
    "備考": st.text_area("備考", value="新規事業に積極的"),
}
# 入力された情報をDataFrameに変換
target_df = pd.DataFrame([target_info])

# 検索ボタン
button_pushed = st.button("過去事例検索")


archive_df = pd.read_csv("./db/companies_archive.csv", index_col=0)

# レコメンド
if button_pushed:
    encoder = Encoder()
    try:
        encoder.load_weights(path="./weights/triplet_encoder_best.pth")
    except FileNotFoundError:
        # Google Driveの重みファイルID
        gdrive_url = "https://drive.google.com/uc?id=1a1PPVXInIcNsOJSM7JgEvcRmY7y8FyVD"
        weights_path = "triplet_encoder_best.pth"
        # weightsディレクトリがなければ作成
        # os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        # 重みファイルがなければgdownでダウンロード
        if not os.path.exists(weights_path):
            with st.spinner("重みファイルをダウンロード中..."):
                gdown.download(gdrive_url, weights_path, quiet=False)
        encoder.load_weights(weights_path)
    encoder.eval()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    archive_dataset = CompanyDataset(archive_df, tokenizer)
    target_dataset = CompanyDataset(target_df, tokenizer)

    device = "cpu"
    encoder = encoder.to(device)

    with torch.no_grad():
        archive_vecs = encode_dataset(encoder, archive_dataset, device=device)
        target_vec = encode_dataset(encoder, target_dataset, device=device).flatten()  # 1件のみ

    # 入力値を取得
    lambda_d = 10 ** log_lambda_d
    lambda_c = 10 ** log_lamnda_c

    # OptimizerConfigのインスタンスを作成
    config = OptimizerConfig(mode=mode, L=L, lambda_d=lambda_d, lambda_c=lambda_c)

    # Optimizerのインスタンスを作成
    optimizer = Optimizer(archive_vecs, target_vec, config)

    # 最適化を実行
    nonzero_keys = optimizer.optimize()

    # 選ばれた行だけ表示
    selected_df = archive_df.loc[nonzero_keys]
    st.subheader("参考となる過去事例")
    st.dataframe(selected_df, use_container_width=True, hide_index=True)
    print("最適化結果:", nonzero_keys)

# 過去事例の表示
st.divider()
st.subheader("過去事例一覧")
st.dataframe(archive_df, use_container_width=True, hide_index=True)