import os
from PIL import Image, ImageDraw
import pandas as pd

# CSVファイルのパス
CSV_PATH = os.path.join(os.path.dirname(__file__), 'db', 'companies_archive.csv')
IMG_DIR = os.path.join(os.path.dirname(__file__), 'db', 'figs')

# CSVから画像ファイル名リストを取得
companies = pd.read_csv(CSV_PATH)
image_files = companies['画像'].dropna().unique()

for idx, img_file in enumerate(image_files):
    img_path = os.path.join(IMG_DIR, img_file)
    # 既に存在する場合はスキップ
    if os.path.exists(img_path):
        continue
    # 100x100の色付きPNG画像を生成
    img = Image.new('RGB', (100, 100), (100 + idx*3 % 155, 100 + idx*7 % 155, 200 + idx*11 % 55))
    draw = ImageDraw.Draw(img)
    draw.text((10, 40), str(idx+1), fill=(255,255,255))
    img.save(img_path, format='PNG')
print('companies_archive.csvに記載された全企業分のダミー画像を生成しました')
