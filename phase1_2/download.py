# download.py
"""
multi_dsprites を Python (requests) でダウンロードする。
進捗表示と整合性チェック付き。
"""

import requests
import os
from tqdm import tqdm

URL = "https://storage.googleapis.com/multi-object-datasets/multi_dsprites/multi_dsprites_colored_on_grayscale.tfrecords"
OUTPUT = "data/multi_dsprites_colored_on_grayscale.tfrecords"

os.makedirs("data", exist_ok=True)

print(f"ダウンロード開始")
print(f"  URL: {URL}")
print(f"  保存先: {OUTPUT}")

# ストリーミングでダウンロード
response = requests.get(URL, stream=True, timeout=60)
response.raise_for_status()

# サーバーが言うサイズを取得
total_size = int(response.headers.get('content-length', 0))
print(f"  期待サイズ: {total_size:,} bytes ({total_size / 1e9:.2f} GB)")

# ダウンロード
chunk_size = 8192 * 16  # 128KB
downloaded = 0

with open(OUTPUT, 'wb') as f:
    with tqdm(total=total_size, unit='B', unit_scale=True, desc='Download') as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                pbar.update(len(chunk))

# 整合性確認
final_size = os.path.getsize(OUTPUT)
print(f"\n結果:")
print(f"  実サイズ: {final_size:,} bytes ({final_size / 1e9:.2f} GB)")
print(f"  期待サイズ: {total_size:,} bytes")
print(f"  一致: {'✅ はい' if final_size == total_size else '❌ いいえ'}")

if final_size != total_size:
    print(f"  ⚠️ {total_size - final_size:,} バイト不足、再実行してください")
else:
    print(f"  ダウンロード成功")