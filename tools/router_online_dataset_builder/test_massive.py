"""Diagnostic test: verify all three MASSIVE id-ID Parquet splits."""
from datasets import load_dataset

BASE_URL = (
    "https://huggingface.co/datasets/AmazonScience/massive/resolve/"
    "refs%2Fconvert%2Fparquet/id-ID"
)

data_files = {
    "train": f"{BASE_URL}/train/0000.parquet",
    "validation": f"{BASE_URL}/validation/0000.parquet",
    "test": f"{BASE_URL}/test/0000.parquet",
}

print("Memuat MASSIVE Indonesia dari Parquet...")
dataset = load_dataset("parquet", data_files=data_files)

for split, expected in (("train", 11514), ("validation", 2033), ("test", 2974)):
    data = dataset[split]
    print(f"\n=== {split.upper()} ===")
    print("Jumlah:", len(data), "(referensi:", expected, ")")
    print("Contoh:", data[0])
    print("Features:", data.features)
    feature = data.features["intent"]
    print("Intent contoh:", feature.int2str(data[0]["intent"]))
