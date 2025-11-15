from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import os

# Tải dataset từ Hugging Face và cache trong thư mục local
dataset = load_dataset("PeterPanTheGenius/CUHK-PEDES", cache_dir="./CUHK-PEDES")

# Tạo thư mục lưu dataset
output_dir = "./CUHK_PEDES_images"
os.makedirs(output_dir, exist_ok=True)

# Duyệt qua các split có sẵn trong dataset (train, test, val)
for split in dataset.keys():
    print(f"Processing split: {split}")
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    for i, sample in enumerate(tqdm(dataset[split], desc=f"Saving {split}")):
        image = sample["image"]   # ảnh
        captions = sample["text"] # danh sách caption (list)

        # Đảm bảo ảnh là RGB
        if image.mode == "RGBA":
            image = image.convert("RGB")

        # Lưu ảnh
        img_path = os.path.join(split_dir, f"{i:06d}.jpg")
        image.save(img_path)

        # Lưu caption (nếu có nhiều, ghi tất cả)
        txt_path = os.path.join(split_dir, f"{i:06d}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for cap in captions:
                # f.write(cap.strip() + "\n")
                f.write(cap)
