import os
from huggingface_hub import snapshot_download

# Định nghĩa tên mô hình và đường dẫn lưu cục bộ
model_name = "intfloat/multilingual-e5-large"
local_model_path = "./local_models/multilingual-e5-large" # Thư mục bạn muốn lưu

# Tạo thư mục nếu nó chưa tồn tại
os.makedirs(local_model_path, exist_ok=True)

print(f"Đang tải mô hình '{model_name}' về '{local_model_path}'...")
try:
    # Sử dụng snapshot_download để tải toàn bộ repo mô hình
    # local_dir_use_symlinks=False để đảm bảo tất cả file được tải vào local_model_path
    snapshot_download(repo_id=model_name, local_dir=local_model_path, local_dir_use_symlinks=False)
    print(f"Tải mô hình '{model_name}' hoàn tất vào '{local_model_path}'.")
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    print("Vui lòng kiểm tra kết nối internet và tên mô hình.")