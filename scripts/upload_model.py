"""
上传模型到 Hugging Face
用法：python scripts/upload_model.py YOUR_HF_TOKEN
"""

import sys
import os

# 清除可能冲突的代理
for key in ['HTTP_PROXY', 'http_proxy', 'HTTPS_PROXY', 'https_proxy']:
    os.environ.pop(key, None)

# 设置正确的代理（Clash）
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

from huggingface_hub import HfApi, login

REPO_ID = "zzyy540886666/corneal-model"
LOCAL_PATH = "checkpoints/best_model.pth"
REMOTE_PATH = "best_model.pth"

def main():
    if len(sys.argv) > 1:
        token = sys.argv[1]
    else:
        token = input("粘贴你的 HF Token (hf_xxxxx): ").strip()

    print(f"正在验证 Token...")
    try:
        login(token=token, add_to_git_credential=False)
        print("登录成功！")
    except Exception as e:
        print(f"登录失败: {e}")
        print("请确认 Token 正确，且 VPN 已开启")
        sys.exit(1)

    print(f"正在上传 {LOCAL_PATH} -> {REPO_ID}/{REMOTE_PATH}")
    print("文件约 1 GB，请耐心等待...\n")

    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=LOCAL_PATH,
            path_in_repo=REMOTE_PATH,
            repo_id=REPO_ID,
            repo_type="model",
            token=token,
        )
        print(f"\n上传成功！")
        print(f"在线地址: https://huggingface.co/{REPO_ID}/blob/main/{REMOTE_PATH}")
    except Exception as e:
        print(f"上传失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
