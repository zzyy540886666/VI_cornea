"""
模型文件下载脚本
从网盘链接下载 best_model.pth 到 checkpoints/ 目录

使用方法：
  python scripts/download_model.py

将下方 MODEL_URL 替换为你的网盘直链即可。
如果网盘没有直链功能，请手动下载后放入 checkpoints/ 目录。
"""

import os
import sys
import urllib.request
from pathlib import Path

# ══════════════════════════════════════════
#  在这里填入你的网盘直链（替换下方 URL）
# ══════════════════════════════════════════
MODEL_URL = ""
MODEL_FILE = "best_model.pth"
# ══════════════════════════════════════════


def download_file(url: str, dest: Path):
    """下载文件并显示进度"""
    if not url:
        print("未配置下载链接，请手动下载模型文件。")
        print(f"将 {MODEL_FILE} 放入以下目录：")
        print(f"  {dest.parent.absolute()}")
        return False

    print(f"正在下载: {url}")
    print(f"保存到:   {dest}")

    try:
        # 显示下载进度
        def reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(downloaded * 100 / total_size, 100)
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r  进度: {mb_down:.1f} / {mb_total:.1f} MB ({percent:.1f}%)", end="", flush=True)

        urllib.request.urlretrieve(url, dest, reporthook)
        print(f"\n下载完成！文件大小: {dest.stat().st_size / (1024*1024):.1f} MB")
        return True
    except Exception as e:
        print(f"\n下载失败: {e}")
        print("请手动下载模型文件。")
        # 删除不完整的文件
        if dest.exists():
            dest.unlink()
        return False


def main():
    project_root = Path(__file__).parent.parent
    checkpoints_dir = project_root / "checkpoints"
    model_path = checkpoints_dir / MODEL_FILE

    # 检查模型是否已存在
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"模型文件已存在: {model_path} ({size_mb:.1f} MB)")
        overwrite = input("是否重新下载？(y/N): ").strip().lower()
        if overwrite != "y":
            print("跳过下载。")
            return

    # 确保目录存在
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # 下载
    success = download_file(MODEL_URL, model_path)

    if success:
        print("\n完成！现在可以启动应用：")
        print("  streamlit run app.py")
    else:
        print("\n手动下载步骤：")
        print(f"  1. 从网盘下载 {MODEL_FILE}")
        print(f"  2. 将文件放入: {checkpoints_dir.absolute()}")
        print(f"  3. 运行: streamlit run app.py")


if __name__ == "__main__":
    main()
