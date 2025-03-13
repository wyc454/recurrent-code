import numpy as np
import torch
from PIL import Image
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import random
# 设置随机种子以确保结果可复现
random.seed(42)
def generate_residue_image(interferogram):
    """使用2x2邻域积分计算残差图像"""
    rows, cols = interferogram.shape
    residue_image = np.zeros_like(interferogram, dtype=bool)

    for i in range(rows - 1):
        for j in range(cols - 1):
            # 计算2x2邻域闭合路径积分
            sum_phase = (interferogram[i + 1, j] - interferogram[i, j]) + \
                        (interferogram[i, j + 1] - interferogram[i + 1, j]) + \
                        (interferogram[i + 1, j + 1] - interferogram[i, j + 1]) + \
                        (interferogram[i, j] - interferogram[i + 1, j + 1])

            # 判断是否存在残点（模2π后非零）
            if np.abs(sum_phase % (2 * np.pi)) > 1e-6:
                residue_image[i, j] = True

    return residue_image


def generate_branch_cut_label(reference_unwrapped_phase, residue_image):
    """生成分支切割标签图像"""
    wrapped_phase = np.angle(np.exp(1j * reference_unwrapped_phase))
    k_opt = np.round((reference_unwrapped_phase - wrapped_phase) / (2 * np.pi))

    # 计算梯度
    k_opt_grad_x, k_opt_grad_y = np.gradient(k_opt)
    res_grad_x, res_grad_y = np.gradient(residue_image.astype(np.float32))

    # 比较梯度差异
    branch_cut = (res_grad_x != k_opt_grad_x) | (res_grad_y != k_opt_grad_y)
    return branch_cut.astype(np.uint8) * 255  # 转换为0-255灰度值


# 配置路径
TRAIN_DATA_DIR = r"，，，，，"  # 请替换为实际数据路径
train_wrapped_dir = os.path.join(TRAIN_DATA_DIR, "。。。。。")
train_absolute_dir = os.path.join(TRAIN_DATA_DIR, "。。。。。")

# 创建输出目录
residue_dir = os.path.join(TRAIN_DATA_DIR, "residue_images")
branch_cut_dir = os.path.join(TRAIN_DATA_DIR, "branch_cut_labels")
os.makedirs(residue_dir, exist_ok=True)
os.makedirs(branch_cut_dir, exist_ok=True)

# 获取所有图像文件
image_files = os.listdir(train_wrapped_dir)
num_samples = len(image_files)

for sample_idx in range(num_samples):
    # 读取干涉图（已包含噪声）
    wrapped_path = os.path.join(train_wrapped_dir, image_files[sample_idx])
    wrapped_img = Image.open(wrapped_path).convert('L')
    interferogram = np.array(wrapped_img, dtype=np.float32) * (2 * np.pi / 255)

    # 读取参考解缠相位
    absolute_path = os.path.join(train_absolute_dir, image_files[sample_idx])
    absolute_img = Image.open(absolute_path).convert('L')
    reference_phase = np.array(absolute_img, dtype=np.float32) * (2 * np.pi / 255)

    # 生成残差图像
    residue = generate_residue_image(interferogram)
    # 生成分支切割标签
    branch_cut = generate_branch_cut_label(reference_phase, residue)
    # 保存结果
    residue_path = os.path.join(residue_dir, f"train_residue_{sample_idx:05d}.png")
    branch_cut_path = os.path.join(branch_cut_dir, f"train_branch_cut_{sample_idx:05d}.png")
    Image.fromarray((residue * 255).astype(np.uint8)).save(residue_path)
    Image.fromarray(branch_cut).save(branch_cut_path)
