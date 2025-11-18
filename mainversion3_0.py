#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ---------- 核心算法 ----------
def date2bytes(date_str: str) -> np.ndarray:
    """yyyy-mm-dd -> 3 字节 [y-2000, m, d]"""
    y, m, d = map(int, date_str.split('-'))
    return np.array([y - 2000, m, d], dtype=np.uint8)


def key2bytes(key: str) -> np.ndarray:
    """将用户自定义密钥转换为字节数组"""
    # 使用SHA-256哈希函数将密钥转换为固定长度的字节序列
    hash_obj = hashlib.sha256(key.encode('utf-8'))
    hash_bytes = hash_obj.digest()
    # 取前8个字节作为密钥字节
    return np.frombuffer(hash_bytes[:8], dtype=np.uint8)


def combine_seed(t: np.ndarray, k: np.ndarray) -> int:
    """将时间t和密钥k结合生成种子"""
    combined = np.concatenate([t, k])
    return int.from_bytes(combined.tobytes(), 'big')


def encrypt_color(image: np.ndarray, date_str: str, user_key: str) -> np.ndarray:
    """输入彩色图像，返回同尺寸加密图（version 3.0）"""
    t = date2bytes(date_str)
    k = key2bytes(user_key)
    seed = combine_seed(t, k)
    
    # 获取图像形状
    if len(image.shape) == 3:
        h, w, c = image.shape
    else:
        h, w = image.shape
        c = 1
    
    # 分别处理每个颜色通道
    encrypted_channels = []
    
    for i in range(c):
        if c == 1:
            channel = image
        else:
            channel = image[:, :, i]
            
        flat = channel.flatten()
        L = flat.size
        if L < 3:
            raise ValueError('图像像素不足 3 个')
        
        # 为每个通道使用相同的种子但加上通道索引以确保不同
        channel_seed = seed + i
        rng = np.random.default_rng(channel_seed)
        idx = rng.permutation(L - 3)
        body_shuffled = flat[3:][idx]
        cipher_flat = np.concatenate([t, body_shuffled])
        encrypted_channel = cipher_flat.reshape(channel.shape)
        encrypted_channels.append(encrypted_channel)
    
    # 合并通道
    if c == 1:
        return encrypted_channels[0]
    else:
        return np.stack(encrypted_channels, axis=2)


def decrypt_color(cipher: np.ndarray, user_key: str) -> np.ndarray:
    """输入加密图，返回原尺寸彩色图像（version 3.0）"""
    # 获取图像形状
    if len(cipher.shape) == 3:
        h, w, c = cipher.shape
    else:
        h, w = cipher.shape
        c = 1
    
    t = cipher.flat[:3].astype(np.uint8)  # 从第一个通道获取时间信息
    k = key2bytes(user_key)
    
    # 分别处理每个颜色通道
    decrypted_channels = []
    
    for i in range(c):
        if c == 1:
            channel = cipher
        else:
            channel = cipher[:, :, i]
            
        flat = channel.flatten()
        # 为每个通道使用相同的种子但加上通道索引以确保不同
        seed = combine_seed(t, k) + i
        rng = np.random.default_rng(seed)
        L = flat.size
        idx = rng.permutation(L - 3)
        body_rec = np.empty(L - 3, dtype=flat.dtype)
        body_rec[idx] = flat[3:]
        plain_flat = np.concatenate([t, body_rec])
        decrypted_channel = plain_flat.reshape(channel.shape)
        decrypted_channels.append(decrypted_channel)
    
    # 合并通道
    if c == 1:
        return decrypted_channels[0]
    else:
        return np.stack(decrypted_channels, axis=2)


# ---------- 可视化 ----------
def show_pair_color(img1, img2, title1='Original', title2='Encrypted'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 显示原始图像
    if len(img1.shape) == 3:
        # 彩色图像
        axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    else:
        # 灰度图像
        axes[0].imshow(img1, cmap='gray')
    axes[0].set_title(title1)
    axes[0].axis('off')
    
    # 显示加密图像
    if len(img2.shape) == 3:
        # 彩色图像
        axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    else:
        # 灰度图像
        axes[1].imshow(img2, cmap='gray')
    axes[1].set_title(title2)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


# ---------- 工具函数 ----------
def is_color_image(image):
    """判断图像是否为彩色图像"""
    return len(image.shape) == 3 and image.shape[2] >= 3


# ---------- 主入口 ----------
def main():
    # 获取用户输入
    img_path = input("请输入图片路径（默认为'cat.jpeg'）: ").strip()
    if not img_path:
        img_path = 'cat.jpeg'
    
    user_key = input("请输入密钥（默认为空）: ").strip()
    if not user_key:
        user_key = ''  # 默认为空字符串
    
    date_str = input("请输入日期（格式 yyyy-mm-dd，默认为'2025-06-25'）: ").strip()
    if not date_str:
        date_str = '2025-06-25'

    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {img_path}")
    
    print(f"原始图像形状: {img.shape}")
    print(f"是否为彩色图像: {is_color_image(img)}")

    enc = encrypt_color(img, date_str, user_key)
    
    # 保存加密后的图片
    cv2.imwrite('encrypted_image_v3.png', enc)
    print("加密图片已保存为 'encrypted_image_v3.png'")
    print(f"加密图像形状: {enc.shape}")
    
    show_pair_color(img, enc, title1='Original', title2='Encrypted (noise-like)')

    # 解密并展示
    dec = decrypt_color(enc, user_key)
    
    # 确保解密图像的数据类型与原始图像一致
    if len(img.shape) == 3:
        dec = np.clip(dec, 0, 255).astype(np.uint8)
    else:
        dec = np.clip(dec, 0, 255).astype(np.uint8)
    
    show_pair_color(img, dec, title1='Original', title2='Decrypted')
    
    # 计算PSNR
    if len(img.shape) == 3 and len(dec.shape) == 3:
        # 对于彩色图像，分别计算每个通道的PSNR然后取平均
        psnr_values = []
        for i in range(img.shape[2]):
            psnr = cv2.PSNR(img[:, :, i], dec[:, :, i])
            psnr_values.append(psnr)
        avg_psnr = np.mean(psnr_values)
        print(f'PSNR 原图 vs 解密图：{avg_psnr:.2f} dB')
    else:
        print('PSNR 原图 vs 解密图：', cv2.PSNR(img, dec), 'dB')


def main_decrypt():
    """如果手里只有加密图，直接跑这段"""
    enc_path = input("请输入加密图片路径（默认为'encrypted_image_v3.png'）: ").strip()
    if not enc_path:
        enc_path = 'encrypted_image_v3.png'
    
    user_key = input("请输入密钥（必须与加密时一致，默认为空）: ").strip()
    if not user_key:
        user_key = ''  # 默认为空字符串
    
    enc = cv2.imread(enc_path)
    if enc is None:
        raise FileNotFoundError(f"无法读取加密图片: {enc_path}")
        
    print(f"加密图像形状: {enc.shape}")
        
    dec = decrypt_color(enc, user_key)
    
    # 确保解密图像的数据类型正确
    dec = np.clip(dec, 0, 255).astype(np.uint8)
    
    show_pair_color(enc, dec, title1='Encrypted', title2='Decrypted')


if __name__ == '__main__':
    main()          # 一次性看"原图-加密-解密"三张图
    # main_decrypt()   # 直接解密并展示