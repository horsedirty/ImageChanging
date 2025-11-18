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


def encrypt_v2(image: np.ndarray, date_str: str, user_key: str) -> np.ndarray:
    """输入灰度图，返回同尺寸加密图（version 2.0）"""
    t = date2bytes(date_str)
    k = key2bytes(user_key)
    seed = combine_seed(t, k)
    
    flat = image.flatten()
    L = flat.size
    if L < 3:
        raise ValueError('图像像素不足 3 个')
    
    rng = np.random.default_rng(seed)
    idx = rng.permutation(L - 3)
    body_shuffled = flat[3:][idx]
    cipher_flat = np.concatenate([t, body_shuffled])
    return cipher_flat.reshape(image.shape)


def decrypt_v2(cipher: np.ndarray, user_key: str) -> np.ndarray:
    """输入加密图，返回原尺寸灰度图（version 2.0）"""
    flat = cipher.flatten()
    t = flat[:3].astype(np.uint8)
    k = key2bytes(user_key)
    seed = combine_seed(t, k)
    
    rng = np.random.default_rng(seed)
    L = flat.size
    idx = rng.permutation(L - 3)
    body_rec = np.empty(L - 3, dtype=flat.dtype)
    body_rec[idx] = flat[3:]
    plain_flat = np.concatenate([t, body_rec])
    return plain_flat.reshape(cipher.shape)


# ---------- 可视化 ----------
def show_pair(img1, img2, title1='Original', title2='Encrypted'):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.title(title1)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.title(title2)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


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

    # 自动读入并转灰度
    img_bgr = cv2.imread(img_path)        # BGR
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图片: {img_path}")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    enc = encrypt_v2(gray, date_str, user_key)
    
    # 保存加密后的图片
    cv2.imwrite('encrypted_image_v2.png', enc)
    print("加密图片已保存为 'encrypted_image_v2.png'")
    
    show_pair(gray, enc, title2='Encrypted (noise-like)')

    # 解密并展示
    dec = decrypt_v2(enc, user_key)
    show_pair(gray, dec, title1='Original', title2='Decrypted')
    print('PSNR 原图 vs 解密图：', cv2.PSNR(gray, dec), 'dB')


def main_decrypt():
    """如果手里只有加密图，直接跑这段"""
    enc_path = input("请输入加密图片路径（默认为'encrypted_image_v2.png'）: ").strip()
    if not enc_path:
        enc_path = 'encrypted_image_v2.png'
    
    user_key = input("请输入密钥（必须与加密时一致，默认为空）: ").strip()
    if not user_key:
        user_key = ''  # 默认为空字符串
    
    enc = cv2.imread(enc_path, cv2.IMREAD_GRAYSCALE)
    if enc is None:
        raise FileNotFoundError(f"无法读取加密图片: {enc_path}")
        
    dec = decrypt_v2(enc, user_key)
    show_pair(enc, dec, title1='Encrypted', title2='Decrypted')


if __name__ == '__main__':
    #main()          # 一次性看"原图-加密-解密"三张图
    main_decrypt()   # 直接解密并展示