#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import datetime
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ---------- 核心算法 ----------
def date2bytes(date_str: str) -> np.ndarray:
    """yyyy-mm-dd -> 3 字节 [y-2000, m, d]"""
    y, m, d = map(int, date_str.split('-'))
    return np.array([y - 2000, m, d], dtype=np.uint8)


def encrypt(image: np.ndarray, date_str: str) -> np.ndarray:
    """输入灰度图，返回同尺寸加密图"""
    t = date2bytes(date_str)
    flat = image.flatten()
    L = flat.size
    if L < 3:
        raise ValueError('图像像素不足 3 个')
    seed = int.from_bytes(t.tobytes(), 'big')
    rng = np.random.default_rng(seed)
    idx = rng.permutation(L - 3)
    body_shuffled = flat[3:][idx]
    cipher_flat = np.concatenate([t, body_shuffled])
    return cipher_flat.reshape(image.shape)


def decrypt(cipher: np.ndarray) -> np.ndarray:
    """输入加密图，返回原尺寸灰度图"""
    flat = cipher.flatten()
    t = flat[:3].astype(np.uint8)
    seed = int.from_bytes(t.tobytes(), 'big')
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
    img_path = 'cat.jpeg'          # 图片路径
    date_str = '2025-06-25'        # 想写的日期

    # 自动读入并转灰度
    img_bgr = cv2.imread(img_path)        # BGR
    if img_bgr is None:
        raise FileNotFoundError(img_path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    enc = encrypt(gray, date_str)
    
    # 保存加密后的图片
    cv2.imwrite('encrypted_image.png', enc)
    print("加密图片已保存为 'encrypted_image.png'")
    
    show_pair(gray, enc, title2='Encrypted (noise-like)')

    # 解密并展示
    dec = decrypt(enc)
    show_pair(gray, dec, title1='Original', title2='Decrypted')
    print('PSNR 原图 vs 解密图：', cv2.PSNR(gray, dec), 'dB')


def main_decrypt():
    """如果手里只有加密图，直接跑这段"""
    enc_path = 'encrypted_image.png'   # 加密图路径
    enc = cv2.imread(enc_path, cv2.IMREAD_GRAYSCALE)
    dec = decrypt(enc)
    show_pair(enc, dec, title1='Encrypted', title2='Decrypted')

if __name__ == '__main__':
    main()          # 一次性看“原图-加密-解密”三张图
    #main_decrypt()   # 直接解密并展示