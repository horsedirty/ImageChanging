#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')  # 设置matplotlib后端
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from PIL import Image, ImageTk

# ========== 版本1.0核心算法 ==========
def date2bytes_v1(date_str: str) -> np.ndarray:
    """yyyy-mm-dd -> 3 字节 [y-2000, m, d]"""
    y, m, d = map(int, date_str.split('-'))
    return np.array([y - 2000, m, d], dtype=np.uint8)


def encrypt_v1(image: np.ndarray, date_str: str) -> np.ndarray:
    """输入灰度图，返回同尺寸加密图"""
    t = date2bytes_v1(date_str)
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


def decrypt_v1(cipher: np.ndarray) -> np.ndarray:
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


# ========== 版本2.0核心算法 ==========
def date2bytes_v2(date_str: str) -> np.ndarray:
    """yyyy-mm-dd -> 3 字节 [y-2000, m, d]"""
    y, m, d = map(int, date_str.split('-'))
    return np.array([y - 2000, m, d], dtype=np.uint8)


def key2bytes_v2(key: str) -> np.ndarray:
    """将用户自定义密钥转换为字节数组"""
    # 使用SHA-256哈希函数将密钥转换为固定长度的字节序列
    hash_obj = hashlib.sha256(key.encode('utf-8'))
    hash_bytes = hash_obj.digest()
    # 取前8个字节作为密钥字节
    return np.frombuffer(hash_bytes[:8], dtype=np.uint8)


def combine_seed_v2(t: np.ndarray, k: np.ndarray) -> int:
    """将时间t和密钥k结合生成种子"""
    combined = np.concatenate([t, k])
    return int.from_bytes(combined.tobytes(), 'big')


def encrypt_v2(image: np.ndarray, date_str: str, user_key: str) -> np.ndarray:
    """输入灰度图，返回同尺寸加密图（version 2.0）"""
    t = date2bytes_v2(date_str)
    k = key2bytes_v2(user_key)
    seed = combine_seed_v2(t, k)
    
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
    k = key2bytes_v2(user_key)
    seed = combine_seed_v2(t, k)
    
    rng = np.random.default_rng(seed)
    L = flat.size
    idx = rng.permutation(L - 3)
    body_rec = np.empty(L - 3, dtype=flat.dtype)
    body_rec[idx] = flat[3:]
    plain_flat = np.concatenate([t, body_rec])
    return plain_flat.reshape(cipher.shape)


# ========== 版本3.0核心算法 ==========
def date2bytes_v3(date_str: str) -> np.ndarray:
    """yyyy-mm-dd -> 3 字节 [y-2000, m, d]"""
    y, m, d = map(int, date_str.split('-'))
    return np.array([y - 2000, m, d], dtype=np.uint8)


def key2bytes_v3(key: str) -> np.ndarray:
    """将用户自定义密钥转换为字节数组"""
    # 使用SHA-256哈希函数将密钥转换为固定长度的字节序列
    hash_obj = hashlib.sha256(key.encode('utf-8'))
    hash_bytes = hash_obj.digest()
    # 取前8个字节作为密钥字节
    return np.frombuffer(hash_bytes[:8], dtype=np.uint8)


def combine_seed_v3(t: np.ndarray, k: np.ndarray) -> int:
    """将时间t和密钥k结合生成种子"""
    combined = np.concatenate([t, k])
    return int.from_bytes(combined.tobytes(), 'big')


def encrypt_v3(image: np.ndarray, date_str: str, user_key: str) -> np.ndarray:
    """输入彩色图像，返回同尺寸加密图（version 3.0）"""
    t = date2bytes_v3(date_str)
    k = key2bytes_v3(user_key)
    seed = combine_seed_v3(t, k)
    
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
        
        # 只在第一个通道保存时间信息，避免重复
        if i == 0:
            cipher_flat = np.concatenate([t, body_shuffled])
        else:
            cipher_flat = np.concatenate([flat[:3], body_shuffled])
        
        encrypted_channel = cipher_flat.reshape(channel.shape)
        encrypted_channels.append(encrypted_channel)
    
    # 合并通道
    if c == 1:
        return encrypted_channels[0]
    else:
        return np.stack(encrypted_channels, axis=2)


def decrypt_v3(cipher: np.ndarray, user_key: str) -> np.ndarray:
    """输入加密图，返回原尺寸彩色图像（version 3.0）"""
    # 获取图像形状
    if len(cipher.shape) == 3:
        h, w, c = cipher.shape
    else:
        h, w = cipher.shape
        c = 1
    
    # 从第一个通道获取时间信息
    if c == 1:
        t = cipher.flat[:3].astype(np.uint8)
    else:
        t = cipher[:, :, 0].flat[:3].astype(np.uint8)  # 只从第一个通道获取时间信息
    k = key2bytes_v3(user_key)
    
    # 分别处理每个颜色通道
    decrypted_channels = []
    
    for i in range(c):
        if c == 1:
            channel = cipher
        else:
            channel = cipher[:, :, i]
            
        flat = channel.flatten()
        # 为每个通道使用相同的种子但加上通道索引以确保不同
        seed = combine_seed_v3(t, k) + i
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


# ========== 工具函数 ==========
def is_color_image(image):
    """判断图像是否为彩色图像"""
    return len(image.shape) == 3 and image.shape[2] >= 3


def show_images_in_gui(original_img, processed_img, title1='Original', title2='Processed'):
    """在GUI中显示图像"""
    # 创建新的顶层窗口
    window = tk.Toplevel()
    window.title(f"{title1} vs {title2}")
    window.geometry("1200x700")
    
    # 添加标签显示信息
    info_label = tk.Label(window, text=f"左侧: {title1}, 右侧: {title2}", 
                         font=("Arial", 14, "bold"))
    info_label.pack(pady=10)
    
    # 创建主框架
    main_frame = tk.Frame(window)
    main_frame.pack(expand=True, fill="both", padx=20, pady=10)
    
    try:
        # 转换图像格式用于显示
        if len(original_img.shape) == 3:  # 彩色图像
            # OpenCV使用BGR，需要转换为RGB
            original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            processed_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB) if len(processed_img.shape) == 3 else cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
        else:  # 灰度图像
            original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
            processed_rgb = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB) if len(processed_img.shape) == 2 else cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        
        # 将numpy数组转换为PIL图像
        original_pil = Image.fromarray(original_rgb)
        processed_pil = Image.fromarray(processed_rgb)
        
        # 调整图像大小以适应窗口，保持宽高比
        max_width = 500
        max_height = 400
        
        def resize_image(img, max_width, max_height):
            ratio = min(max_width/img.width, max_height/img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            return img.resize(new_size, Image.Resampling.LANCZOS)
        
        original_resized = resize_image(original_pil, max_width, max_height)
        processed_resized = resize_image(processed_pil, max_width, max_height)
        
        # 转换为PhotoImage
        original_photo = ImageTk.PhotoImage(original_resized)
        processed_photo = ImageTk.PhotoImage(processed_resized)
        
        # 创建图像显示框架
        image_frame = tk.Frame(main_frame)
        image_frame.pack(expand=True, fill="both")
        
        # 显示原始图像
        orig_frame = tk.Frame(image_frame, relief="solid", bd=2)
        orig_frame.pack(side="left", padx=10, expand=True)
        
        orig_title = tk.Label(orig_frame, text=title1, font=("Arial", 12, "bold"))
        orig_title.pack(pady=5)
        
        orig_label = tk.Label(orig_frame, image=original_photo)
        orig_label.pack(pady=5)
        orig_label.image = original_photo  # 保持引用
        
        orig_info = tk.Label(orig_frame, 
                           text=f"尺寸: {original_img.shape[1]}x{original_img.shape[0]}\n类型: {'彩色' if len(original_img.shape) == 3 else '灰度'}",
                           font=("Arial", 10))
        orig_info.pack(pady=5)
        
        # 显示处理后图像
        proc_frame = tk.Frame(image_frame, relief="solid", bd=2)
        proc_frame.pack(side="right", padx=10, expand=True)
        
        proc_title = tk.Label(proc_frame, text=title2, font=("Arial", 12, "bold"))
        proc_title.pack(pady=5)
        
        proc_label = tk.Label(proc_frame, image=processed_photo)
        proc_label.pack(pady=5)
        proc_label.image = processed_photo  # 保持引用
        
        proc_info = tk.Label(proc_frame,
                           text=f"尺寸: {processed_img.shape[1]}x{processed_img.shape[0]}\n类型: {'彩色' if len(processed_img.shape) == 3 else '灰度'}",
                           font=("Arial", 10))
        proc_info.pack(pady=5)
        
    except Exception as e:
        # 如果图像显示失败，显示错误信息和图像统计
        error_frame = tk.Frame(main_frame)
        error_frame.pack(expand=True, fill="both")
        
        error_label = tk.Label(error_frame, 
                             text=f"图像预览功能暂时不可用: {str(e)}\n\n显示图像统计信息:",
                             font=("Arial", 12), fg="red", wraplength=600)
        error_label.pack(pady=20)
        
        # 显示统计信息
        stats_frame = tk.Frame(error_frame)
        stats_frame.pack(expand=True, fill="both", padx=20)
        
        orig_stats = f"{title1}:\n尺寸: {original_img.shape}\n最小值: {original_img.min()}\n最大值: {original_img.max()}\n平均值: {original_img.mean():.2f}"
        proc_stats = f"{title2}:\n尺寸: {processed_img.shape}\n最小值: {processed_img.min()}\n最大值: {processed_img.max()}\n平均值: {processed_img.mean():.2f}"
        
        orig_label = tk.Label(stats_frame, text=orig_stats, font=("Arial", 10), 
                            justify="left", relief="solid", padx=10, pady=10)
        orig_label.pack(side="left", padx=10, expand=True)
        
        proc_label = tk.Label(stats_frame, text=proc_stats, font=("Arial", 10), 
                            justify="left", relief="solid", padx=10, pady=10)
        proc_label.pack(side="right", padx=10, expand=True)
    
    # 添加关闭按钮
    close_button = tk.Button(window, text="关闭", command=window.destroy, 
                           bg="#4CAF50", fg="white", font=("Arial", 12))
    close_button.pack(pady=20)


# ========== GUI应用程序 ==========
class ImageEncryptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图像加密解密工具")
        self.root.geometry("600x500")
        
        # 变量
        self.image_path = tk.StringVar()
        self.version = tk.StringVar(value="3.0")
        self.operation = tk.StringVar(value="encrypt")
        self.date = tk.StringVar(value="2025-06-25")
        self.key = tk.StringVar()
        
        self.create_widgets()
        
    def create_widgets(self):
        # 标题
        title_label = tk.Label(self.root, text="图像加密解密工具", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # 文件选择框架
        file_frame = tk.Frame(self.root)
        file_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(file_frame, text="图像文件:").grid(row=0, column=0, sticky="w")
        tk.Entry(file_frame, textvariable=self.image_path, width=50).grid(row=0, column=1, padx=5)
        tk.Button(file_frame, text="浏览", command=self.browse_file).grid(row=0, column=2)
        
        # 版本选择框架
        version_frame = tk.Frame(self.root)
        version_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(version_frame, text="选择版本:").grid(row=0, column=0, sticky="w")
        version_combo = ttk.Combobox(version_frame, textvariable=self.version, 
                                    values=["1.0", "2.0", "3.0"], state="readonly", width=10)
        version_combo.grid(row=0, column=1, padx=5)
        
        # 操作选择框架
        operation_frame = tk.Frame(self.root)
        operation_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(operation_frame, text="操作类型:").grid(row=0, column=0, sticky="w")
        tk.Radiobutton(operation_frame, text="加密", variable=self.operation, value="encrypt").grid(row=0, column=1, padx=10)
        tk.Radiobutton(operation_frame, text="解密", variable=self.operation, value="decrypt").grid(row=0, column=2, padx=10)
        
        # 日期输入框架
        date_frame = tk.Frame(self.root)
        date_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(date_frame, text="日期 (yyyy-mm-dd):").grid(row=0, column=0, sticky="w")
        tk.Entry(date_frame, textvariable=self.date, width=20).grid(row=0, column=1, padx=5)
        
        # 密钥输入框架
        key_frame = tk.Frame(self.root)
        key_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(key_frame, text="密钥 (可选):").grid(row=0, column=0, sticky="w")
        tk.Entry(key_frame, textvariable=self.key, width=30).grid(row=0, column=1, padx=5)
        
        # 执行按钮
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="执行", command=self.execute_operation, bg="#4CAF50", fg="white", 
                 font=("Arial", 12, "bold"), padx=20, pady=5).pack(side="left", padx=10)
        
        tk.Button(button_frame, text="重置", command=self.reset_fields, bg="#f44336", fg="white", 
                 font=("Arial", 12, "bold"), padx=20, pady=5).pack(side="left", padx=10)
        
        # 结果文本框
        result_frame = tk.Frame(self.root)
        result_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        tk.Label(result_frame, text="操作结果:").pack(anchor="w")
        self.result_text = tk.Text(result_frame, height=8, wrap="word")
        self.result_text.pack(fill="both", expand=True, pady=5)
        
        scrollbar = tk.Scrollbar(self.result_text)
        scrollbar.pack(side="right", fill="y")
        self.result_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.result_text.yview)
        
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.image_path.set(file_path)
            
    def reset_fields(self):
        self.image_path.set("")
        self.version.set("3.0")
        self.operation.set("encrypt")
        self.date.set("2025-06-25")
        self.key.set("")
        self.result_text.delete(1.0, tk.END)
        
    def log_result(self, message):
        self.result_text.insert(tk.END, message + "\n")
        self.result_text.see(tk.END)
        
    def execute_operation(self):
        # 获取输入参数
        image_path = self.image_path.get()
        version = self.version.get()
        operation = self.operation.get()
        date_str = self.date.get()
        user_key = self.key.get()
        
        # 验证输入
        if not image_path:
            messagebox.showerror("错误", "请选择图像文件")
            return
            
        if not os.path.exists(image_path):
            messagebox.showerror("错误", "选定的图像文件不存在")
            return
            
        try:
            # 根据操作类型执行相应功能
            if operation == "encrypt":
                self.encrypt_image(image_path, version, date_str, user_key)
            else:
                self.decrypt_image(image_path, version, user_key)
                
        except Exception as e:
            messagebox.showerror("错误", f"操作失败: {str(e)}")
            self.log_result(f"错误: {str(e)}")
            
    def encrypt_image(self, image_path, version, date_str, user_key):
        self.log_result(f"开始加密图像: {image_path}")
        self.log_result(f"使用版本: {version}")
        
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("无法读取图像文件")
            
        self.log_result(f"原始图像形状: {img.shape}")
        self.log_result(f"是否为彩色图像: {is_color_image(img)}")
        
        # 根据版本选择加密方法
        if version == "1.0":
            # Version 1.0 只处理灰度图像
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            encrypted = encrypt_v1(gray, date_str)
            output_filename = "encrypted_image_v1.png"
        elif version == "2.0":
            # Version 2.0 只处理灰度图像
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            encrypted = encrypt_v2(gray, date_str, user_key)
            output_filename = "encrypted_image_v2.png"
        else:  # version == "3.0"
            # Version 3.0 处理彩色图像
            encrypted = encrypt_v3(img, date_str, user_key)
            output_filename = "encrypted_image_v3.png"
            
        # 保存加密图像
        cv2.imwrite(output_filename, encrypted)
        self.log_result(f"加密图像已保存为: {output_filename}")
        self.log_result(f"加密图像形状: {encrypted.shape}")
        
        # 显示结果
        if version in ["1.0", "2.0"]:
            show_images_in_gui(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), encrypted, 
                              "Original (Grayscale)", "Encrypted")
        else:
            show_images_in_gui(img, encrypted, "Original", "Encrypted")
            
        messagebox.showinfo("成功", f"图像加密完成!\n保存为: {output_filename}")
        
    def decrypt_image(self, image_path, version, user_key):
        self.log_result(f"开始解密图像: {image_path}")
        self.log_result(f"使用版本: {version}")
        
        # 读取加密图像
        enc_img = cv2.imread(image_path)
        if enc_img is None:
            raise ValueError("无法读取加密图像文件")
            
        self.log_result(f"加密图像形状: {enc_img.shape}")
        
        # 根据版本选择解密方法
        if version == "1.0":
            # 版本1.0只处理灰度图像
            if len(enc_img.shape) == 3:
                enc_img = cv2.cvtColor(enc_img, cv2.COLOR_BGR2GRAY)
                self.log_result("将加密图像转换为灰度图像")
            decrypted = decrypt_v1(enc_img)
            output_filename = "decrypted_image_v1.png"
        elif version == "2.0":
            # 版本2.0只处理灰度图像
            if len(enc_img.shape) == 3:
                enc_img = cv2.cvtColor(enc_img, cv2.COLOR_BGR2GRAY)
                self.log_result("将加密图像转换为灰度图像")
            decrypted = decrypt_v2(enc_img, user_key)
            output_filename = "decrypted_image_v2.png"
        else:  # version == "3.0"
            decrypted = decrypt_v3(enc_img, user_key)
            output_filename = "decrypted_image_v3.png"
            
        # 确保数据类型正确
        if len(decrypted.shape) == 3:
            decrypted = np.clip(decrypted, 0, 255).astype(np.uint8)
        else:
            decrypted = np.clip(decrypted, 0, 255).astype(np.uint8)
            
        # 保存解密图像
        cv2.imwrite(output_filename, decrypted)
        self.log_result(f"解密图像已保存为: {output_filename}")
        self.log_result(f"解密图像形状: {decrypted.shape}")
        
        # 显示结果
        show_images_in_gui(enc_img, decrypted, "Encrypted", "Decrypted")
        
        messagebox.showinfo("成功", f"图像解密完成!\n保存为: {output_filename}")


# ========== 主程序入口 ==========
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEncryptionApp(root)
    root.mainloop()