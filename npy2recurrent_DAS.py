import os
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler


class NumpyArrayProcessor:
    def __init__(self, input_dirs, output_dirs):
        """
        :param input_dirs: 输入目录字典，例如 {"wind": "path1", "knocking": "path2", "machine": "path3"}
        :param output_dirs: 输出目录字典，例如 {"wind": "path1", "knocking": "path2", "machine": "path3"}
        """
        self.input_dirs = input_dirs
        self.output_dirs = output_dirs

        for path in self.output_dirs.values():
            os.makedirs(path, exist_ok=True)

    def get_all_records(self, category):
        """获取指定类别的所有记录文件名（去除后缀）"""
        input_dir = self.input_dirs[category]
        return [file.split('.')[0] for file in listdir(input_dir) if file.endswith('.txt')]

    def numpy_array2RP(self, rec, category, eps=0.05, steps=3):
        """处理 NumPy 数组并生成递归图"""
        input_dir = self.input_dirs[category]
        np_array = np.loadtxt(f'{input_dir}/{rec}.txt')  # 形状 (2, 9600)
        np_array = np_array.reshape(-1, 1)
        print(f"{rec} array shape: {np_array.shape}")

        # 归一化处理
        scaler = StandardScaler()
        signal = scaler.fit_transform(np_array.reshape(-1, 1)).flatten()

        print(f"Processed signal shape: {signal.shape}")

        # 生成并保存递归图
        self.save_plot(signal, rec, eps, steps, category)

    @staticmethod
    def recurrence_plot(signal, eps=0.10, steps=3):
        """生成递归图"""
        _2d_array = signal[:, None]
        distance = pdist(_2d_array)
        distance = np.floor(distance / eps)
        distance[distance > steps] = steps
        return squareform(distance)

    def save_plot(self, signal, rec, eps, steps, category):
        """绘制并保存递归图"""
        rp = self.recurrence_plot(signal, eps, steps)

        plt.figure(figsize=(6, 6))
        plt.imshow(rp, cmap='binary')
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        save_path = f"{self.output_dirs[category]}/{rec}.png"
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        print(f"Saved RP plot: {save_path}")

    def process_all_records(self, eps=0.05, steps=3):
        """处理所有 NumPy 记录"""
        for category in self.input_dirs.keys():
            print(f"Processing category: {category}")
            for rec in tqdm(self.get_all_records(category)):
                self.numpy_array2RP(rec, category, eps, steps)


if __name__ == "__main__":
    processor = NumpyArrayProcessor(
        {"wind": "DAS/0wind", "knocking": "DAS/1manual", "machine": "DAS/2digger"},
        {"wind": "Recurrent_map_DAS/0wind", "knocking": "Recurrent_map_DAS/1manual", "machine": "Recurrent_map_DAS/2digger"}
    )
    processor.process_all_records(eps=0.05, steps=3)