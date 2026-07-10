import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import DeepMIMO

# ----------------------------------------
# Step 1: 辅助函数 - 生成标准 DFT 码本
# ----------------------------------------
def generate_dft_codebook(num_antennas):
    """
    生成一维均匀线性阵列(ULA)的 DFT 码本
    """
    # 构造码本矩阵，形状为 [num_antennas, num_beams]
    # 在标准 DFT 码本中，波束数量等于天线数量
    j = 1j
    n = np.arange(num_antennas).reshape(-1, 1)
    k = np.arange(num_antennas).reshape(1, -1)
    codebook = np.exp(-j * 2 * np.pi * n * k / num_antennas) / np.sqrt(num_antennas)
    return codebook

# ----------------------------------------
# Step 2: 自定义 PyTorch Dataset
# ----------------------------------------
class DeepMIMOBeamSelectionDataset(Dataset):
    def __init__(self, deepmimo_dataset, bs_idx=0):
        """
        参数:
            deepmimo_dataset: 通过 DeepMIMO.generate_data 生成的原始字典数据
            bs_idx: 当前处理的基站索引
        """
        bs_data = deepmimo_dataset[bs_idx]
        
        # 1. 提取原始信道矩阵，假设形状为: (用户数, UE天线数, BS天线数, 子载波数量)
        # 针对波束选择，通常对子载波取平均，或仅选择中心子载波。这里对子载波降维
        # 假设用户为单天线(UE=1)，降维后形状: (用户数, BS天线数)
        channels = np.mean(bs_data['channel'], axis=-1).squeeze() 
        num_users, num_antennas = channels.shape
        
        # 2. 生成码本并计算每个用户的最佳波束标签
        codebook = generate_dft_codebook(num_antennas) # 形状: (BS天线数, 波束总数)
        
        features = []
        labels = []
        
        # 遍历每个用户计算最优波束
        for i in range(num_users):
            h = channels[i] # 形状: (BS天线数,)
            
            # 计算信道在所有波束向量上的投影等效增益: |h^H * W|
            # h.conj() 是共轭，乘以码本矩阵
            beam_gains = np.abs(np.dot(h.conj(), codebook))
            
            # 找到功率最大的波束索引作为分类标签
            best_beam_idx = np.argmax(beam_gains)
            
            # 【输入特征设计】：这里以用户坐标 (X, Y, Z) 作为特征进行位置感知波束预测
            # 你也可以修改这里，改为输入“部分天线信道”或“历史信道”
            user_loc = bs_data['user']['location'][i]
            
            features.append(user_loc)
            labels.append(best_beam_idx)
            
        self.features = torch.tensor(np.array(features), dtype=torch.float32)
        self.labels = torch.tensor(np.array(labels), dtype=torch.long) # 分类任务用 LongTensor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ----------------------------------------
# Step 3: 运行管道演示
# ----------------------------------------
if __name__ == "__main__":
    # 配置并生成 DeepMIMO 原始数据
    parameters = {
        'scenario': 'O1_28_NY',
        'num_antennas_BS': [64, 1, 1], # 64天线线性阵列
        'num_antennas_UE': [1, 1, 1],  # 单天线用户
        'ant_spacing_BS': 0.5,
        'ant_spacing_UE': 0.5,
        'bandwidth': 0.05,
        'num_OFDM': 64,
        'active_BS': [1],
        'user_row_first': 1,
        'user_row_last': 20
    }
    
    print("正在生成数据...")
    raw_dataset = DeepMIMO.generate_data(parameters)
    
    # 实例化数据集
    beam_dataset = DeepMIMOBeamSelectionDataset(raw_dataset, bs_idx=0)
    
    # 划分训练集与测试集
    train_size = int(0.8 * len(beam_dataset))
    test_size = len(beam_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(beam_dataset, [train_size, test_size])
    
    # 封装为 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 验证输出形状
    for sample_feat, sample_label in train_loader:
        print("\n--- 批次数据检查 ---")
        print("输入特征形状 (Batch_size, 坐标XYZ):", sample_feat.shape)
        print("标签形状 (Batch_size, 最佳波束索引):", sample_label.shape)
        break
