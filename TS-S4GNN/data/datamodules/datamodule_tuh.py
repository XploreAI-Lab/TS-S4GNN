import sys
import os
import pytorch_lightning as pl
import pickle
import numpy as np
import h5py
import pandas as pd
import torch
import torch_geometric

from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Data, Dataset
from typing import Optional
from tqdm import tqdm
from constants import TUH_FREQUENCY as FREQ
from data.data_utils.general_data_utils import StandardScaler, ImbalancedDatasetSampler
#FILEMARKER_DIR = "data/file_markers_tuh_v1.5.2"       #  检测 
FILEMARKER_DIR = "data/file_markers_classification_copy"        #分类


class TUHDataset(InMemoryDataset):                                    #处理和存储 TUH 数据集中的单个样本
    def __init__(                                                     #初始化数据集的根目录、原始数据路径、文件标记、数据集划分（训练/验证/测试）、序列长度、节点数量、邻接矩阵目录、标准化器等
        self,                                                         
        root,
        raw_data_path,
        file_marker,
        split,
        seq_len,
        num_nodes,
        adj_mat_dir,
        scaler=None,
        transform=None,
        pre_transform=None,
        repreproc=False,
    ):
        self.root = root
        self.raw_data_path = raw_data_path
        self.file_marker = file_marker
        self.split = split
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.adj_mat_dir = adj_mat_dir
        self.scaler = scaler

        self.df_file = file_marker                                     #读取文件标记信息，并提取文件名、标签和剪辑索引
        self.file_names = self.df_file["file_name"].tolist()
        #self.labels = self.df_file["is_seizure"].tolist()         #检测
        self.labels = self.df_file["class"].tolist()          #分类
        self.clip_idxs = self.df_file["clip_index"].tolist()
        ''' 检测分类标签
        # ===== 在这里添加标签验证代码 =====
        print("==== 原始标签验证 ====")
        print("CSV中唯一标签:", self.df_file["class"].unique())
        print("CSV中标签分布:\n", self.df_file["class"].value_counts())
        
        unique_labels = list(set(self.labels))
        print("self.labels中的唯一标签:", sorted(unique_labels))
        assert len(unique_labels) == 4, f"标签缺失！实际标签: {unique_labels}"
        # ===== 验证代码结束 =====
        '''
        # process
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):                                           #返回原始数据文件的路径列表(resampled_data)
        return [
            os.path.join(self.raw_data_path, fn)
            for fn in os.listdir(self.raw_data_path)
        ]

    @property
    def processed_file_names(self):                                     #返回处理后的数据文件的路径列表(preproc_data)
        return ["{}_{}.pt".format(self.file_names[idx].split(".h5")[0], self.clip_idxs[idx]) for idx in range(len(self.df_file))]

    def len(self):                                                      #返回数据集中的样本数量
        return len(self.file_names)

    def _get_combined_graph(self):                                      #返回邻接矩阵
        with open(self.adj_mat_dir, "rb") as pf:
            adj_mat = pickle.load(pf)
            adj_mat = adj_mat[-1]
        return adj_mat

    def get_labels(self):                                               #将标签转换为 FloatTensor
        return torch.LongTensor(self.labels)  # 原先是FloatTensor
       # return torch.FloatTensor(self.labels)

    def process(self):                                                  #处理原始数据文件，将其转换为图数据格式，并保存为 .pt 文件
        for idx in tqdm(range(len(self.file_names))):

            h5_file_name = self.file_names[idx]                         #读取 HDF5 文件中的脑电图信号
            y = self.labels[idx]                                        #标签：当前样本的类别
            clip_idx = int(self.df_file.iloc[idx]["clip_index"])        #剪辑索引用于确定从脑电图信号中提取的特定片段

            writeout_fn = h5_file_name.split(".h5")[0] + "_" + str(clip_idx)    #生成处理后的数据文件的名称

            if os.path.exists(
                os.path.join(self.processed_dir, "{}.pt".format(writeout_fn))
            ):
                continue

            with h5py.File(os.path.join(self.raw_data_path, h5_file_name), "r") as hf:     #打开文件
                x = hf["resampled_signal"][()]                                             # x = (num_nodes, time * freq) 下采样后的数据
                total_length = x.shape[1]  # 获取数据总长度
                print(x.shape)
            time_start_idx = clip_idx * int(FREQ * self.seq_len)                           #片段开始时间
            time_end_idx = time_start_idx + int(FREQ * self.seq_len)                       #片段结束时间
            print(time_start_idx,time_end_idx)
            if time_end_idx > total_length:
                print(f"跳过 {h5_file_name} 的片段 {clip_idx}（结束位置 {time_end_idx} > 总长度 {total_length}）")
                continue  # 直接跳过这个片段
            x = x[:, time_start_idx:time_end_idx]         # (num_nodes, seq_len*freq) 当前剪辑片段的数据

            print(f"Debug - x.shape: {x.shape}, FREQ: {FREQ}, seq_len: {self.seq_len}")  # 添加这行 看x为什么不匹配。
            assert x.shape[1] == FREQ * self.seq_len
            x = np.expand_dims(x, axis=-1)  # (num_nodes, seq_len*freq, 1)  在 x 的最后一个维度上增加一个维度

            # get edge index
            adj_mat = self._get_combined_graph()                                #获取邻接矩阵 combined graph
            edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(    #将邻接矩阵转换为稀疏表示形式，得到边索引 edge_index 和边权重 edge_weight。
                torch.FloatTensor(adj_mat)
            )

            # pyg graph
            x = torch.FloatTensor(x)  # (num_nodes, seq_len*freq, 1)
         #   y = torch.FloatTensor([y])
            y = torch.LongTensor([int(y)])  # 确保是整数类型
            data = Data(
                x=x,         
                edge_index=edge_index.contiguous(),
                edge_attr=edge_weight,
                y=y,
                adj_mat=torch.FloatTensor(adj_mat).unsqueeze(0),
            )#创建一个 Data 对象，包含节点特征 x、边索引 edge_index、边权重 edge_attr、标签 y 和邻接矩阵 adj_mat。表示一个图数据样本。

            data.writeout_fn = writeout_fn                      

            torch.save( 
                data,
                os.path.join(self.processed_dir, "{}.pt".format(writeout_fn)),
            )  #将 data 对象保存到 self.processed_dir 目录中

    def get(self, idx):

        h5_file_name = self.file_names[idx]
        y = self.labels[idx]
        clip_idx = int(self.df_file.iloc[idx]["clip_index"])

        writeout_fn = h5_file_name.split(".h5")[0] + "_" + str(clip_idx)
    
        data = torch.load(os.path.join(self.processed_dir, "{}.pt".format(writeout_fn)))

        if self.scaler is not None:
            # standardize
            data.x = self.scaler.transform(data.x)

        data.x = data.x.float()

        return data


class TUH_DataModule(pl.LightningDataModule):              #用于管理TUH数据集的加载和预处理。
    def __init__(
        self,
        raw_data_path,
        preproc_save_dir,
        seq_len,                                           #每个样本的时间窗口大小
        num_nodes,
        train_batch_size,
        test_batch_size,
        num_workers,
        adj_mat_dir="/local2/zhangxinyu/graphs4mer/graphs4mer-main/data/eeg_electrode_graph/adj_mx_3d.pkl",            #预定义邻接矩阵路径
        standardize=True,
        balanced_sampling=False,
        pin_memory=False,
    ):
        super().__init__()

        self.raw_data_path = raw_data_path
        self.preproc_save_dir = preproc_save_dir
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.adj_mat_dir = adj_mat_dir
        self.standardize = standardize
        self.balanced_sampling = balanced_sampling
        self.pin_memory = pin_memory

        self.file_markers = {}
        for split in ["train", "val", "test"]:
            self.file_markers[split] = pd.read_csv(
                os.path.join(
                    FILEMARKER_DIR, "{}_file_markers_{}s.csv".format(split, seq_len),
                ),
            )

        if standardize:                                  #设置为true才执行    在154行                     
            train_files = list(set(self.file_markers["train"]["file_name"].tolist()))
            train_files = [os.path.join(raw_data_path, fn) for fn in train_files]
            self.mean, self.std = self._compute_mean_std(
                train_files, num_nodes=num_nodes
            )                                             #计算训练数据的均值和标准差
            print("mean:", self.mean.shape)

            self.scaler = StandardScaler(mean=self.mean, std=self.std)
        else:
            self.scaler = None

        self.train_dataset = TUHDataset(
            root=self.preproc_save_dir,
            raw_data_path=self.raw_data_path,
            file_marker=self.file_markers["train"],
            split="train",
            seq_len=self.seq_len,
            num_nodes=self.num_nodes,
            adj_mat_dir=self.adj_mat_dir,
            scaler=self.scaler,
            transform=None,
            pre_transform=None,
        )

        self.val_dataset = TUHDataset(
            root=self.preproc_save_dir,
            raw_data_path=self.raw_data_path,
            file_marker=self.file_markers["val"],
            split="val",
            seq_len=self.seq_len,
            num_nodes=self.num_nodes,
            adj_mat_dir=self.adj_mat_dir,
            scaler=self.scaler,
            transform=None,
            pre_transform=None,
        )

        self.test_dataset = TUHDataset(
            root=self.preproc_save_dir,
            raw_data_path=self.raw_data_path,
            file_marker=self.file_markers["test"],
            split="test",
            seq_len=self.seq_len,
            num_nodes=self.num_nodes,
            adj_mat_dir=self.adj_mat_dir,
            scaler=self.scaler,
            transform=None,
            pre_transform=None,
        )

    def train_dataloader(self):

        if self.balanced_sampling:
            num_pos = torch.sum(self.train_dataset.get_labels() == 1)
            sampler = ImbalancedDatasetSampler(
                dataset=self.train_dataset,
                num_samples=num_pos * 2,
                replacement=False,
            )
            shuffle = False

        else:
            # 即使不balance，也确保每个batch包含少数类
            minority_indices = [i for i, y in enumerate(self.train_dataset.labels) if y in [2,3]]
            sampler = torch.utils.data.RandomSampler(
                indices = list(range(len(self.train_dataset))) + minority_indices * 5  # 过采样
            )
            #sampler = None
            shuffle = True

        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            sampler=sampler,
            shuffle=shuffle,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )
        return train_dataloader

    def val_dataloader(self):

        val_dataloader = DataLoader(
            dataset=self.val_dataset,
            shuffle=False,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )
        return val_dataloader

    def test_dataloader(self):

        test_dataloader = DataLoader(
            dataset=self.test_dataset,
            shuffle=False,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )
        return test_dataloader

    def _compute_mean_std(self, train_files, num_nodes=19):
        if ".h5" in train_files[0]:
            count = 0
            signal_sum = np.zeros((num_nodes))
            signal_sum_sqrt = np.zeros((num_nodes))
            print("Computing mean and std of training data...")
            for idx in tqdm(range(len(train_files))):
                with h5py.File(train_files[idx], "r") as hf:
                    signal = hf["resampled_signal"][()]  # (num_nodes, time * freq)
                signal_sum += signal.sum(axis=-1)
                signal_sum_sqrt += (signal**2).sum(axis=-1)
                count += signal.shape[-1]
            total_mean = signal_sum / count
            total_var = (signal_sum_sqrt / count) - (total_mean**2)
            total_std = np.sqrt(total_var)
        else:
            raise NotImplementedError

        return np.expand_dims(np.expand_dims(total_mean, -1), -1), np.expand_dims(
            np.expand_dims(total_std, -1), -1
        )

    def teardown(self, stage=None):
        # clean up after fit or test
        # called on every process in DDP
        pass