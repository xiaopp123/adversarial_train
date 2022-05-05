# -*- coding: utf-8 -*-


import math
import torch
from tqdm import tqdm
from config import *


def build_vocab(file_path, vocab_size, min_freq):
    """创建词表
    :param file_path: 数据文件
    :param vocab_size: 词表最大词个数
    :param min_freq: 最小词频
    :return:
    """
    vocab_dict = dict()
    with open(file_path, 'r') as fr:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            text = line.split('\t')[0]
            for token in text:
                # 词频
                vocab_dict[token] = vocab_dict.get(token, 0) + 1
        vocab_list = sorted([t for t in vocab_dict.items() if t[1] >= min_freq],
                            key=lambda x: x[1], reverse=True)[:vocab_size]
        vocab_index_dict = dict()
        for idx, vocab in enumerate(vocab_list):
            vocab_index_dict[vocab[0]] = idx
        vocab_index_dict.update({UNK: len(vocab_index_dict),
                                 PAD: len(vocab_index_dict) + 1})
        return vocab_index_dict


def build_dataset(file_path, vocab_dict, max_seq_length):
    """
    :param file_path:
    :param vocab_dict:
    :param max_seq_length:
    :return: [[text, label, seq_length]]
    """
    text_list = []
    label_list = []
    seq_length_list = []
    content_list = []
    with open(file_path, 'r') as fr:
        for line in tqdm(fr):
            line = line.strip()
            if not line:
                continue
            text, label = line.split('\t')
            token_list = [t for t in text]
            cur_len = len(token_list)
            if cur_len > max_seq_length:
                token_list = token_list[:max_seq_length]
                seq_length_list.append(max_seq_length)
            else:
                token_list += [PAD] * (max_seq_length - cur_len)
                seq_length_list.append(cur_len)
            text_list.append([vocab_dict.get(t, vocab_dict.get(UNK)) for t in token_list])
            label_list.append(int(label))
    for idx in range(len(text_list)):
        content_list.append(
            [text_list[idx], label_list[idx], seq_length_list[idx]])
    return content_list


class DataSetIterator(object):
    def __init__(self, dataset, batch_size, device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch = math.ceil(len(dataset) / batch_size)
        self.index = 0
        self.device = device

    def __next__(self):
        if self.index >= self.batch:
            self.index = 0
            raise StopIteration
        cur_data = self.dataset[self.index * self.batch_size: (self.index + 1) * self.batch_size]
        self.index += 1
        # print(len(cur_data))
        # print(cur_data)
        text_tensor = torch.LongTensor([t[0] for t in cur_data]).to(self.device)
        label_tensor = torch.LongTensor([t[1] for t in cur_data]).to(self.device)
        seq_len_tensor = torch.LongTensor([t[2] for t in cur_data]).to(self.device)
        return text_tensor, label_tensor, seq_len_tensor

    def __iter__(self):
        return self

