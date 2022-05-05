# -*- coding: utf-8 -*-


import torch
import torch.nn.functional as F

from tqdm import tqdm

from sklearn import metrics

from config import *
from src.utils.data_util import build_dataset, build_vocab, DataSetIterator
from src.model.text_cnn import TextCNN
from src.model.attack_train import FGM, FGSM, PGD, FreeLB, Smart

import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--attack', type=str, help='attack model',
                        choices=['fgsm', 'fgm', 'pgd', 'freelb', 'smart'])
    args = parser.parse_args()
    return args


def evaluate(model, dev_iterator):
    model.eval()
    loss_total = 0
    label_list = []
    predict_list = []
    with torch.no_grad():
        for idx, iter in tqdm(enumerate(dev_iterator)):
            out = model(iter)
            # 真实标签
            labels = iter[1].data.cpu().numpy().tolist()
            # 模型预估标签
            preds = torch.max(out.data, 1)[1].cpu().numpy().tolist()
            label_list += labels
            predict_list += preds
    acc = metrics.accuracy_score(label_list, predict_list)
    p = metrics.precision_score(label_list, predict_list, average='weighted')
    r = metrics.recall_score(label_list, predict_list, average='weighted')
    f1 = metrics.f1_score(label_list, predict_list, average='weighted')
    print('acc: {}, p: {}, r: {}, f1: {}'.format(acc, p, r, f1))


def main():
    args = get_args()
    print(args)
    # 处理数据
    train_file = os.path.join(DATA_PATH, 'THUCNews/train.txt')
    dev_file = os.path.join(DATA_PATH, 'THUCNews/dev.txt')
    test_file = os.path.join(DATA_PATH, 'THUCNews/test.txt')
    class_file = os.path.join(DATA_PATH, 'THUCNews/class.txt')

    # 标签集合
    class_list = [x.strip() for x in open(class_file, 'r').readlines()]

    # 词表
    vocab_dict = build_vocab(file_path=train_file,
                             vocab_size=VOCAB_SIZE, min_freq=MIN_FREQ)
    # 构建训练集合
    train_dataset = build_dataset(file_path=train_file,
                                  max_seq_length=MAX_SEQ_LENGTH,
                                  vocab_dict=vocab_dict)
    dev_dataset = build_dataset(file_path=dev_file,
                                max_seq_length=MAX_SEQ_LENGTH,
                                vocab_dict=vocab_dict)
    # 根据batch_size生成迭代器
    train_iterator = DataSetIterator(train_dataset,
                                     batch_size=BATCH_SIZE,
                                     device=device)
    dev_iterator = DataSetIterator(dev_dataset,
                                   batch_size=BATCH_SIZE,
                                   device=device)

    # 构建TextCNN模型
    model = TextCNN(num_classes=len(class_list))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    attack_model = None
    if args.attack == 'fgm':
        attack_model = FGM(model)
    elif args.attack == 'fgsm':
        attack_model = FGSM(model)
    elif args.attack == 'pgd':
        attack_model = PGD(model)
    elif args.attack == 'freelb':
        attack_model = FreeLB(model)
    elif args.attack == 'smart':
        attack_model = Smart(model)
    print('args.attack: ', args.attack)

    # freelb的相关参数
    adv_init_mag = 0.01
    adv_lr = 1e-4
    adv_max_norm = 1e-4
    for epoch in range(EPOCHS):
        model.train()
        tr_loss = 0
        for idx, iter in enumerate(train_iterator):
            out = model(iter)
            loss = F.cross_entropy(out, iter[1])
            loss.backward()
            # 对抗训练
            if args.attack in ('fgsm', 'fgm'):
                # 对输入embedding添加对抗扰动形成对抗样本
                attack_model.attack()
                # 使用对抗样本训练
                out = model(iter)
                # 梯度回传
                loss_adv = F.cross_entropy(out, iter[1])
                loss_adv.backward()
                # 恢复原始embedding
                attack_model.restore()
            if args.attack == 'pgd':
                # 缓存模型中所有参数梯度
                attack_model.backup_grad()
                for t in range(ADV_STEPS):
                    # 对输入embedding添加对抗扰动形成对抗样本
                    attack_model.attack(is_first_attack=(t == 0))
                    if t != ADV_STEPS - 1:
                        # 不是最后一次，需要计算梯度
                        model.zero_grad()
                    else:
                        # 最后一次，恢复模型中所有参数原来梯度，
                        # 即只用最后一次的对抗样本来更新参数
                        attack_model.restore_grad()
                    # 使用对抗样本训练, 并回传梯度,
                    out = model(iter)
                    loss_adv = F.cross_entropy(out, iter[1])
                    loss_adv.backward()
                # 恢复原始embedding
                attack_model.restore()
            if args.attack == 'freelb':
                # 输入原始embedding
                embeds_init = attack_model.lookup_emb(input_ids=iter[0])
                # 扰动初始化
                delta = torch.zeros_like(embeds_init).uniform_(
                    -adv_init_mag, adv_init_mag)
                for astep in range(ADV_STEPS):
                    delta.requires_grad_()
                    # 对抗样本
                    input_embeds = delta + embeds_init
                    outputs = model(iter, emb_init=input_embeds)
                    adv_loss = F.cross_entropy(outputs, iter[1])
                    # 求平均
                    adv_loss /= ADV_STEPS
                    # 每次对抗，模型参数梯度都更新
                    adv_loss.backward(retain_graph=True)
                    # 最后一步直接跳出即可，因为下面是求下一步扰动
                    if astep == ADV_STEPS - 1:
                        break
                    # delta的梯度
                    delta_grad = delta.grad.clone().detach()
                    # norm
                    norm = torch.norm(delta_grad)
                    norm = torch.clamp(norm, min=1e-8)
                    # 更新delta
                    delta = delta + adv_lr * delta_grad / norm
                    # delta约束空间
                    if adv_max_norm > 0:
                        delta = torch.clamp(delta, adv_max_norm, adv_max_norm).detach()
            if args.attack == 'smart':
                # 输入原始embedding
                embeds_init = attack_model.lookup_emb(input_ids=iter[0])
                # 扰动初始化
                delta = torch.zeros_like(embeds_init).uniform_(
                    -adv_init_mag, adv_init_mag)
                for astep in range(ADV_STEPS):
                    delta.requires_grad_()
                    # 对抗样本
                    input_embeds = delta + embeds_init
                    outputs = model(iter, emb_init=input_embeds)
                    # KL散度作为对抗损失
                    adv_loss = attack_model.stable_kl(logit=outputs, target=iter[1])
                    adv_loss.backward(retain_graph=True)
                    # delta梯度
                    delta_grad = delta.grad.clone().detach()
                    # norm
                    norm = torch.norm(delta_grad)
                    # 更新扰动
                    delta = delta + adv_lr * delta_grad / norm
                    if adv_max_norm > 0:
                        delta = torch.clamp(delta, adv_max_norm, adv_max_norm).detach()
            # 参数更新
            optimizer.step()
            model.zero_grad()
            tr_loss += loss.item()

            if idx % 300 == 0:
                print('epoch {}, iter {}: loss is {}'.format(epoch + 1, idx, loss.item()))

        print('epoch {}: loss is {}'.format(epoch + 1, tr_loss))
        evaluate(model, dev_iterator)


if __name__ == '__main__':
    main()
