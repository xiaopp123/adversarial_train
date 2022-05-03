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
    parser.add_argument('--attack', type=str, help='attack model')
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
            labels = iter[1].data.cpu().numpy().tolist()
            preds = torch.max(out.data, 1)[1].cpu().numpy().tolist()
            # print(labels, preds)
            # label_list.append()
            label_list += labels
            predict_list += preds
    acc = metrics.accuracy_score(label_list, predict_list)
    print('acc: ', acc)


def main():
    args = get_args()
    adv_init_mag = 0.01
    print(args)
    # 处理数据
    train_file = os.path.join(DATA_PATH, 'THUCNews/train.txt')
    dev_file = os.path.join(DATA_PATH, 'THUCNews/dev.txt')
    test_file = os.path.join(DATA_PATH, 'THUCNews/test.txt')
    class_file = os.path.join(DATA_PATH, 'THUCNews/class.txt')

    class_list = [x.strip() for x in open(class_file, 'r').readlines()]

    vocab_dict = build_vocab(file_path=train_file,
                             vocab_size=VOCAB_SIZE, min_freq=MIN_FREQ)
    train_dataset = build_dataset(file_path=train_file,
                                  max_seq_length=MAX_SEQ_LENGTH,
                                  vocab_dict=vocab_dict)
    dev_dataset = build_dataset(file_path=dev_file,
                                max_seq_length=MAX_SEQ_LENGTH,
                                vocab_dict=vocab_dict)
    # print(len(train_dataset))
    # print(train_dataset)
    train_iterator = DataSetIterator(train_dataset,
                                     batch_size=BATCH_SIZE,
                                     device=device)
    dev_iterator = DataSetIterator(dev_dataset,
                                   batch_size=BATCH_SIZE,
                                   device=device)

    learning_rate = 1e-4
    model = TextCNN(num_classes=len(class_list))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    attack_model = None
    if args.attack == 'fgm':
        attack_model = FGM(model)
    elif args.attack == 'fgsm':
        attack_model = FGSM(model)
    elif args.attack == 'fgd':
        attack_model = PGD(model)
    elif args.attack == 'freelb':
        attack_model = FreeLB(model)
    elif args.attack == 'smart':
        attack_model = Smart(model)

    # freelb
    adv_lr = 1e-4
    adv_max_norm = 1e-4
    epoch_num = 5
    for epoch in range(epoch_num):
        model.train()
        tr_loss = 0
        for idx, iter in enumerate(train_iterator):
            out = model(iter)
            loss = F.cross_entropy(out, iter[1])
            loss.backward()
            # 对抗训练
            if args.attack not in ('fgd', 'freelb', 'smart'):
                attack_model.attack()
                out = model(iter)
                loss_adv = F.cross_entropy(out, iter[1])
                loss_adv.backward()
                attack_model.restore()
            if args.attack == 'fgd':
                pgd_k = 3
                attack_model.backup_grad()
                for t in range(pgd_k):
                    attack_model.attack(is_first_attack=(t == 0))
                    if t != pgd_k - 1:
                        model.zero_grad()
                    else:
                        attack_model.restore_grad()
                    out = model(iter)
                    loss_adv = F.cross_entropy(out, iter[1])
                    loss_adv.backward()
                    attack_model.restore()
            if args.attack == 'freelb':
                embeds_init = attack_model.lookup_emb(input_ids=iter[0])
                # attack_model.attack(delta=None)
                delta = torch.zeros_like(embeds_init).uniform_(
                    -adv_init_mag, adv_init_mag)
                adv_steps = 3
                for astep in range(adv_steps):
                    delta.requires_grad_()
                    input_embeds = delta + embeds_init
                    outputs = model(iter, emb_init=input_embeds)
                    adv_loss = F.cross_entropy(outputs, iter[1])
                    adv_loss /= adv_steps
                    adv_loss.backward(retain_graph=True)

                    if astep == adv_steps - 1:
                        break

                    delta_grad = delta.grad.clone().detach()
                    norm = torch.norm(delta_grad)
                    norm = torch.clamp(norm, min=1e-8)
                    delta = delta + adv_lr * delta_grad / norm
                    if adv_max_norm > 0:
                        delta = torch.clamp(delta, adv_max_norm, adv_max_norm).detach()
            if args.attack == 'smart':
                # embeds_init = attack_model.lookup_emb(input_ids=iter[0])
                # # noise = attack_model.generate_noise(embeds_init)
                # delta = torch.zeros_like(embeds_init).uniform_(
                #     -adv_init_mag, adv_init_mag)
                # adv_steps = 3
                # step_size = 1e-3
                # for astep in range(adv_steps):
                #     print(astep)
                #     delta.requires_grad_()
                #     input_embeds = delta + embeds_init
                #     outputs = model(iter, emb_init=input_embeds)
                #     adv_loss = attack_model.stable_kl(out, iter[1])
                #     # adv_loss = F.cross_entropy(outputs, iter[1])
                #     adv_loss.backward(retain_graph=True)
                #
                #     delta_grad = delta.grad.clone().detach()
                #     norm = torch.norm(delta_grad)
                #     eff_delta_grad = delta_grad * step_size
                #     delta = delta + delta_grad * step_size / norm
                #     delta = torch.clamp(delta, adv_max_norm, adv_max_norm).detach()

                    # delta.detach()
                embeds_init = attack_model.lookup_emb(input_ids=iter[0])
                # attack_model.attack(delta=None)
                delta = torch.zeros_like(embeds_init).uniform_(
                    -adv_init_mag, adv_init_mag)
                adv_steps = 3
                for astep in range(adv_steps):
                    delta.requires_grad_()
                    input_embeds = delta + embeds_init
                    outputs = model(iter, emb_init=input_embeds)
                    # adv_loss = F.cross_entropy(outputs, iter[1])
                    adv_loss = attack_model.stable_kl(logit=outputs, target=iter[1])
                    adv_loss.backward(retain_graph=True)
                    delta_grad = delta.grad.clone().detach()
                    norm = torch.norm(delta_grad)
                    # norm = torch.clamp(norm, min=1e-8)
                    delta = delta + adv_lr * delta_grad / norm
                    delta = delta.detach()
                    # if adv_max_norm > 0:
                    #    delta = torch.clamp(delta, adv_max_norm, adv_max_norm).detach()

            optimizer.step()
            model.zero_grad()
            tr_loss += loss.item()

            if idx % 300 == 0:
                print('epoch {}, iter {}: loss is {}'.format(epoch + 1, idx, loss.item()))

        print('epoch {}: loss is {}'.format(epoch + 1, tr_loss))
        evaluate(model, dev_iterator)


if __name__ == '__main__':
    main()
