import numpy as np
import argparse
import math
import logging
import os, sys
from time import strftime, localtime
import tqdm
import random
import process_twitter as process_data_twitter
import process_weibo as process_data_weibo
import process_gossipcop as process_data_gossipcop
import copy
import pickle as pickle
from random import sample
import torchvision
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

import torchvision.datasets as dsets
import torchvision.transforms as transformss

# from logger import Logger

from sklearn import metrics
from transformers import BertModel, BertTokenizer
from model_RBmodel import CNN_Fusion as rbmodel
from model_CAFE import CNN_Fusion as cafe
from model_BMR import CNN_Fusion as bmr
from model_MCAN import CNN_Fusion as mcan
from model_SAFE import CNN_Fusion as safe
from model_GAMED import CNN_Fusion as gamed

from model_RBmodel import ModelwithMoE as rbmodelmoe
from model_CAFE import ModelwithMoE as cafemoe
from model_BMR import ModelwithMoE as bmrmoe
from model_MCAN import ModelwithMoE as mcanmoe
from model_SAFE import ModelwithMoE as safemoe
from model_GAMED import ModelwithMoE as gamedmoe
from loss import ContrastiveLoss

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Rumor_Data(Dataset):
    def __init__(self, dataset):
        self.text = torch.from_numpy(np.array(dataset['post_text']))
        self.image = list(dataset['image'])
        # self.social_context = torch.from_numpy(np.array(dataset['social_feature']))
        self.mask = torch.from_numpy(np.array(dataset['mask']))
        self.label = torch.from_numpy(np.array(dataset['label']))
        self.event_label = torch.from_numpy(np.array(dataset['event_label']))
        print('TEXT: %d, Image: %d, label: %d, Event: %d'
              % (len(self.text), len(self.image), len(self.label), len(self.event_label)))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (self.text[idx], self.image[idx], self.mask[idx]), self.label[idx], self.event_label[idx]


def to_var(x, device):
    if torch.cuda.is_available():
        x = x.to(device)
    return Variable(x)


def to_np(x):
    return x.data.cpu().numpy()


def select(train, selec_indices):
    temp = []
    for i in range(len(train)):
        print("length is " + str(len(train[i])))
        print(i)
        # print(train[i])
        ele = list(train[i])
        temp.append([ele[i] for i in selec_indices])
    return temp


def split_train_validation(train, percent):
    whole_len = len(train[0])

    train_indices = (sample(range(whole_len), int(whole_len * percent)))
    train_data = select(train, train_indices)
    print("train data size is " + str(len(train[3])))

    validation = select(train, np.delete(range(len(train[0])), train_indices))
    print("validation size is " + str(len(validation[3])))
    print("train and validation data set has been splited")

    return train_data, validation


class EWC:
    def __init__(self, model, datasets, device):
        """
        initialize EWC object
        :param model: current model
        :param datasets: List dataset of each task
        :param device: device
        """
        self.model = model
        self.datasets = datasets
        self.device = device
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}  # save old parameter
        self._precision_matrices = self._calculate_importance()

    def _calculate_importance(self):
        # calculate Fisher Information Matrix
        print('re calculate the fisher information metrix')
        precision_matrices = {}
        for n, p in self.params.items():
            precision_matrices[n] = torch.zeros_like(p)

        self.model.eval()
        for dataset in self.datasets:
            for data, target, _ in dataset:
                text, image, mask = to_var(data[0], self.device), to_var(data[1], self.device), to_var(data[2], self.device)
                target = to_var(target, self.device)
                self.model.zero_grad()
                output = self.model(text, image, mask)[0]
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()

                for n, p in self.model.named_parameters():
                    if p.grad is not None:
                        precision_matrices[n] += p.grad.data ** 2

        # average Fisher Information Matrix
        precision_matrices = {n: p / sum(len(d) for d in self.datasets) for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model):
        # calculate EWC regularization
        loss = 0
        for n, p in model.named_parameters():
            if n in self._precision_matrices:
                loss += (self._precision_matrices[n] * (p - self.params[n]) ** 2).sum()
        return loss


def main(args):
    logger.info('Building model')
    if args.baseline == 'rbmodel':
        model = rbmodelmoe(args) if args.moe else rbmodel(args)
    elif args.baseline == 'bmr':
        model = bmrmoe(args) if args.moe else bmr(args)
    elif args.baseline == 'mcan':
        model = mcanmoe(args) if args.moe else mcan(args)
    elif args.baseline == 'safe':
        model = safemoe(args) if args.moe else safe(args)
    elif args.baseline == 'gamed':
        model = gamedmoe(args) if args.moe else gamed(args)
    else:
        model = cafemoe(args) if args.moe else cafe(args)

    logger.info('Loading data...')
    train_list, validation, test = load_data(args)
    test_id = test['post_id']

    # construct Dataset class
    # merge small datasets
    train_dataset_list = [Rumor_Data(train) for train in train_list]
    train_dataset_list = [dataset for dataset in train_dataset_list if len(dataset) >= 2 * args.batch_size]  # filter small-event Datasets
    # train_dataset = Rumor_Data(train)
    validate_dataset = Rumor_Data(validation)
    test_dataset = Rumor_Data(test)

    # Data Loader (Input Pipeline)
    train_loader_list = [DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.number_workers, shuffle=True, drop_last=True)
                         for train_dataset in train_dataset_list]
    validate_loader = DataLoader(dataset=validate_dataset, batch_size=args.batch_size, num_workers=args.number_workers, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.number_workers, shuffle=False)

    if torch.cuda.is_available():
        print("CUDA")
        # model.cuda()
        model.to(args.device)

    loader_size = sum([len(train_loader) for train_loader in train_loader_list])
    logger.info("loader size " + str(loader_size))
    best_validate_f1 = 0.000
    early_stop = 0
    best_validate_dir = ''

    contrastiveLoss = ContrastiveLoss(batch_size=args.batch_size, temperature=args.temp)
    logger.info('begin training...')
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    diff_part = ["bertModel.embeddings", "bertModel.encoder"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
            "weight_decay": 0.0,
            "lr": args.bert_lr
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
            "weight_decay": 0.0,
            "lr": args.bert_lr
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
            "weight_decay": 0.0,
            "lr": args.learning_rate
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
            "weight_decay": 0.0,
            "lr": args.learning_rate
        },
    ]
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, eps=args.adam_epsilon)

    total_epoch = 0
    for eventi, train_loader in enumerate(train_loader_list):
        logger.info('Training on {}-th event'.format(eventi))
        num_epochs = min(math.ceil(len(train_loader) / loader_size * 50), 20)

        ewc = EWC(model, train_loader_list[:eventi], args.device) if eventi > 0 else None

        for epoch in range(num_epochs):
            total_epoch += 1
            p = float(total_epoch) / 100
            lr = args.learning_rate / (1. + 10 * p) ** 0.75
            optimizer.lr = lr
            cost_vector = []
            ewc_vector = []
            acc_vector = []

            for i, (train_data, train_labels, event_labels) in enumerate(tqdm.tqdm(train_loader)):
                train_text, train_image, train_mask, train_labels, event_labels = to_var(train_data[0], args.device), to_var(train_data[1], args.device), \
                    to_var(train_data[2], args.device), to_var(train_labels, args.device), to_var(event_labels, args.device)
                optimizer.zero_grad()
                if args.baseline == 'safe':
                    class_outputs, _, image_z, text_z, sim = model(train_text, train_image, train_mask)
                    loss = criterion(class_outputs, train_labels) + args.gamma * contrastiveLoss(image_z, text_z) + criterion(sim, train_labels)
                else:
                    class_outputs, _, image_z, text_z = model(train_text, train_image, train_mask)
                    loss = criterion(class_outputs, train_labels) + args.gamma * contrastiveLoss(image_z, text_z)

                if ewc is not None:
                    ewcloss = ewc.penalty(model)
                    loss += 0.75 * ewcloss
                    ewc_vector.append(ewcloss.item())

                loss.backward()
                optimizer.step()
                _, argmax = torch.max(class_outputs, 1)

                accuracy = (train_labels == argmax.squeeze()).float().mean()
                cost_vector.append(loss.item())
                acc_vector.append(accuracy.item())

            # todo TEST: test the performance of target events
            model.eval()
            validate_acc_vector_temp = []
            for i, (validate_data, validate_labels, event_labels) in enumerate(validate_loader):
                validate_text, validate_image, validate_mask, validate_labels, event_labels = to_var(validate_data[0], args.device), \
                    to_var(validate_data[1], args.device), to_var(validate_data[2], args.device), to_var(validate_labels, args.device), to_var(event_labels, args.device)
                if args.baseline == 'safe':
                    validate_outputs, _, _, _, _ = model(validate_text, validate_image, validate_mask)
                else:
                    validate_outputs, _, _, _ = model(validate_text, validate_image, validate_mask)

                _, validate_argmax = torch.max(validate_outputs, 1)
                if i == 0:
                    validate_score = to_np(validate_outputs.squeeze())
                    validate_pred = to_np(validate_argmax.squeeze())
                    validate_true = to_np(validate_labels.squeeze())
                else:
                    validate_score = np.concatenate((validate_score, to_np(validate_outputs.squeeze())), axis=0)
                    validate_pred = np.concatenate((validate_pred, to_np(validate_argmax.squeeze())), axis=0)
                    validate_true = np.concatenate((validate_true, to_np(validate_labels.squeeze())), axis=0)

            validate_accuracy = metrics.accuracy_score(validate_true, validate_pred)
            validate_f1 = metrics.f1_score(validate_true, validate_pred, average='macro')
            logger.info('Epoch [%d/%d], Loss: %.4f, EWC Loss: %.4f, Train_Acc: %.4f, Validate_Acc: %.4f, Validate_F1: : %.4f.' %
                        (epoch + 1, num_epochs, np.mean(cost_vector), np.mean(ewc_vector), np.mean(acc_vector), validate_accuracy, validate_f1))

            if eventi + 1 == len(train_loader_list):
                best_validate_dir = args.output_file + args.id + '.pkl'
                if validate_f1 > best_validate_f1:
                    early_stop = 0
                    best_validate_f1 = validate_f1
                    if not os.path.exists(args.output_file):
                        os.mkdir(args.output_file)
                    torch.save(model.state_dict(), best_validate_dir)
                else:
                    early_stop += 1
                    if early_stop == args.early_stop_epoch:
                        break

        # todo TEST: test the performance of every event when one event is trained
        model.eval()
        logger.info('After trained on {} event'.format(eventi))
        for test_event_i, test_event_loader in enumerate(train_loader_list):
            for i, (validate_data, validate_labels, event_labels) in enumerate(test_event_loader):
                validate_text, validate_image, validate_mask, validate_labels, event_labels = to_var(
                    validate_data[0], args.device), to_var(validate_data[1], args.device), to_var(validate_data[2], args.device), to_var(validate_labels, args.device), to_var(
                    event_labels, args.device)
                if args.baseline == 'safe':
                    validate_outputs, _, _, _, _ = model(validate_text, validate_image, validate_mask)
                else:
                    validate_outputs, _, _, _ = model(validate_text, validate_image, validate_mask)

                _, validate_argmax = torch.max(validate_outputs, 1)
                if i == 0:
                    validate_score = to_np(validate_outputs.squeeze())
                    validate_pred = to_np(validate_argmax.squeeze())
                    validate_true = to_np(validate_labels.squeeze())
                else:
                    validate_score = np.concatenate((validate_score, to_np(validate_outputs.squeeze())), axis=0)
                    validate_pred = np.concatenate((validate_pred, to_np(validate_argmax.squeeze())), axis=0)
                    validate_true = np.concatenate((validate_true, to_np(validate_labels.squeeze())), axis=0)

            validate_accuracy = metrics.accuracy_score(validate_true, validate_pred)
            validate_f1 = metrics.f1_score(validate_true, validate_pred, average='macro')
            logger.info('Performance of %d event: Validate_Acc: %.4f, Validate_F1: : %.4f.' % (test_event_i, validate_accuracy, validate_f1))


    # Test the Model
    logger.info('testing model')
    if args.baseline == 'rbmodel':
        model = rbmodelmoe(args) if args.moe else rbmodel(args)
    elif args.baseline == 'bmr':
        model = bmrmoe(args) if args.moe else bmr(args)
    elif args.baseline == 'mcan':
        model = mcanmoe(args) if args.moe else mcan(args)
    elif args.baseline == 'safe':
        model = safemoe(args) if args.moe else safe(args)
    elif args.baseline == 'gamed':
        model = gamedmoe(args) if args.moe else gamed(args)
    else:
        model = cafemoe(args) if args.moe else cafe(args)
    model.load_state_dict(torch.load(best_validate_dir))
    if torch.cuda.is_available():
        # model.cuda()
        model.to(args.device)
    model.eval()
    test_score = []
    test_pred = []
    test_true = []
    for i, (test_data, test_labels, event_labels) in enumerate(test_loader):
        test_text, test_image, test_mask, test_labels = to_var(
            test_data[0], args.device), to_var(test_data[1], args.device), to_var(test_data[2], args.device), to_var(test_labels, args.device)

        if args.baseline == 'safe':
            test_outputs, _, _, _, _ = model(test_text, test_image, test_mask)
        else:
            test_outputs, _, _, _ = model(test_text, test_image, test_mask)

        _, test_argmax = torch.max(test_outputs, 1)
        if i == 0:
            test_score = to_np(test_outputs.squeeze())
            test_pred = to_np(test_argmax.squeeze())
            test_true = to_np(test_labels.squeeze())
        else:
            test_score = np.concatenate((test_score, to_np(test_outputs.squeeze())), axis=0)
            test_pred = np.concatenate((test_pred, to_np(test_argmax.squeeze())), axis=0)
            test_true = np.concatenate((test_true, to_np(test_labels.squeeze())), axis=0)

    test_accuracy = metrics.accuracy_score(test_true, test_pred)
    test_f1 = metrics.f1_score(test_true, test_pred, average='macro')
    test_precision = metrics.precision_score(test_true, test_pred, average='macro')
    test_recall = metrics.recall_score(test_true, test_pred, average='macro')
    test_score_convert = [x[1] for x in test_score]
    test_aucroc = metrics.roc_auc_score(test_true, test_score_convert, average='macro')

    test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)

    logger.info("Classification Acc: %.4f, AUC-ROC: %.4f" % (test_accuracy, test_aucroc))
    logger.info("Classification report:\n%s\n" % (metrics.classification_report(test_true, test_pred, digits=4)))
    logger.info("Classification confusion matrix:\n%s\n" % (test_confusion_matrix))


def get_top_post(output, label, test_id, top_n=500):
    filter_output = []
    filter_id = []

    for i, l in enumerate(label):
        if np.argmax(output[i]) == l and int(l) == 1:
            filter_output.append(output[i][1])
            filter_id.append(test_id[i])

    filter_output = np.array(filter_output)

    top_n_indice = filter_output.argsort()[-top_n:][::-1]

    top_n_id = np.array(filter_id)[top_n_indice]
    top_n_id_dict = {}
    for i in top_n_id:
        top_n_id_dict[i] = True

    pickle.dump(top_n_id_dict, open("../Data/weibo/top_n_id.pickle", "wb"))

    return top_n_id


def re_tokenize_sentence(flag, max_length, tokenizer):
    tokenized_texts = []
    original_texts = flag['original_post']
    for sentence in original_texts:
        tokenized_text = tokenizer.encode(sentence)[:max_length]
        tokenized_texts.append(tokenized_text)
    flag['post_text'] = tokenized_texts


def get_all_text(train_list, validate, test):
    train_text = []
    for train in train_list:
        train_text.extend(list(train['post_text']))
    all_text = train_text + list(validate['post_text']) + list(test['post_text'])
    return all_text


def align_data(flag, args):
    text = []
    mask = []
    for sentence in flag['post_text']:
        sen_embedding = []
        mask_seq = np.zeros(args.sequence_len, dtype=np.float32)
        mask_seq[:len(sentence)] = 1.0
        for i, word in enumerate(sentence):
            sen_embedding.append(word)

        while len(sen_embedding) < args.sequence_len:
            sen_embedding.append(0)

        text.append(copy.deepcopy(sen_embedding))
        mask.append(copy.deepcopy(mask_seq))
    flag['post_text'] = text
    flag['mask'] = mask


def load_data(args):
    train_list, validate, test = args.process_data.get_data(args.text_only, args.continual)
    if args.dataset == 'weibo':
        tokenizer = BertTokenizer.from_pretrained('../../huggingface/bert-base-chinese')
    else:
        tokenizer = BertTokenizer.from_pretrained('../../huggingface/bert-base-uncased')

    for train in train_list:
        re_tokenize_sentence(train, max_length=args.max_length, tokenizer=tokenizer)
    re_tokenize_sentence(validate, max_length=args.max_length, tokenizer=tokenizer)
    re_tokenize_sentence(test, max_length=args.max_length, tokenizer=tokenizer)

    all_text = get_all_text(train_list, validate, test)
    max_len = len(max(all_text, key=len))
    args.sequence_len = max_len

    for train in train_list:
        align_data(train, args)
    align_data(validate, args)
    align_data(test, args)
    return train_list, validate, test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='twitter', help='weibo, gossip, twitter')
    parser.add_argument('--baseline', type=str, default='cafe', help='rbmodel, cafe, bmr, mcan, safe, gamed')
    parser.add_argument('--moe', type=bool, default=False, help='')
    parser.add_argument('--continual', type=bool, default=True, help='')
    parser.add_argument('--static', type=bool, default=True, help='')
    parser.add_argument('--sequence_length', type=int, default=28, help='')
    parser.add_argument('--class_num', type=int, default=2, help='')
    parser.add_argument('--hidden_dim', type=int, default=32, help='')
    parser.add_argument('--embed_dim', type=int, default=32, help='')
    parser.add_argument('--vocab_size', type=int, default=300, help='')
    parser.add_argument('--seed', type=int, default=1, help='')
    parser.add_argument('--gpu', type=int, default=0, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='')
    parser.add_argument('--filter_num', type=int, default=5, help='')
    parser.add_argument('--lambd', type=int, default=1, help='')
    parser.add_argument('--text_only', type=bool, default=False, help='')
    parser.add_argument('--d_iter', type=int, default=3, help='')
    parser.add_argument('--number_workers', type=int, default=4, help='')

    parser.add_argument('--max_length', type=int, default=128, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--num_epochs', type=int, default=100, help='')
    parser.add_argument('--early_stop_epoch', type=int, default=10, help='')

    parser.add_argument('--temp', type=float, default=0.2, help='')
    parser.add_argument('--gamma', type=float, default=0.0, help='corf of pretraining loss')
    parser.add_argument('--balanced', type=float, default=0.01, help='corf of pretraining loss')

    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--bert_lr', type=float, default=0.00003, help='')
    parser.add_argument('--event_num', type=int, default=10, help='')
    parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
    # the road of the dataset is written in process_twitter_changed.py
    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    args.device = "cuda:{}".format(args.gpu)
    args.output_file = '../Data/' + args.dataset + '/RESULT_text_image/'
    args.id = '{}-{}-s{}-{}.log'.format(args.dataset, args.baseline + 'moeewc' if args.moe else args.baseline + 'ewc',
                                        seed, strftime("%m%d-%H%M", localtime()))
    log_file = '../log/{}/'.format(args.dataset) + args.id
    logger.addHandler(logging.FileHandler(log_file))
    if args.dataset == 'gossip':
        args.process_data = process_data_gossipcop
    elif args.dataset == 'weibo':
        args.process_data = process_data_weibo
    else:
        args.process_data = process_data_twitter
    # output arguments into the logger
    logger.info('> training arguments:')
    for arg in vars(args):
        logger.info('>>> {0}: {1}'.format(arg, getattr(args, arg)))

    main(args)


