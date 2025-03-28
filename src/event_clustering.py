from transformers import BertTokenizer, BertModel
import torch
import tqdm
import csv
import json
import jieba
import argparse
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import numpy as np
import re
try:
    import cPickle as pickle
except ImportError:
    import pickle
from process_twitter import get_text_dict as get_text_dic_twitter


def read_twitter_data():
    text_dict = get_text_dic_twitter('train')
    pre_path = "../Data/twitter/"
    origin_texts = open(pre_path + 'train_posts.txt', 'r').readlines()

    ids = []
    texts = []
    for i, line in enumerate(origin_texts):
        if i == 0: continue
        post_id = line.split('\t')[0]
        ids.append(post_id)
        texts.append(text_dict[post_id])
    return texts, ids


def read_gossip_data():
    pre_path = "../Data/AAAI_dataset/"
    alldata = []
    file_name = "../Data/AAAI_dataset/gossip_train.csv"
    with open(file_name) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            alldata.append(row)

    ids = []
    texts = []
    for i, line in enumerate(alldata):
        if i == 0: continue  # skip the title line
        ids.append(line[0])
        texts.append(line[1])
    return texts, ids


def read_weibo_data():
    def stopwordslist(filepath='../Data/weibo/stop_words.txt'):
        stopwords = {}
        for line in open(filepath, 'r').readlines():
            line = line.strip()
            stopwords[line] = 1
        return stopwords

    stop_words = stopwordslist()
    pre_path = "../Data/weibo/tweets/"
    file_list = [pre_path + "test_nonrumor.txt", pre_path + "test_rumor.txt", pre_path + "train_nonrumor.txt", pre_path + "train_rumor.txt"]
    id = pickle.load(open("../Data/weibo/train_id.pickle", 'rb'))

    def clean_str_sst(string):
        string = re.sub(u"[，。 :,.；|-“”——_/nbsp+&;@、《》～（）())#O！：【】]", "", string)
        return string.strip().lower()

    ids = []
    texts = []
    for k, f in enumerate(file_list):
        f = open(f, 'r')
        for i, l in enumerate(f.readlines()):
            if (i + 1) % 3 == 1:
                current_id = l.split('|')[0]
                ids.append(current_id)
            if (i + 1) % 3 == 0:
                l = clean_str_sst(l)
                seg_list = jieba.cut_for_search(l)
                new_seg_list = []
                for word in seg_list:
                    if word not in stop_words:
                        new_seg_list.append(word)

                clean_l = " ".join(new_seg_list)
                if len(clean_l) > 10 and current_id in id:  # truncrate small length
                    texts.append(l)
    return texts, ids


def cluster(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.model)
    model = BertModel.from_pretrained(args.model)
    model.to(device)
    model.eval()

    # get text list
    if args.dataset == 'twitter':
        texts, ids = read_twitter_data()
    elif args.dataset == 'gossip':
        texts, ids = read_gossip_data()
    elif args.dataset == 'weibo':
        texts, ids = read_weibo_data()
    else:
        texts, ids = [], []
        ValueError('non-existing dataset name')

    def get_bert_embeddings(texts):
        embeddings = []
        with torch.no_grad():
            for text in tqdm.tqdm(texts):
                inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=False, max_length=128).to(device)
                outputs = model(**inputs)
                hidden_states = outputs.last_hidden_state
                cls_embedding = hidden_states[0, 0, :].cpu().numpy()
                embeddings.append(cls_embedding)
        return np.array(embeddings)

    embeddings = get_bert_embeddings(texts)

    clustering_model = AgglomerativeClustering(n_clusters=args.n_clusters)
    cluster_labels = clustering_model.fit_predict(embeddings)

    id_to_label = {id_: int(label) for id_, label in zip(ids, cluster_labels)}

    with open("event_labels_{}.json".format(args.dataset), "w") as f:
        json.dump(id_to_label, f, indent=4)

    print("event labels have been saved!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='weibo', help='weibo, gossip, twitter')
    parser.add_argument('--n_clusters', type=int, default=4, help='')
    args = parser.parse_args()

    args.model = '../../huggingface/bert-base-chinese' if args.dataset == 'weibo' else '../../huggingface/bert-base-uncased'

    cluster(args=args)
