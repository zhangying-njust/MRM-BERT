import csv
import json
import os
import sys
from torch import nn
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import AdamW,  get_linear_schedule_with_warmup
from transformers import BertConfig, BertModel, BertForSequenceClassification, DNATokenizer,BertForMaskedLM,BertModel, BertTokenizer
from transformers import BertPreTrainedModel
# from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import copy
from data_processing_mult import read_test_txt_to01_to0123
class zyClassifier2(BertPreTrainedModel):
    """
    Bert Model with an additional simple classification head.
    Can be used to fine tune BERT's [CLS] output for text classification tasks.
    :param path: str, path to pre-trained BERT model.
    :param num_classes: the number of output classes.
    :param freeze_bert: bool, True if BERT's weights are frozen. Set `False` to fine-tune the BERT model
    """

    def __init__(self, config):

        super(zyClassifier2, self).__init__(config)
        # specify hidden size of BERT, hidden size of the classifier, and number of output labels

        self.num_labels = 2
        # load BERT model
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, self.config.num_labels)
        )
        self.init_weights()


    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,):
        """
        forward pass for the joint BERT-classifier model.
        :param batch: tensor of shape (batch_size, max_length) of token ids.
        :returns logits: tensor of shape (batch_size, num_labels)
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]# 本质是第一个符号池化后

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)
        outputs = (logits,) + outputs[2:]

        return outputs


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def read_tsv(input_file, quotechar=None):
    with open(input_file, "r", encoding="utf-8-sig") as f:
        return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

def create_examples(lines, set_type):
    examples = []
    for (i, line) in enumerate(lines):

        guid = "%s-%s" % (set_type, i)
        text_a = line[0]
        label = '0'
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples
def seq2kmer(seq, k):
    kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    kmers = " ".join(kmer)
    return kmers
def get_dev_examples(data_dir):

    [seq_name, each_len, seq, _] = read_test_txt_to01_to0123(data_dir + jobid + ".txt")
    num_of_seq = len(seq)
    seq_cut_kmer = []
    for ii in range(num_of_seq):
        p = seq2kmer(seq[ii], 6)
        seq_cut_kmer.append([p])


    return create_examples(seq_cut_kmer, "dev")

def load_and_cache_examples(data_path, tokenizer):
    processor = processors["dnaprom"]()
    output_mode = "classification"
    # Load data features from cache or dataset file
    label_list = processor.get_labels()



    examples = get_dev_examples(data_path)



    max_length = 38
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    features = convert_examples_to_features(
        examples,
        tokenizer,
        label_list=label_list,
        max_length=max_length,
        output_mode=output_mode,
        pad_on_left=False,  # pad on the left for xlnet
        pad_token=pad_token,
        pad_token_segment_id=False, )
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
    return dataset

def seq2seqv(s):
    M = np.array(['A', 'C', 'G', 'T'])
    l = len(s)
    outp = []
    outp.append(s)

    ZiRanShu0 = np.linspace(0, 19, 20).astype(int)
    ZiRanShu1 = np.linspace(21, 40, 20).astype(int)
    Lianjie00 = np.hstack((ZiRanShu0, ZiRanShu1)).tolist()

    for ii in Lianjie00:
        orinal = s[ii]
        [index] = np.where(M == orinal)
        fd = np.delete(M, index, axis=0)

        new = list(s)
        for jj in range(3):
            new[ii] = fd[jj]
            t = ''.join(new)
            outp.append(t)
    return outp

def load_and_cache_examples2(data_path, tokenizer):
    processor = processors["dnaprom"]()
    output_mode = "classification"
    # Load data features from cache or dataset file
    label_list = processor.get_labels()

    examples = get_dev_examples2(data_path)

    max_length = 38
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    features = convert_examples_to_features(
        examples,
        tokenizer,
        label_list=label_list,
        max_length=max_length,
        output_mode=output_mode,
        pad_on_left=False,  # pad on the left for xlnet
        pad_token=pad_token,
        pad_token_segment_id=False, )
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
    return dataset
def get_dev_examples2(data_dir):

    [seq_name, each_len, seq, _] = read_test_txt_to01_to0123(data_dir + jobid + ".txt")

    seq = seq2seqv(seq[0])

    num_of_seq = len(seq)
    seq_cut_kmer = []
    for ii in range(num_of_seq):
        p = seq2kmer(seq[ii], 6)
        seq_cut_kmer.append([p])


    return create_examples(seq_cut_kmer, "dev")


def get_activations(modelpath, data_path, s):


    [seq_name, each_len, seq, _] = read_test_txt_to01_to0123(data_path + jobid + ".txt")

    config = BertConfig.from_pretrained(modelpath, num_labels=2, finetuning_task="dnaprom",cache_dir=None)
    config.output_attentions = True
    model = zyClassifier2.from_pretrained(modelpath, from_tf=0, config=config, cache_dir=None, )
    tokenizer = DNATokenizer.from_pretrained(modelpath, do_lower_case=False, cache_dir=None)

    dataset = load_and_cache_examples(data_path, tokenizer)


    # Note that DistributedSampler samples randomly
    pred_dataloader = DataLoader(dataset, batch_size=20, shuffle=False)

    kmer=6


    batch_size = 20

    preds = np.zeros([len(dataset), 2])

    attention_scores = np.zeros([len(dataset), 12, 38, 38])

    for index, batch in enumerate(tqdm(pred_dataloader, desc="Predicting")):
        model.eval()
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            inputs["token_type_ids"] = (None)
            outputs = model(**inputs)
            # outputs是tuple,选择后一个,表示head
            # attention = outputs[-1][-1]
            attention = outputs[-1][-1]
            logits = outputs[0]

            preds[index * batch_size:index * batch_size + len(batch[0]), :] = logits.detach().cpu().numpy()
            attention_scores[index * batch_size:index * batch_size + len(batch[0]), :, :, :] = attention.cpu().numpy()

        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(torch.tensor(preds, dtype=torch.float32))[:, 1].numpy()

    scores = np.zeros([attention_scores.shape[0], 41])

    for index, attention_score in enumerate(attention_scores):
        attn_score = []
        for i in range(1, attention_score.shape[-1] - 1):
            # attn_score.append(float(attention_score[:, 0, i].sum()))
            attn_score.append(float(attention_score[:, 0,
                                    i].sum()))

        counts = np.zeros([len(attn_score) + kmer - 1])
        real_scores = np.zeros([len(attn_score) + kmer - 1])
        for i, score in enumerate(attn_score):
            for j in range(kmer):
                counts[i + j] += 1.0
                real_scores[i + j] += score
        real_scores = real_scores / counts

        # real_scores = real_scores / np.linalg.norm(real_scores)
        real_scores = real_scores / real_scores.sum()

        scores[index] = real_scores

    with open(data_path + 'attentions' + jobid + '.txt', "a") as fd:
        fd.write('Nucleotide' + ' ' + 'Attention' + '\n')
        for ii in range(len(seq[0])):
            fd.write(seq[0][ii] + ' ' + str(int(scores[0,ii]* 10000 + 0.5) / 10000) + '\n')

    return scores, probs

def myanalyze(modelpath, data_path, model_choose):
    [seq_name, each_len, seq, _] = read_test_txt_to01_to0123(data_path + jobid + ".txt")
    model = zyClassifier2.from_pretrained(modelpath)
    tokenizer = DNATokenizer.from_pretrained(modelpath, do_lower_case=False)
    dataset = load_and_cache_examples2(data_path, tokenizer)

    softmax = torch.nn.Softmax(dim=1)
    pred_dataloader = DataLoader(dataset, batch_size=20, shuffle=False)
    preds = None
    for batch in tqdm(pred_dataloader, desc="Predicting"):
        model.eval()
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            inputs["token_type_ids"] = (None)
            outputs = model(**inputs)
            logits = outputs[0]
        if preds is None:
            preds = logits.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
    y_score_all = softmax(torch.tensor(preds, dtype=torch.float32))[:, 1].numpy()

    with open(data_path + 'mutationResults' + jobid + '.txt', "a") as fd:
        fd.write(seq_name[0] + ', Orders for mutation Results: A->CGT,C->AGT,G->ACT,T->ACG' + '\n')

        check = ['T', 'A', 'C', 'A']
        dict = {'0': 'pseudouridine', '1': 'm1A', '2': 'm5C', '3': 'm6A'}

        # the_nuc = seq[0][20]
        modtype = model_choose[2:]
        if modtype is 'Y':
            fd.write('pseudouridine' + '\n')
            fd.write(str(int(y_score_all[0] * 10000 + 0.5) / 10000) + '\n')
            fd.write('position mutationResults' + '\n')
            count = 0
            for ii in range(1, len(y_score_all), 3):
                if (count < 20):
                    {
                        fd.write(str(count + 1) + ' ' + seq[0][count] + ' ' + str(
                            int(y_score_all[ii] * 10000 + 0.5) / 10000) + ' ' + str(
                            int(y_score_all[ii + 1] * 10000 + 0.5) / 10000) + ' ' + str(
                            int(y_score_all[ii + 2] * 10000 + 0.5) / 10000) + '\n')
                    }
                else:
                    {
                        fd.write(str(count + 2) + ' ' + seq[0][count + 1] + ' ' + str(
                            int(y_score_all[ii] * 10000 + 0.5) / 10000) + ' ' + str(
                            int(y_score_all[ii + 1] * 10000 + 0.5) / 10000) + ' ' + str(
                            int(y_score_all[ii + 2] * 10000 + 0.5) / 10000) + '\n')
                    }
                count = count + 1;

        elif modtype is 'm5C':
            fd.write('m5C' + '\n')
            fd.write(str(int(y_score_all[0] * 10000 + 0.5) / 10000) + '\n')
            fd.write('position mutationResults' + '\n')
            count = 0
            for ii in range(1, len(y_score_all), 3):
                if (count < 20):
                    {
                        fd.write(str(count + 1) + ' ' + seq[0][count] + ' ' + str(
                            int(y_score_all[ii] * 10000 + 0.5) / 10000) + ' ' + str(
                            int(y_score_all[ii + 1] * 10000 + 0.5) / 10000) + ' ' + str(
                            int(y_score_all[ii + 2] * 10000 + 0.5) / 10000) + '\n')
                    }
                else:
                    {
                        fd.write(str(count + 2) + ' ' + seq[0][count + 1] + ' ' + str(
                            int(y_score_all[ii] * 10000 + 0.5) / 10000) + ' ' + str(
                            int(y_score_all[ii + 1] * 10000 + 0.5) / 10000) + ' ' + str(
                            int(y_score_all[ii + 2] * 10000 + 0.5) / 10000) + '\n')
                    }
                count = count + 1;

        elif modtype is 'm1A':
            fd.write('m1A' + '\n')
            fd.write(str(int(y_score_all[0] * 10000 + 0.5) / 10000) + '\n')
            fd.write('position mutationResults' + '\n')
            count = 0
            for ii in range(1, len(y_score_all), 3):
                if (count < 20):
                    {
                        fd.write(str(count + 1) + ' ' + seq[0][count] + ' ' + str(
                            int(y_score_all[ii] * 10000 + 0.5) / 10000) + ' ' + str(
                            int(y_score_all[ii + 1] * 10000 + 0.5) / 10000) + ' ' + str(
                            int(y_score_all[ii + 2] * 10000 + 0.5) / 10000) + '\n')
                    }
                else:
                    {
                        fd.write(str(count + 2) + ' ' + seq[0][count + 1] + ' ' + str(
                            int(y_score_all[ii] * 10000 + 0.5) / 10000) + ' ' + str(
                            int(y_score_all[ii + 1] * 10000 + 0.5) / 10000) + ' ' + str(
                            int(y_score_all[ii + 2] * 10000 + 0.5) / 10000) + '\n')
                    }
                count = count + 1;
        else:
            fd.write('m6A' + '\n')
            fd.write(str(int(y_score_all[0] * 10000 + 0.5) / 10000) + '\n')
            fd.write('position mutationResults' + '\n')
            count = 0
            for ii in range(1, len(y_score_all), 3):
                if (count < 20):
                    {
                        fd.write(str(count + 1) + ' ' + seq[0][count] + ' ' + str(
                            int(y_score_all[ii] * 10000 + 0.5) / 10000) + ' ' + str(
                            int(y_score_all[ii + 1] * 10000 + 0.5) / 10000) + ' ' + str(
                            int(y_score_all[ii + 2] * 10000 + 0.5) / 10000) + '\n')
                    }
                else:
                    {
                        fd.write(str(count + 2) + ' ' + seq[0][count + 1] + ' ' + str(
                            int(y_score_all[ii] * 10000 + 0.5) / 10000) + ' ' + str(
                            int(y_score_all[ii + 1] * 10000 + 0.5) / 10000) + ' ' + str(
                            int(y_score_all[ii + 2] * 10000 + 0.5) / 10000) + '\n')
                    }
                count = count + 1;


    return 0

def network_load(s):

    mydict = {'scY': curPath + "/modelbertft/sc/Y",
            'scm1A': curPath + "/modelbertft/sc/m1A",
            'scm6A': curPath + "/modelbertft/sc/m6A",
            'mmY': curPath + "/modelbertft/mm/Y",
            'mmm1A': curPath + "/modelbertft/mm/m1A",
            'mmm5C': curPath + "/modelbertft/mm/m5C",
            'mmm6A': curPath + "/modelbertft/mm/m6A",
            'atY': curPath + "/modelbertft/at/Y",
            'at5C': curPath + "/modelbertft/at/m5C",
            'atm6A': curPath + "/modelbertft/at/m6A",
            }

    return mydict[s]

if __name__ == '__main__':


    jobid = 'data_02'
    model_choose = 'mmm6A'

    if jobid.split() != "":

        data_path = rootPath + "/PredictDataOFUsers/"
        modelpath = network_load(model_choose)
        get_activations(modelpath, data_path, model_choose)
        myanalyze(modelpath, data_path, model_choose)




