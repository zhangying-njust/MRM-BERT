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


def mypred(models, data_path, s):

    [seq_name, each_len, seq, _] = read_test_txt_to01_to0123(data_path + jobid + ".txt")
    y_score_all = np.empty((len(seq), 0))

    for modelpath in models:
        model = zyClassifier2.from_pretrained(modelpath)
        tokenizer = DNATokenizer.from_pretrained(modelpath, do_lower_case=False)
        dataset = load_and_cache_examples(data_path, tokenizer)

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
        probs = softmax(torch.tensor(preds, dtype=torch.float32))[:, 1].numpy()

        y_score_all = np.concatenate((y_score_all, probs[:,np.newaxis]), axis=1)

    # save results
    if (s == 'mm'):
        with open(data_path + 'results' + jobid + '.txt', "a") as fd:
            fd.write('# position sequence nucleotide pseu m1A m5C m6A' + '\n')
            global_count = 0

            check = ['T','A','C','A']
            dict = {'0': 'pseudouridine', '1': 'm1A', '2': 'm5C', '3': 'm6A'}

            for ii in range(len(seq_name)):
                fd.write(seq_name[ii] + '\n')
                count = 0
                while count < each_len[ii]-40:
                    the_pos = count + 21  # 从1开始
                    the_nuc = seq[global_count][20]
                    the_possib_4 = y_score_all[global_count, :]


                    # fd.write(str(global_count) + ' '
                    #          + str(the_pos) + ' '
                    #          + str(seq[global_count]) + ' '
                    #          + str(the_nuc) + ' '
                    #          + str(int(the_possib_4[0] * 10000 + 0.5) / 10000) + ' '
                    #          + str(int(the_possib_4[1] * 10000 + 0.5) / 10000) + ' '
                    #          + str(int(the_possib_4[2] * 10000 + 0.5) / 10000) + ' '
                    #          + str(int(the_possib_4[3] * 10000 + 0.5) / 10000) + '\n')

                    if (the_nuc=='T' or the_nuc=='U'):
                        fd.write(str(global_count) + ' '
                                 + str(the_pos) + ' '
                                 + str(seq[global_count]) + ' '
                                 + str(the_nuc) + ' '
                                 + str(int(the_possib_4[0] * 10000 + 0.5) / 10000) + ' '
                                 + str('*') + ' '
                                 + str('*') + ' '
                                 + str('*') + '\n')
                    elif(the_nuc=='A'):
                        fd.write(str(global_count) + ' '
                                 + str(the_pos) + ' '
                                 + str(seq[global_count]) + ' '
                                 + str(the_nuc) + ' '
                                 + str('*') + ' '
                                 + str(int(the_possib_4[1] * 10000 + 0.5) / 10000) + ' '
                                 + str('*') + ' '
                                 + str(int(the_possib_4[3] * 10000 + 0.5) / 10000) + '\n')
                    elif(the_nuc=='C'):
                        fd.write(str(global_count) + ' '
                                 + str(the_pos) + ' '
                                 + str(seq[global_count]) + ' '
                                 + str(the_nuc) + ' '
                                 + str('*') + ' '
                                 + str('*') + ' '
                                 + str(int(the_possib_4[2] * 10000 + 0.5) / 10000) + ' '
                                 + str('*') + '\n')
                    else:
                        fd.write(str(global_count) + ' '
                                 + str(the_pos) + ' '
                                 + str(seq[global_count]) + ' '
                                 + str(the_nuc) + ' '
                                 + str('*') + ' '
                                 + str('*') + ' '
                                 + str('*') + ' '
                                 + str('*') + '\n')

                    global_count = global_count + 1
                    count = count + 1
    elif(s == 'at'):
        with open(data_path + 'results' + jobid + '.txt', "a") as fd:
            fd.write('# position sequence nucleotide pseu m1A m5C m6A' + '\n')
            global_count = 0

            check = ['T', 'C', 'A']
            dict = {'0': 'pseudouridine', '1': 'm5C', '2': 'm6A'}

            for ii in range(len(seq_name)):
                fd.write(seq_name[ii] + '\n')
                count = 0
                while count < each_len[ii]-40:
                    the_pos = count + 21  # 从1开始
                    the_nuc = seq[global_count][20]
                    the_possib_4 = y_score_all[global_count, :]

                    # fd.write(str(global_count) + ' '
                    #          + str(the_pos) + ' '
                    #          + str(seq[global_count]) + ' '
                    #          + str(the_nuc) + ' '
                    #          + str(int(the_possib_4[0] * 10000 + 0.5) / 10000) + ' '
                    #          + str('*') + ' '
                    #          + str(int(the_possib_4[1] * 10000 + 0.5) / 10000) + ' '
                    #          + str(int(the_possib_4[2] * 10000 + 0.5) / 10000) + '\n')

                    if (the_nuc=='T' or the_nuc=='U'):
                        fd.write(str(global_count) + ' '
                                 + str(the_pos) + ' '
                                 + str(seq[global_count]) + ' '
                                 + str(the_nuc) + ' '
                                 + str(int(the_possib_4[0] * 10000 + 0.5) / 10000) + ' '
                                 + str('*') + ' '
                                 + str('*') + ' '
                                 + str('*') + '\n')
                    elif (the_nuc == 'A'):
                        fd.write(str(global_count) + ' '
                                 + str(the_pos) + ' '
                                 + str(seq[global_count]) + ' '
                                 + str(the_nuc) + ' '
                                 + str('*') + ' '
                                 + str('*') + ' '
                                 + str('*') + ' '
                                 + str(int(the_possib_4[2] * 10000 + 0.5) / 10000) + '\n')
                    elif (the_nuc == 'C'):
                        fd.write(str(global_count) + ' '
                                 + str(the_pos) + ' '
                                 + str(seq[global_count]) + ' '
                                 + str(the_nuc) + ' '
                                 + str('*') + ' '
                                 + str('*') + ' '
                                 + str(int(the_possib_4[1] * 10000 + 0.5) / 10000) + ' '
                                 + str('*') + '\n')
                    else:
                        fd.write(str(global_count) + ' '
                                 + str(the_pos) + ' '
                                 + str(seq[global_count]) + ' '
                                 + str(the_nuc) + ' '
                                 + str('*') + ' '
                                 + str('*') + ' '
                                 + str('*') + ' '
                                 + str('*') + '\n')
                    global_count = global_count + 1
                    count = count + 1
    else:
        with open(data_path + 'results' + jobid + '.txt', "a") as fd:
            fd.write('# position sequence nucleotide pseu m1A m5C m6A' + '\n')
            global_count = 0

            check = ['T', 'A', 'A']
            dict = {'0': 'pseudouridine', '1': 'm1A', '2': 'm6A'}

            for ii in range(len(seq_name)):
                fd.write(seq_name[ii] + '\n')
                count = 0
                while count < each_len[ii]-40:
                    the_pos = count + 21  # 从1开始
                    the_nuc = seq[global_count][20]
                    the_possib_4 = y_score_all[global_count, :]

                    # fd.write(str(global_count) + ' '
                    #          + str(the_pos) + ' '
                    #          + str(seq[global_count]) + ' '
                    #          + str(the_nuc) + ' '
                    #          + str(int(the_possib_4[0] * 10000 + 0.5) / 10000) + ' '
                    #          + str(int(the_possib_4[1] * 10000 + 0.5) / 10000) + ' '
                    #          + str('*') + ' '
                    #          + str(int(the_possib_4[2] * 10000 + 0.5) / 10000) + '\n')

                    if (the_nuc=='T' or the_nuc=='U'):
                        fd.write(str(global_count) + ' '
                                 + str(the_pos) + ' '
                                 + str(seq[global_count]) + ' '
                                 + str(the_nuc) + ' '
                                 + str(int(the_possib_4[0] * 10000 + 0.5) / 10000) + ' '
                                 + str('*') + ' '
                                 + str('*') + ' '
                                 + str('*') + '\n')
                    elif (the_nuc == 'A'):
                        fd.write(str(global_count) + ' '
                                 + str(the_pos) + ' '
                                 + str(seq[global_count]) + ' '
                                 + str(the_nuc) + ' '
                                 + str('*') + ' '
                                 + str(int(the_possib_4[1] * 10000 + 0.5) / 10000) + ' '
                                 + str('*') + ' '
                                 + str(int(the_possib_4[2] * 10000 + 0.5) / 10000) + '\n')
                    else:
                        fd.write(str(global_count) + ' '
                                 + str(the_pos) + ' '
                                 + str(seq[global_count]) + ' '
                                 + str(the_nuc) + ' '
                                 + str('*') + ' '
                                 + str('*') + ' '
                                 + str('*') + ' '
                                 + str('*') + '\n')

                    global_count = global_count + 1
                    count = count + 1
    return 0


def network_load(s):
    if(s == 'mm'):
        models = [curPath + "/modelbertft/mm/Y", curPath + "/modelbertft/mm/m1A", curPath + "/modelbertft/mm/m5C", curPath + "/modelbertft/mm/m6A"]
    elif (s == 'at'):
        models = [curPath + "/modelbertft/at/Y", curPath + "/modelbertft/at/m5C", curPath + "/modelbertft/at/m6A"]
    else:
        models = [curPath + "/modelbertft/sc/Y", curPath + "/modelbertft/sc/m1A", curPath + "/modelbertft/sc/m6A"]
    return models

if __name__ == '__main__':

    jobid = 'data_01'
    model_choose = 'mm'  # 'mm''at''sc'

    if jobid.split() != "":

        data_path = rootPath + "/PredictDataOFUsers/"
        models = network_load(model_choose)
        mypred(models, data_path, model_choose)

