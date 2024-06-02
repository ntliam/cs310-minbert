import time
import random
import numpy as np
import argparse
import sys
import re
import os
from types import SimpleNamespace
import csv
import json

import math
from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score

# change it with respect to the original model
from tokenizer import BertTokenizer
from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm


TQDM_DISABLE = False
# fix the random seed


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class BertSentimentClassifier(torch.nn.Module):
    '''
    This module performs sentiment classification using BERT embeddings on the SST dataset.

    In the SST dataset, there are 5 sentiment categories (from 0 - "negative" to 4 - "positive").
    Thus, your forward() should return one logit for each of the 5 classes.
    '''

    def __init__(self, config):
        super(BertSentimentClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Pretrain mode does not require updating bert paramters.
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True

        # TODO
        # raise NotImplementedError
        # Drop out layer
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        # Linear layer
        self.linear = torch.nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask):
        '''Takes a batch of sentences and returns logits for sentiment classes'''
        # The final BERT contextualized embedding is the hidden state of [CLS] token (the first token).
        # HINT: you should consider what is the appropriate output to return given that
        # the training loop currently uses F.cross_entropy as the loss function.
        # TODO
        # raise NotImplementedError

        # Get BERT hidden states
        outputs = self.bert(input_ids, attention_mask)
        pooler_output = outputs["pooler_output"]
        pooler_output = self.dropout(pooler_output)
        sentiment_logits = self.linear(pooler_output)

        return sentiment_logits


########### LoRA #############
""" 
Modified from: https://github.com/alexriggio/BERT-LoRA-TensorRT
Credits to: https://github.com/alexriggio/BERT-LoRA-TensorRT
"""


class LinearLoRA(nn.Module):
    """
    A low-rank adapted linear layer. 

    Args:
        in_dim: int = An integer representing the input dimension of the linear layer 
        out_dim: int = An integer representing the output dimension of the linear layer
        r: int = An integer representing the rank of the low-rank approximated matrices
        lora_alpha: int = An integer representing the numerator of the scaling constant alpha / r 
        lora_dropout: float = A float between 0 and 1 representing the dropout probability      
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout)

        # Check that the rank is at least 1
        assert r > 0, "Variable 'r' is not greater than zero. Choose a rank of 1 or greater."

        # recreate the linear layer and freeze it (the actual weight values will be copied in outside of this class)
        self.pretrained = nn.Linear(in_dim, out_dim, bias=True)
        self.pretrained.weight.requires_grad = False

        # create the low-rank A matrix and initialize with same method as in Hugging Face PEFT library
        self.lora_A = nn.Linear(in_dim, r, bias=True)
        # nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)

        # create the low-rank B matrix and initialize to zero
        self.lora_B = nn.Linear(r, out_dim, bias=True)
        nn.init.constant_(self.lora_B.weight, 0)

        # scaling constant
        self.scaling = self.lora_alpha / self.r

    def forward(self, x):
        pretrained_out = self.pretrained(x)
        lora_out = self.lora_dropout(x)
        lora_out = self.lora_A(lora_out)
        lora_out = self.lora_B(lora_out)
        lora_out = lora_out * self.scaling
        return pretrained_out + lora_out


def freeze_model(model):
    """Freezes all layers except the LoRa modules and classifier."""
    for name, param in model.named_parameters():
        if "lora" not in name and "classifier" not in name:
            param.requires_grad = False


def create_lora(module, r, lora_dropout, lora_alpha):
    """Converts a linear module to a LoRA linear module."""
    k, d = module.weight.shape  # pytorch nn.Linear weights are transposed, that is why shape is (k, d) and not (d, k)
    # (768, 768)
    lora = LinearLoRA(d, k, r, lora_dropout=lora_dropout,
                      lora_alpha=lora_alpha)
    with torch.no_grad():
        lora.pretrained.weight.copy_(module.weight)
        lora.pretrained.bias.copy_(module.bias)
    return lora


def add_lora_layers(
    model,
    module_names: Tuple = ("query", "value", "key"),
    r: int = 64,
    lora_alpha: float = 256,
    lora_dropout: float = 0.3,
    ignore_layers: List[int] = []
):
    """
        Replaces chosen linear modules with LoRA equivalents. 

        Args:
            model: torch.nn.Module = The PyTorch model to be used
            module_names: Tuple = A tuple containing the names of the linear layers to replace
                Ex. ("query") to replace the linear modules with "query" in the name --> bert.encoder.layer.0.attention.self.query
            r: int = 
            lora_alpha: int = An integer representing the numerator of the scaling constant alpha / r 
            lora_dropout: float = A float between 0 and 1 representing the dropout probability
            ignore_layers: list = A list with the indices of all BERT layers NOT to add LoRA modules 
        """
    module_types: Tuple = (nn.Linear,)

    # disable dropout in frozen layers
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    # replace chosen linear modules with lora modules
    for name, module in model.named_children():
        if isinstance(module, module_types) and name in module_names:
            temp_lora = create_lora(
                module, r=r, lora_dropout=lora_dropout, lora_alpha=lora_alpha)
            setattr(model, name, temp_lora)
        else:
            ignore_layers_str = [str(i) for i in ignore_layers]
            if name not in ignore_layers_str:
                add_lora_layers(module, module_names, r,
                                lora_dropout, lora_alpha, ignore_layers)


def unfreeze_model(model):
    """Unfreezes all parameters in a model by setting requires_grad to True."""
    for name, param in model.named_parameters():
        param.requires_grad = True


def create_linear(module):
    """Converts a LoRA linear module back to a linear module."""
    k, d = module.pretrained.weight.shape  # pytorch nn.Linear weights are transposed, that is why variables are k, d and not d, k
    linear = nn.Linear(d, k, bias=True)

    with torch.no_grad():
        linear.weight.copy_(module.pretrained.weight +
                            (module.lora_B.weight @ module.lora_A.weight) * module.scaling)
        linear.bias.copy_(module.pretrained.bias)

    return linear


def merge_lora_layers(model, module_names: Tuple = ("query", "value"), dropout=0.1):
    """
        Replaces LoRA modules with their original linear equivalents. 

        Args:
            model: torch.nn.Module = The PyTorch model to be used
            module_names: Tuple = A tuple containing the names of the LoRA layers to replace
                Ex. ("query") to replace the LoRA modules with "query" in the name --> bert.encoder.layer.0.attention.self.query
            r: int = 
            dropout: float = A float between 0 and 1 representing the dropout probability    
        """
    # enable dropout in frozen layers
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout
    # replace chosen linear modules with lora modules
    for name, module in model.named_children():
        if name in module_names and hasattr(module, "pretrained"):
            temp_linear = create_linear(module)
            setattr(model, name, temp_linear)
        else:
            merge_lora_layers(module, module_names=module_names, dropout=0.1)


class BERT_LoRA(BertSentimentClassifier):
    def __init__(self, config):
        super(BERT_LoRA, self).__init__(config)
        # inject the LoRA layers into the model
        self.config = config

        add_lora_layers(self)
        for name, param in self.named_parameters():
            if "lora" not in name and "classifier" not in name and "bias" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

# use create_lora(module) on all the query/value layers

########### End of LoRA ###############

#### RMSNorm ####


class BERT_RMSNorm(BertSentimentClassifier):
    def __init__(self, config):
        super(BERT_RMSNorm, self).__init__(config)
        self.config = config
        self.bert._replace_layernorm_with_rmsnorm()

#### End of RMSNorm ####

#### SwiGLU ####


class SwiGLU(nn.Module):
    def __init__(self, config):
        super(SwiGLU, self).__init__()
        self.linear1 = nn.Linear(
            config.hidden_size * 4, config.hidden_size * 4)
        self.linear2 = nn.Linear(
            config.hidden_size * 4, config.hidden_size * 4)

    def forward(self, x):
        interm = self.linear1(x)
        return interm * F.sigmoid(interm) + self.linear2(x)


class BERT_SwiGLU(BertSentimentClassifier):
    def __init__(self, config):
        super(BERT_SwiGLU, self).__init__(config)
        self.config = config
        self.bert._replace_gelu_with_swiglu()


#### End of SwiGLU ####


class SentimentDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):

        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding = self.tokenizer(
            sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        labels = torch.LongTensor(labels)

        return token_ids, attention_mask, labels, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents, sent_ids = self.pad_data(
            all_data)

        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'sents': sents,
            'sent_ids': sent_ids
        }

        return batched_data


class SentimentTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):

        sents = [x[0] for x in data]
        sent_ids = [x[1] for x in data]

        encoding = self.tokenizer(
            sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        return token_ids, attention_mask, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, sents, sent_ids = self.pad_data(all_data)

        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'sents': sents,
            'sent_ids': sent_ids
        }

        return batched_data

# Load the data: a list of (sentence, label)


def load_data(filename, flag='train'):
    num_labels = {}
    data = []
    if flag == 'test':
        with open(filename, 'r', encoding='utf-8') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                data.append((sent, sent_id))
    else:
        with open(filename, 'r', encoding='utf-8') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                label = int(record['sentiment'].strip())
                if label not in num_labels:
                    num_labels[label] = len(num_labels)
                data.append((sent, label, sent_id))
        print(f"load {len(data)} data from {filename}")

    if flag == 'train':
        return data, len(num_labels)
    else:
        return data

# Evaluate the model for accuracy.


def model_eval(dataloader, model, device):
    model.eval()  # switch to eval model, will turn off randomness like dropout
    y_true = []
    y_pred = []
    sents = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids, b_mask, b_labels, b_sents, b_sent_ids = batch['token_ids'], batch['attention_mask'],  \
            batch['labels'], batch['sents'], batch['sent_ids']

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits = model(b_ids, b_mask)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        b_labels = b_labels.flatten()
        y_true.extend(b_labels)
        y_pred.extend(preds)
        sents.extend(b_sents)
        sent_ids.extend(b_sent_ids)

    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return acc, f1, y_pred, y_true, sents, sent_ids


def model_test_eval(dataloader, model, device):
    model.eval()  # switch to eval model, will turn off randomness like dropout
    y_pred = []
    sents = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids, b_mask, b_sents, b_sent_ids = batch['token_ids'], batch['attention_mask'],  \
            batch['sents'], batch['sent_ids']

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits = model(b_ids, b_mask)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        y_pred.extend(preds)
        sents.extend(b_sents)
        sent_ids.extend(b_sent_ids)

    return y_pred, sents, sent_ids


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train(args, save_metrics, model_name='baseline'):

    model_dict = {
        'baseline': BertSentimentClassifier,
        'LoRA': BERT_LoRA,
        'RMS': BERT_RMSNorm,
        'SwiGLU': BERT_SwiGLU
    }

    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    train_data, num_labels = load_data(args.train, 'train')
    dev_data = load_data(args.dev, 'valid')

    train_dataset = SentimentDataset(train_data, args)
    dev_dataset = SentimentDataset(dev_data, args)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    # model = BertSentimentClassifier(config)
    model = model_dict[model_name](config)

    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = F.cross_entropy(
                logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_acc, train_f1, *_ = model_eval(train_dataloader, model, device)
        dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(
            f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
        save_metrics["train_loss"].append(train_loss)
        save_metrics["train_acc"].append(train_acc)
        save_metrics["dev_acc"].append(dev_acc)
        save_metrics["train_f1"].append(train_f1)
        save_metrics["dev_f1"].append(dev_f1)


def test(args, save_metrics, model_name='baseline'):

    model_dict = {
        'baseline': BertSentimentClassifier,
        'LoRA': BERT_LoRA,
        'RMS': BERT_RMSNorm,
        'SwiGLU': BERT_SwiGLU
    }

    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']
        # model = BertSentimentClassifier(config)
        model = model_dict[model_name](config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"load model from {args.filepath}")

        dev_data = load_data(args.dev, 'valid')
        dev_dataset = SentimentDataset(dev_data, args)
        dev_dataloader = DataLoader(
            dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

        test_data = load_data(args.test, 'test')
        test_dataset = SentimentTestDataset(test_data, args)
        test_dataloader = DataLoader(
            test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)

        dev_acc, dev_f1, dev_pred, dev_true, dev_sents, dev_sent_ids = model_eval(
            dev_dataloader, model, device)
        print('DONE DEV')
        test_pred, test_sents, test_sent_ids = model_test_eval(
            test_dataloader, model, device)
        print('DONE Test')
        save_metrics["test_acc"].append(dev_acc)
        save_metrics["test_f1"].append(dev_f1)
        with open(args.dev_out, "w+") as f:
            print(f"dev acc :: {dev_acc :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sent_ids, dev_pred):
                f.write(f"{p} , {s} \n")

        with open(args.test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sent_ids, test_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--dev_out", type=str, default="cfimdb-dev-output.txt")
    parser.add_argument("--test_out", type=str,
                        default="cfimdb-test-output.txt")

    parser.add_argument(
        "--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=1)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    SST_JSON = './stats/sst_classifier_saved_metrics.json'
    CFIMDB_JSON = './stats/cfimdb_classifier_saved_metrics.json'

    args = get_args()
    seed_everything(args.seed)
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}.pt'

    print('Training Sentiment Classifier on SST...')
    config = SimpleNamespace(
        filepath='sst-classifier.pt',
        lr=args.lr,
        use_gpu=args.use_gpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dropout_prob=args.hidden_dropout_prob,
        train='data/ids-sst-train.csv',
        dev='data/ids-sst-dev.csv',
        test='data/ids-sst-test-student.csv',
        option=args.option,
        dev_out='predictions/'+args.option+'-sst-dev-out.csv',
        test_out='predictions/'+args.option+'-sst-test-out.csv'
    )

    sst_save_metrics = {
        "batch_size": config.batch_size,
        "lr": config.lr,
        "hidden_dropout_prob": config.hidden_dropout_prob,
        "option": config.option,
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "dev_acc": [],
        "dev_f1": [],
        "test_acc": [],
        "test_f1": []
    }

    train(config, sst_save_metrics)

    print('Evaluating on SST...')
    test(config, sst_save_metrics)

    # Save save_metrics to a JSON file
    with open(SST_JSON, 'w') as f:
        json.dump(sst_save_metrics, f, indent=4)

    print('Metrics saved to sst_classifier_saved_metrics.json')

    print('Training Sentiment Classifier on cfimdb...')
    config = SimpleNamespace(
        filepath='cfimdb-classifier.pt',
        lr=args.lr,
        use_gpu=args.use_gpu,
        epochs=args.epochs,
        batch_size=8,
        hidden_dropout_prob=args.hidden_dropout_prob,
        train='data/ids-cfimdb-train.csv',
        dev='data/ids-cfimdb-dev.csv',
        test='data/ids-cfimdb-test-student.csv',
        option=args.option,
        dev_out='predictions/'+args.option+'-cfimdb-dev-out.csv',
        test_out='predictions/'+args.option+'-cfimdb-test-out.csv'
    )

    cfimdb_save_metrics = {
        "batch_size": config.batch_size,
        "lr": config.lr,
        "hidden_dropout_prob": config.hidden_dropout_prob,
        "option": config.option,
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "dev_acc": [],
        "dev_f1": [],
        "test_acc": [],
        "test_f1": []
    }

    train(config, cfimdb_save_metrics)

    print('Evaluating on cfimdb...')
    test(config, cfimdb_save_metrics)

    # Save save_metrics to a JSON file
    with open(CFIMDB_JSON, 'w') as f:
        json.dump(cfimdb_save_metrics, f, indent=4)

    print('Metrics saved to cfimdb_classifier_saved_metrics.json')
