import time
import random
import numpy as np
import argparse
import sys
import re
import os
from types import SimpleNamespace
import json

import math
from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from itertools import cycle
from bert import BertModel, RoBertModel, RMSRoBertModel, SRMSRoBertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, SentencePairTestDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_sst, test_model_multitask, model_eval_multitask


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


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''

    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        # TODO
        # raise NotImplementedError

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE),
            nn.BatchNorm1d(BERT_HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        )

        self.paraphrase_classifier = nn.Sequential(
            nn.Linear(BERT_HIDDEN_SIZE * 2, BERT_HIDDEN_SIZE),
            nn.BatchNorm1d(BERT_HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(BERT_HIDDEN_SIZE, 1)
        )

        self.similarity_classifier = nn.Sequential(
            nn.Linear(1, BERT_HIDDEN_SIZE),
            nn.BatchNorm1d(BERT_HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(BERT_HIDDEN_SIZE, 1)
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        # TODO
        # raise NotImplementedError
        outputs = self.bert(input_ids, attention_mask)
        cls_output = outputs['pooler_output']

        return cls_output

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        # TODO
        # raise NotImplementedError
        # Get the BERT embeddings
        embeddings = self.forward(input_ids, attention_mask)

        # Pass the embeddings through the sentiment classifier
        logits = self.sentiment_classifier(self.dropout(embeddings))
        return logits

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        # TODO
        # raise NotImplementedError

        # Get BERT embeddings for both sentences in each pair
        cls_embedding_1 = self.forward(input_ids_1, attention_mask_1)
        cls_embedding_2 = self.forward(input_ids_2, attention_mask_2)

        out1 = self.dropout(cls_embedding_1)
        out2 = self.dropout(cls_embedding_2)

        # Concatenate the embeddings
        embeddings = torch.cat((out1, out2), dim=1)

        # Pass the concatenated embeddings through the paraphrase classifier
        logits = self.paraphrase_classifier(embeddings)

        return logits

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        # TODO
        # raise NotImplementedError
        # Get BERT embeddings for both sentences in each pair
        cls_embedding_1 = self.forward(input_ids_1, attention_mask_1)
        cls_embedding_2 = self.forward(input_ids_2, attention_mask_2)

        # Calculate cosine similarity between the embeddings
        logits = F.cosine_similarity(cls_embedding_1, cls_embedding_2, dim=1)

        return logits


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
        self.lora_A = nn.Linear(in_dim, r, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

        # create the low-rank B matrix and initialize to zero
        self.lora_B = nn.Linear(r, out_dim, bias=False)
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
    lora = LinearLoRA(d, k, r, lora_dropout=lora_dropout,
                      lora_alpha=lora_alpha)
    with torch.no_grad():
        lora.pretrained.weight.copy_(module.weight)
        lora.pretrained.bias.copy_(module.bias)
    return lora


def add_lora_layers(
    model,
    module_names: Tuple = ("query", "value"),
    r: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.1,
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


class MultitaskBERT_LoRA(MultitaskBERT):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''

    def __init__(self, config):
        super(MultitaskBERT_LoRA, self).__init__(config)
        # inject the LoRA layers into the model
        self.config = config

        add_lora_layers(self, r=8, lora_alpha=16)
        freeze_model(self)  # freeze the non-LoRA parameters


########### End of LoRA ###############

########### LoRA + RoPE #############

class MutitaskBERT_LoRA_RoPE(MultitaskBERT):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''

    def __init__(self, config):
        super(MutitaskBERT_LoRA_RoPE, self).__init__(config)
        self.config = config
        self.bert = RoBertModel.from_pretrained('bert-base-uncased')
        add_lora_layers(self, r=8, lora_alpha=16)
        freeze_model(self)  # freeze the non-LoRA parameters

########### End of LoRA + RoPE ###############


########### LoRA + RoPE + RMS #############

class MutitaskBERT_LoRA_RoPE_RMS(MultitaskBERT):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''

    def __init__(self, config):
        super(MutitaskBERT_LoRA_RoPE_RMS, self).__init__(config)
        self.config = config
        self.bert = RMSRoBertModel.from_pretrained('bert-base-uncased')
        add_lora_layers(self, r=8, lora_alpha=16)
        freeze_model(self)  # freeze the non-LoRA parameters

########### End of LoRA + RoPE + RMS ###############

########### LoRA + RoPE + RMS #############


class SwiGLU(nn.Module):
    def __init__(self, hidden_size):
        super(SwiGLU, self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size * 2)

    def forward(self, x):
        x, gate = self.linear(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class MutitaskBERT_LoRA_RoPE_RMS_SwiGLU(MultitaskBERT):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''

    def __init__(self, config):
        super(MutitaskBERT_LoRA_RoPE_RMS_SwiGLU, self).__init__(config)
        self.config = config
        self.bert = SRMSRoBertModel.from_pretrained('bert-base-uncased')
        add_lora_layers(self, r=8, lora_alpha=16)
        freeze_model(self)  # freeze the non-LoRA parameters

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE),
            nn.BatchNorm1d(BERT_HIDDEN_SIZE),
            SwiGLU(BERT_HIDDEN_SIZE),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        )

        self.paraphrase_classifier = nn.Sequential(
            nn.Linear(BERT_HIDDEN_SIZE * 2, BERT_HIDDEN_SIZE),
            nn.BatchNorm1d(BERT_HIDDEN_SIZE),
            SwiGLU(BERT_HIDDEN_SIZE),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(BERT_HIDDEN_SIZE, 1)
        )

        self.similarity_classifier = nn.Sequential(
            nn.Linear(1, BERT_HIDDEN_SIZE),
            nn.BatchNorm1d(BERT_HIDDEN_SIZE),
            SwiGLU(BERT_HIDDEN_SIZE),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(BERT_HIDDEN_SIZE, 1)
        )

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        # TODO
        # raise NotImplementedError
        outputs = self.bert(input_ids, attention_mask)
        cls_output = outputs['pooler_output']

        return cls_output

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        # TODO
        # raise NotImplementedError
        # Get the BERT embeddings
        embeddings = self.forward(input_ids, attention_mask)

        # Pass the embeddings through the sentiment classifier
        logits = self.sentiment_classifier(self.dropout(embeddings))
        return logits

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        # TODO
        # raise NotImplementedError

        # Get BERT embeddings for both sentences in each pair
        cls_embedding_1 = self.forward(input_ids_1, attention_mask_1)
        cls_embedding_2 = self.forward(input_ids_2, attention_mask_2)

        out1 = self.dropout(cls_embedding_1)
        out2 = self.dropout(cls_embedding_2)

        # Concatenate the embeddings
        embeddings = torch.cat((out1, out2), dim=1)

        # Pass the concatenated embeddings through the paraphrase classifier
        logits = self.paraphrase_classifier(embeddings)

        return logits

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        # TODO
        # raise NotImplementedError
        # Get BERT embeddings for both sentences in each pair
        cls_embedding_1 = self.forward(input_ids_1, attention_mask_1)
        cls_embedding_2 = self.forward(input_ids_2, attention_mask_2)

        # Calculate cosine similarity between the embeddings
        logits = F.cosine_similarity(cls_embedding_1, cls_embedding_2, dim=1)

        return logits

########### End of LoRA + RoPE + RMS ###############


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


def train_multitask(args, save_metrics, model_name='LoRA'):

    model_dict = {
        'baseline': MultitaskBERT,
        'LoRA': MultitaskBERT_LoRA,
        'RoPE': MutitaskBERT_LoRA_RoPE,
        'RMS': MutitaskBERT_LoRA_RoPE_RMS,
        'SwiGLU': MutitaskBERT_LoRA_RoPE_RMS_SwiGLU
    }

    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    print("========================Loading data========================")
    sst_train_data, num_labels_sst, para_train_data, sts_train_data = load_multitask_data(
        args.sst_train, args.para_train, args.sts_train, split='train')
    sst_dev_data, num_labels_sst, para_dev_data, sts_dev_data = load_multitask_data(
        args.sst_dev, args.para_dev, args.sts_dev, split='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(
        sst_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(
        sst_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=sst_dev_data.collate_fn)

    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(
        para_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(
        para_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=para_dev_data.collate_fn)

    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    sts_train_dataloader = DataLoader(
        sts_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(
        sts_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=sts_dev_data.collate_fn)

    print("========================Data loaded========================")

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels_sst,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = model_dict[model_name](config)

    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)

    with open(model_name + '.txt', 'w') as f:
        f.write(f"Total Trainable Parameters: {total_params}\n\n")
        f.write(str(model.parameters))

    model = model.to(device)
    print("========================Model Created========================")

    print("Model Name: ", model_name)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_avg_normalized_score = 0

    print("========================Training========================")

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Training loop with mixed precision
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        for dataloader in [sst_train_dataloader, para_train_dataloader, sts_train_dataloader]:
            for batch in tqdm(dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                optimizer.zero_grad()

                if dataloader == sst_train_dataloader:
                    input_ids, attention_mask, labels = (
                        batch['token_ids'].to(device),
                        batch['attention_mask'].to(device),
                        batch['labels'].to(device)
                    )
                    logits = model.predict_sentiment(input_ids, attention_mask)
                    loss = (F.cross_entropy(logits, labels.view(-1),
                            reduction='sum') / args.batch_size)

                elif dataloader == para_train_dataloader:
                    input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, labels = (
                        batch['token_ids_1'].to(device),
                        batch['attention_mask_1'].to(device),
                        batch['token_ids_2'].to(device),
                        batch['attention_mask_2'].to(device),
                        batch['labels'].to(device),
                    )
                    logits = model.predict_paraphrase(
                        input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
                    loss = F.binary_cross_entropy_with_logits(
                        logits.squeeze(), labels.float())

                elif dataloader == sts_train_dataloader:
                    input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, labels = (
                        batch['token_ids_1'].to(device),
                        batch['attention_mask_1'].to(device),
                        batch['token_ids_2'].to(device),
                        batch['attention_mask_2'].to(device),
                        batch['labels'].to(device),
                    )
                    if args.option == 'pretrain':
                        logits = model.predict_similarity(
                            input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
                        loss = F.mse_loss(logits.squeeze(), labels.float())
                    else:
                        labels_scaled = labels.float() / 5.0
                        cos_sim = model.predict_similarity(
                            input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
                        loss = F.mse_loss(cos_sim.squeeze(), labels_scaled)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                num_batches += 1

        train_loss = train_loss / num_batches

        sentiment_accuracy, sst_y_pred, sst_sent_ids, \
            paraphrase_accuracy, para_y_pred, para_sent_ids, \
            sts_corr, sts_y_pred, sts_sent_ids = model_eval_multitask(
                sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device
            )

        avg_normalized_score = (sentiment_accuracy +
                                paraphrase_accuracy + ((sts_corr + 1) / 2)) / 3

        if avg_normalized_score > best_avg_normalized_score:
            best_avg_normalized_score = avg_normalized_score
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss:.3f}, "
              f"sentiment acc :: {sentiment_accuracy:.3f}, paraphrase acc :: {paraphrase_accuracy:.3f}, sts corr :: {sts_corr:.3f}, "
              f"avg acc :: {avg_normalized_score:.3f}")

        save_metrics["epoch"].append(epoch)
        save_metrics["train_loss"].append(train_loss)
        save_metrics["train_sentiment_acc"].append(sentiment_accuracy)
        save_metrics["train_paraphrase_acc"].append(paraphrase_accuracy)
        save_metrics["train_sts_corr"].append(sts_corr)
        save_metrics["train_avg_normalized_score"].append(avg_normalized_score)


def test_model(args, save_metrics, model_name):

    model_dict = {
        'baseline': MultitaskBERT,
        'LoRA': MultitaskBERT_LoRA,
        # 'RoPE': MutitaskBERT_LoRA_RoPE
    }

    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = model_dict[model_name](config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device, save_metrics)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str,
                        default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str,
                        default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str,
                        default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str,
                        default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str,
                        default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="finetune")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str,
                        default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str,
                        default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str,
                        default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str,
                        default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str,
                        default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str,
                        default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument(
        "--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=32)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)
    args, unknown = parser.parse_known_args()
    return args


def main():
    args = get_args()
    # save path
    models = ['baseline', 'LoRA', 'RoPE']

    model_name = models[1]

    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask-{model_name}.pt'
    seed_everything(args.seed)  # fix the seed for reproducibility

    save_metrics = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_dropout_prob": args.hidden_dropout_prob,
        "option": args.option,
        "epoch": [],
        "train_loss": [],
        "train_sentiment_acc": [],
        "train_paraphrase_acc": [],
        "train_sts_corr": [],
        "train_avg_normalized_score": [],
        "test_sentiment_accuracy": [],
        "test_paraphrase_accuracy": [],
        "test_sts_corr": []
    }

    train_multitask(args, save_metrics, model_name=model_name)
    test_model(args, save_metrics, model_name)

    # Save save_metrics to a JSON file
    with open(f'stats/multitask_{model_name}_saved_metrics.json', 'w') as f:
        json.dump(save_metrics, f, indent=4)

    print(f'stats/multitask_{model_name}_saved_metrics.json')


if __name__ == "__main__":
    main()
