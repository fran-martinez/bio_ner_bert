from data_utils import *
from optimizers import get_optimizer_with_weight_decay
from transformers import (get_linear_schedule_with_warmup,
                          get_cosine_schedule_with_warmup,
                          get_cosine_with_hard_restarts_schedule_with_warmup,
                          BertForTokenClassification,
                          AdamW)
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from trainer import BertTrainer
from torch.optim import Adam

# https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data
DATA_TR_PATH = './data/JNLPBA/Genia4ERtask1.iob2'
DATA_TS_PATH = './data/JNLPBA/Genia4EReval1.iob2'

# SEED = 42

# MODEL
# MODEL_NAME = 'allenai/biomed_roberta_base'
MODEL_NAME = 'allenai/scibert_scivocab_uncased'
MAX_LEN_SEQ = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Optimization parameters
N_EPOCHS = 6
BATCH_SIZE = 8
BATCH_SIZE_VAL = 28
WEIGHT_DECAY = 0
LEARNING_RATE = 5e-5  # 2e-4
RATIO_WARMUP_STEPS = .1
DROPOUT = .3
ACUMULATE_GRAD_EVERY = 4
OPTIMIZER = Adam

# Seeds
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)

# get data
training_set = read_data_from_file(DATA_TR_PATH)
test_set = read_data_from_file(DATA_TS_PATH)

# Automatically extract labels and their indexes from data.
labels2ind, labels_count = get_labels(training_set + test_set)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Create loaders for datasets
training_set = NerDataset(dataset=training_set,
                          tokenizer=tokenizer,
                          labels2ind=labels2ind,
                          max_len_seq=MAX_LEN_SEQ)

test_set = NerDataset(dataset=test_set,
                      tokenizer=tokenizer,
                      labels2ind=labels2ind,
                      max_len_seq=MAX_LEN_SEQ)

dataloader_tr = DataLoader(dataset=training_set,
                           batch_size=BATCH_SIZE,
                           shuffle=True)

dataloader_ts = DataLoader(dataset=test_set,
                           batch_size=BATCH_SIZE_VAL,
                           shuffle=False)

# Load model
nerbert = BertForTokenClassification.from_pretrained(MODEL_NAME,
                                                     hidden_dropout_prob=DROPOUT,
                                                     attention_probs_dropout_prob=DROPOUT,
                                                     label2id=labels2ind,
                                                     num_labels=len(labels2ind),
                                                     id2label={str(v): k for k, v in labels2ind.items()})

# Prepare optimizer and schedule (linear warmup and decay)
optimizer = get_optimizer_with_weight_decay(model=nerbert,
                                            optimizer=OPTIMIZER,
                                            learning_rate=LEARNING_RATE,
                                            weight_decay=WEIGHT_DECAY)

training_steps = (len(dataloader_tr)//ACUMULATE_GRAD_EVERY) * N_EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                            num_warmup_steps=training_steps * RATIO_WARMUP_STEPS,
                                            num_training_steps=training_steps)

# Trainer
trainer = BertTrainer(model=nerbert,
                      tokenizer=tokenizer,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      dataloader_train=dataloader_tr,
                      dataloader_test=dataloader_ts,
                      labels2ind=labels2ind,
                      device=DEVICE,
                      n_epochs=N_EPOCHS,
                      accumulate_grad_every=ACUMULATE_GRAD_EVERY,
                      output_dir='./trained_models')

tr_losses, val_losses = trainer.train()
