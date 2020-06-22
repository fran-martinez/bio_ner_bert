from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer


@dataclass
class DataSample:
    """
    A single training/test example (sentence) for token classification.

    """
    words: List[str]
    labels: List[str]


@dataclass
class InputBert:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a BERT model.

    """
    input_ids: torch.tensor
    attention_mask: torch.tensor
    token_type_ids: torch.tensor
    labels: Optional[torch.tensor] = None


class NerDataset(Dataset):
    def __init__(self,
                 dataset: List[DataSample],
                 tokenizer: PreTrainedTokenizer,
                 labels2ind: Dict[str, int],
                 max_len_seq: int = 512,
                 bert_hugging: bool = True):
        """
        Class that builds a torch Dataset specially designed for NER data.
        Args:
            dataset (list of `DataSample` instances): Each data sample is a dataclass
                that contains two fields: `words` and `labels`. Both are lists of `str`.
            tokenizer (`PreTrainedTokenizer`): Pre-trained tokenizer from transformers
                library. Usually loaded as `AutoTokenizer.from_pretrained(...)`.
            labels2ind (`dict`): maps `str` class labels into `int` indexes.
            max_len_seq (`int`): Max length sequence for each example (sentence).
            bert_hugging (`bool`):
        """
        super(NerDataset).__init__()
        self.bert_hugging = bert_hugging
        self.max_len_seq = max_len_seq
        self.label2ind = labels2ind
        self.features = data2tensors(data=dataset,
                                     tokenizer=tokenizer,
                                     label2idx=self.label2ind,
                                     max_seq_len=max_len_seq,
                                     pad_token_label_id=nn.CrossEntropyLoss().ignore_index)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> Union[Dict[str, torch.tensor],
                                      Tuple[List[torch.tensor], torch.tensor]]:
        if self.bert_hugging:
            return asdict(self.features[i])
        else:
            inputs = asdict(self.features[i])
            labels = inputs.pop('labels')
            return list(inputs.values()), labels


def get_labels(data: List[DataSample]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Automatically extract labels types from the data and its count.
    Args:
        data (list of `DataSample`): Each data sample is a dataclass that contains
            two fields: `words` and `labels`. Both are lists of `str`.

    Returns:
        labels2idx (`dict`): maps `str` class labels into `int` indexes.
        labels_count(`dict`): The number of words for each class label that appears in
            the dataset. Usufull information if you want to apply class weights on
            imbalanced data.

    """
    labels = set()
    labels_counts = defaultdict(int)
    for sent in data:
        labels.update(sent.labels)

        for label_ in sent.labels:
            labels_counts[label_] += 1

    if "O" not in labels:
        labels.add('O')
        labels_counts['0'] = 0

    # Convert list of labels ind a mapping labels -> index
    labels2idx = {label_: i for i, label_ in enumerate(labels)}
    return labels2idx, dict(labels_counts)


def get_class_weight_tensor(labels2ind: Dict[str, int],
                            labels_count: Dict[str, int]) -> torch.Tensor:
    """
    Get the class weights based on the class labels frequency within the dataset.
    Args:
        labels2ind (`dict`): maps `str` class labels into `int` indexes.
        labels_count (`dict`): The number of words for each class label that appears in
            the dataset.

    Returns:
        torch.Tensor with the class weights. Size (num_classes).

    """
    label2ind_list = [(k, v) for k, v in labels2ind.items()]
    label2ind_list.sort(key=lambda x: x[1])
    total_labels = sum([count for label, count in labels_count.items()])
    class_weights = [total_labels/labels_count[label] for label, _ in label2ind_list]
    return torch.tensor(np.array(class_weights)/max(class_weights), dtype=torch.float32)


def read_data_from_file(file_path: str, sep: str = '\t') -> List[DataSample]:
    """
    Load data from a txt file (BIO tagging format) and transform it into the
    required format (list of `DataSample` instances).
    Args:
        file_path (`str`): complete path where the data is located (path + filename).
        sep (`str`): Symbol used to separete word from label at each line. Default `\t`.

    Returns:
        List of `DataSample` instances containing words and labels.

    """
    examples = []
    words = []
    labels = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            splits = line.split(sep)
            if len(splits) > 1:
                words.append(splits[0])
                labels.append(splits[-1].replace('\n', ''))
            else:
                examples.append(DataSample(words=words, labels=labels))
                words = []
                labels = []
    return examples


def data2tensors(data: List[DataSample],
                 tokenizer: PreTrainedTokenizer,
                 label2idx: Dict[str, int],
                 pad_token_label_id: int = -100,
                 max_seq_len: int = 512) -> List[InputBert]:
    """
    Takes data and converts it into tensors to feed the neural network.
    Args:
        data (`list`): List of `DataSample` instances containing words and labels.
        tokenizer (`PreTrainedTokenizer`): Pre-trained tokenizer from transformers
            library. Usually loaded as `AutoTokenizer.from_pretrained(...)`.
        label2idx (`dict`): maps `str` class labels into `int` indexes.
        pad_token_label_id (`int`): index to define the special token [PAD]
        max_seq_len (`int`): Max sequence length.

    Returns:
        List of `InputBert` instances. `InputBert` is a dataclass that contains
        `input_ids`, `attention_mask`, `token_type_ids` and `labels` (Optional).

    """

    features = []
    for sentence in data:
        tokens = []
        label_ids = []
        for word, label in zip(sentence.words, sentence.labels):
            subword_tokens = tokenizer.tokenize(text=word)

            # BERT could return an empty list of subtokens
            if len(subword_tokens) > 0:
                tokens.extend(subword_tokens)

                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label2idx[label]] + [pad_token_label_id] * (len(subword_tokens) - 1))
                # if label.startswith('B'):
                #     label_ids.extend([label2idx[label]] + [label2idx[f"I{label[1:]}"]] * (len(subword_tokens) - 1))
                # else:
                #     label_ids.extend([label2idx[label]] + [label2idx[label]] * (len(subword_tokens) - 1))

        # Drop part of the sequence longer than max_seq_len (account also for [CLS] and [SEP])
        if len(tokens) > max_seq_len - 2:
            tokens = tokens[:max_seq_len - 2]
            label_ids = label_ids[: max_seq_len - 2]

        # Add special tokens  for the list of tokens and its corresponding labels.
        # For BERT: cls_token = '[CLS]' and sep_token = '[SEP]'
        # For RoBERTa: cls_token = '<s>' and sep_token = '</s>'
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        label_ids = [pad_token_label_id] + label_ids + [pad_token_label_id]

        # Create an attention mask (used to locate the padding)
        padding_len = (max_seq_len - len(tokens))
        attention_mask = [1] * len(tokens) + [0] * padding_len

        # Add padding
        tokens += [tokenizer.pad_token] * padding_len
        label_ids += [pad_token_label_id] * padding_len

        # Convert tokens to ids
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Create segment_id. All zeros since we only have one sentence
        segment_ids = [0] * max_seq_len

        # Assert all the input has the expected length
        assert len(input_ids) == max_seq_len
        assert len(label_ids) == max_seq_len
        assert len(attention_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len

        # Append input features for each sequence/sentence
        features.append((InputBert(input_ids=torch.tensor(input_ids),
                                   attention_mask=torch.tensor(attention_mask),
                                   token_type_ids=torch.tensor(segment_ids),
                                   labels=torch.tensor(label_ids))))
    return features
