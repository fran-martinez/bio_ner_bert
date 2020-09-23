# Finetuning SciBERT on NER downstream task
![nerbert](imgs/demo.gif)

This repository contains code to finetune BERT-based models on Named Entity Recognition downstream tasks.
A part from providing the code, the repository also provides results for several 
biomedical datasets as well as the models (which I have uploaded into HuggingFace models [website](https://huggingface.co/models))

## Model Usage
If you are only interested on using the models, you can do it directly from 🤗/[transformers](https://huggingface.co/transformers/) 
library. The models websites are available from the following links:
- [`scibert_scivocab_cased_ner_jnlpba`](https://huggingface.co/fran-martinez/scibert_scivocab_cased_ner_jnlpba). 

### Example of usage
Use the pipeline:
````python
from transformers import pipeline

text = "Mouse thymus was used as a source of glucocorticoid receptor from normal CS lymphocytes."

nlp_ner = pipeline("ner",
                   model='fran-martinez/scibert_scivocab_cased_ner_jnlpba',
                   tokenizer='fran-martinez/scibert_scivocab_cased_ner_jnlpba')

nlp_ner(text)

"""
 Output:
---------------------------
[
{'word': 'glucocorticoid', 
'score': 0.9894881248474121, 
'entity': 'B-protein'}, 

{'word': 'receptor', 
'score': 0.989505410194397, 
'entity': 'I-protein'}, 

{'word': 'normal', 
'score': 0.7680378556251526, 
'entity': 'B-cell_type'}, 
 
{'word': 'cs', 
'score': 0.5176806449890137, 
'entity': 'I-cell_type'}, 

{'word': 'lymphocytes', 
'score': 0.9898491501808167, 
'entity': 'I-cell_type'}
]
"""
````
Or load model and tokenizer as follows:
````python
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Example
text = "Mouse thymus was used as a source of glucocorticoid receptor from normal CS lymphocytes."

# Load model
tokenizer = AutoTokenizer.from_pretrained("fran-martinez/scibert_scivocab_cased_ner_jnlpba")
model = AutoModelForTokenClassification.from_pretrained("fran-martinez/scibert_scivocab_cased_ner_jnlpba")

# Get input for BERT
input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)

# Predict
with torch.no_grad():
  outputs = model(input_ids)

# From the output let's take the first element of the tuple.
# Then, let's get rid of [CLS] and [SEP] tokens (first and last)
predictions = outputs[0].argmax(axis=-1)[0][1:-1]

# Map label class indexes to string labels.
for token, pred in zip(tokenizer.tokenize(text), predictions):
  print(token, '->', model.config.id2label[pred.numpy().item()])

"""
Output:
---------------------------
mouse -> O
thymus -> O
was -> O
used -> O
as -> O
a -> O
source -> O
of -> O
glucocorticoid -> B-protein
receptor -> I-protein
from -> O
normal -> B-cell_type
cs -> I-cell_type
lymphocytes -> I-cell_type
. -> O
"""
````

## Training your own model
- The script [`train_ner.py`](https://github.com/fran-martinez/bio_ner_bert/blob/master/train_ner.py) is ready to use 
in order to train and end-to-end BERT-based NER. You just need to download the 
[data](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data) and locate it into the corresponding
 directory (./data/JNLPBA/) or to change the path within `train_ner.py`.  
 
- The script [`find_learning_rate.py`](https://github.com/fran-martinez/bio_ner_bert/blob/master/find_learning_rate.py) can
be used to find an initial learning rate based on the range test detailed in 
[Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186). It makes use of the Pytorch
Implementation [`pytorch-lr-finder`](https://github.com/davidtvs/pytorch-lr-finder). The output given by `BertForTokenClassification`
from `transformers` library implementation is not compatible with `pytorch-lr-finder`. For this reason, I have created 
[`BertForTokenClassificationCustom`](https://github.com/fran-martinez/bio_ner_bert/blob/master/nn_utils/neural_architectures.py) 
class which is a subclass of `torch.nn.Module`. It instantiates `BertForTokenClassification`
and changes the output given by the `forward` method in order to make BERT compatible with `pytorch-lr-finder`. In other words,
the output is reshaped to be directly compatible with `nn.module.CrossEntropyLoss` (`batch_size, num_classes, sequence_len`).
I have mimicked the way of loading the model to be the same as the BERT from `transformers`:

````python
from nn_utils.neural_architectures import BertForTokenClassificationCustom

nerbert = BertForTokenClassificationCustom.from_pretrained('allenai/scibert_scivocab_uncased')
````

The main three elements to train a NER are: 
### The data
It is represented through [`NerDataset`](https://github.com/fran-martinez/bio_ner_bert/blob/master/data_utils/data_utils.py#L35) class.
It is a subclass of `torch.utils.data.dataset.Dataset` and it has a `__getitem__` method that returns the BERT input and
label for a given index. `NerDataset` has a boolean argument, `bert_hugging`, which provides different behaviours. If `True`, 
the returned data by `__getitem__` is a `python` dictionary with `input_ids`, `attention_mask`, `token_type_ids`, and `labels` 
tensors. This format is used during NER training (`train_ner.py`), since it is compatible with `BertForTokenClassification` 
from `transformers` library. During training, `BertForTokenClassification` estimates the loss inside the `forward` method, 
so labels are passed as input. During inference there is no need to provide the labels. 

If `bert_hugging=False`, the returned data is is a tuple with two elements. The first one is a list of tensors with the 
BERT's input (`input_ids`, `attention_mask`, `token_type_ids`). The second is the tensor for the labels. This format is 
compatible with `pytorch-lr-finder` and used in `find_learning_rate.py`.

Internally, `NerDataset` calls a function `data2tensors` that transforms input examples into tensors. The input examples are
represented as a list of a dataclass called `DataSample` that contains words and labels. The output tensors are stored in
a list of a dataclass called `InputBert`. 

### The model
It is represented through `BertForTokenClassification` and `AutoTokenizer` classes from 
[`transformers`](https://github.com/huggingface/transformers) library.

### The trainer
It is represented through [`BertTrainer`](https://github.com/fran-martinez/bio_ner_bert/blob/master/trainer.py) class.
It is responsible of performing a complete training and evaluation loop in Pytorch and it is specially designed for BERT-based 
models from transformers library. It allows to save the model from the epoch with the best F1-score and the tokenizer. The class
optionally generates reports and figures with the obtained results that are automatically stored in disk. 

The report contains the following info that is saved in a file called `classification_report.txt` within the directory `output_dir` 
for the model from the best epoch:
- Classification report at span/entity level (for validation dataset).
- Classification report at word level (for validation dataset).
- Epoch where the best model was found (best F1-score in validation dataset)
- Training loss from the best epoch.
- Validation loss from the best epoch.

Optionally, the class can print validation examples (sentences) where the model commits at least one mistake. It is 
printed at the end of each epoch. This is very useful to inspect the behaviour of your model. This is how the print looks:

```
TOKEN          LABEL          PRED
immunostaining O              O
showed         O              O
the            O              O
estrogen       B-cell_type    B-cell_type
receptor       I-cell_type    I-cell_type
cells          I-cell_type    O
                  ·
                  ·
                  ·
synovial       O              O
tissues        O              O
.              O              O
```
Another thing that it is worth mentioning about this class it concerns the input argument `accumulate_grad_every`.
This parameter sets how often you want to accumulate the gradient. This is useful when there are limitations in the 
batch size due to memory issues. Let's say that in your GPU only fits a model with batch size of 8 and you want to try 
a batch size of 32. Then, you should set this parameter to 4 (8*4=32). Internally, a loop will be run 4 times 
accumulating the gradient for each step. Later, the network parameters will be updated. So at the end, this is equivalent 
to train your network with a batch size of 32. The batch size is inferred from `dataloader_train` argument.

## Experiments
### SciBERT finetuned on JNLPA
#### Language Model
The used pre-trained model is [`allenai/scibert_scivocab_cased`](https://huggingface.co/allenai/scibert_scivocab_cased#).

[SciBERT](https://arxiv.org/pdf/1903.10676.pdf) is a pretrained language model based on BERT and trained by the 
[Allen Institute for AI](https://allenai.org/) on papers from the corpus of 
[Semantic Scholar](https://www.semanticscholar.org/). 
Corpus size is 1.14M papers, 3.1B tokens. SciBERT has its own vocabulary (scivocab) that's built to best match 
the training corpus.

#### Data
The corpus used to fine-tune the NER is [BioNLP / JNLPBA shared task](http://www.geniaproject.org/shared-tasks/bionlp-jnlpba-shared-task-2004).

- Training data consist of 2,000 PubMed abstracts with term/word annotation. This corresponds to 18,546 samples (senteces).
- Evaluation data consist of 404 PubMed abstracts with term/word annotation. This corresponds to 3,856 samples (sentences).

The classes (at word level) and its distribution (number of examples for each class) for training and evaluation datasets are shown below:
 
| Class Label         | # training examples| # evaluation examples|
|:--------------|--------------:|----------------:|
|O              |   382,963     |     81,647      |
|B-protein      |    30,269     |      5,067      |
|I-protein      |    24,848     |      4,774      |
|B-cell_type    |     6,718     |      1,921      |
|I-cell_type    |     8,748     |      2,991      |
|B-DNA          |     9,533     |      1,056      |
|I-DNA          |    15,774     |      1,789      |
|B-cell_line    |     3,830     |        500      |
|I-cell_line    |     7,387     |       9,89      |
|B-RNA          |       951     |        118      |
|I-RNA          |     1,530     |        187      |

#### Model
An exhaustive hyperparameter search was done.
The hyperparameters that provided the best results are:

- Max length sequence: 128
- Number of epochs: 6
- Batch size: 32
- Dropout: 0.3
- Optimizer: Adam

The used learning rate was 5e-5 with a decreasing linear schedule. A warmup was used at the beggining of the training
with a ratio of steps equal to 0.1 from the total training steps.

The model from the epoch with the best F1-score was selected, in this case, the model from epoch 5.


#### Evaluation
The following table shows the evaluation metrics calculated at span/entity level:

|          |   precision|    recall|  f1-score|   
|:---------|-----------:|---------:|---------:|
cell_line   |  0.5205   | 0.7100   | 0.6007   | 
cell_type   |  0.7736   | 0.7422   | 0.7576   |
protein     |  0.6953   | 0.8459   | 0.7633   |
DNA         |  0.6997   | 0.7894   | 0.7419   | 
RNA         |  0.6985   | 0.8051   | 0.7480   | 
|           |          |          |
**micro avg**   |  0.6984   | 0.8076  |  0.7490|
**macro avg**   | 0.7032   | 0.8076   | 0.7498 |

The macro F1-score is equal to 0.7498, compared to the value provided by the Allen Institute for AI in their
[paper](https://arxiv.org/pdf/1903.10676.pdf), which is equal to 0.7728. This drop in performance could be due to 
several reasons, but one hypothesis could be the fact that the authors used an additional conditional random field, 
while this model uses a regular classification layer with softmax activation on top of SciBERT model.

At word level, this model achieves a precision of 0.7742, a recall of 0.8536 and a F1-score of 0.8093.
