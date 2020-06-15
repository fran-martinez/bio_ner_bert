# Finetuning SciBERT on NER downstream task
This repository contains code to finetune BERT-based models on Named Entity Recogniton downstream tasks.
A part from just providing the code to train a BERT-based NER, the idea is to provide results for several 
biomedical datasets as well as the models (which are uploaded into HuggingFace models [website](https://huggingface.co/models))

## Usage
If you are only interested on using the models in inference, they are available from their HuggingFace model website:
- [`scibert_scivocab_cased_ner_jnlpba`](https://huggingface.co/fran-martinez/scibert_scivocab_cased_ner_jnlpba). 

### Example of usage (scibert_scivocab_cased_ner_jnlpba)
````python
from transformers import pipeline

text = "Mouse thymus was used as a source of glucocorticoid receptor from normal CS lymphocytes."

nlp_ner = pipeline("ner",
                   model='fran-martinez/scibert_scivocab_cased_ner_jnlpba',
                   tokenizer='fran-martinez/scibert_scivocab_cased_ner_jnlpba')

nlp_ner(text)

# Output:
#---------------------------
# [
# {'word': 'glucocorticoid', 
# 'score': 0.9894881248474121, 
# 'entity': 'B-protein'}, 
# 
# {'word': 'receptor', 
# 'score': 0.989505410194397, 
# 'entity': 'I-protein'}, 
# 
# {'word': 'normal', 
# 'score': 0.7680378556251526, 
# 'entity': 'B-cell_type'}, 
# 
# {'word': 'cs', 
# 'score': 0.5176806449890137, 
# 'entity': 'I-cell_type'}, 
# 
# {'word': 'lymphocytes', 
# 'score': 0.9898491501808167, 
# 'entity': 'I-cell_type'}
# ]
````

## Code explanined
To be done ...

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
several reasons, but one hypothesis could be the fact the the authors used an additional conditional random field, 
while this model uses a regular classification layer with softmax activation on top of SciBERT model.

At word level, this model achieves a precision of 0.7742, a recall of 0.8536 and a F1-score of 0.8093.