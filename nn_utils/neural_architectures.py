from torch import nn
from transformers import BertForTokenClassification


class BertForTokenClassificationCustom(nn.Module):
    def __init__(self,
                 model_name: str,
                 num_labels: int,
                 hidden_dropout_prob: float,
                 attention_probs_dropout_prob: float):
        """
        This model is a replica of BertForTokenClassification class but instead of being
        a subclass of `PreTrainedModel` (transformers library) it is a subclass of `nn.Module`
        from Pytorch. In fact, `BertForTokenClassification` is instantiated through `from_pretrained`
        within the `init` method.
        The only difference in functionallity is within the `forward` method.
        Here the output of BERT model is reshaped to be directly compatible with
        `nn.module.CrossEntropyLoss` (batch_size, num_classes, sequence_len). This is done in
        order to make compatible this BERT model with the `torch_lr_finder` library.
        Args:
            model_name (`str`): model name as expected in Transformers library
            num_labels (`int`): number of labels
            hidden_dropout_prob(`float`): droput
            attention_probs_dropout_prob (`float`): dropout in attention
        """
        super().__init__()
        self.bert = BertForTokenClassification.from_pretrained(model_name,
                                                               num_labels=num_labels,
                                                               hidden_dropout_prob=hidden_dropout_prob,
                                                               attention_probs_dropout_prob=attention_probs_dropout_prob)

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path: str,
                        num_labels: int = 2,
                        hidden_dropout_prob: float = 0.,
                        attention_probs_dropout_prob: float = 0.):

        return BertForTokenClassificationCustom(model_name=pretrained_model_name_or_path,
                                                num_labels=num_labels,
                                                hidden_dropout_prob=hidden_dropout_prob,
                                                attention_probs_dropout_prob=attention_probs_dropout_prob)

    def forward(self, input_bert):
        outputs = self.bert(*input_bert)

        # Prepare output to be compatible with nn.module.CrossEntropyLoss()
        bert_shape = outputs[0].shape
        return outputs[0].view(bert_shape[0], bert_shape[-1], -1)

