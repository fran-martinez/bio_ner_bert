import torch
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss
from transformers import BertForTokenClassification


class BertForTokenClassificationSimple(BertForTokenClassification):
    def __init__(self,
                 config,
                 freeze_bert=False):
        super().__init__(config)

        # Freeze bert layers
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                class_weights=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        out = self.dropout(outputs[0])
        logits = self.classifier(out)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            if class_weights is None:
                loss_fct = CrossEntropyLoss()
            else:
                loss_fct = CrossEntropyLoss(weight=class_weights)

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(active_loss,
                                            labels.view(-1),
                                            torch.tensor(loss_fct.ignore_index).type_as(labels))

                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


class BertLstmForTokenClassification(BertForTokenClassification):
    def __init__(self,
                 config,
                 freeze_bert=False,
                 dropout_lstm=0.05,
                 num_hidden_units_lstm=256,
                 num_lstm_layers=1):
        super().__init__(config)

        if num_lstm_layers == 1:
            dropout_lstm = .0

        # Let's build a BiLSTM on top of BERT
        self.lstm = nn.LSTM(self.config.hidden_size,
                            hidden_size=num_hidden_units_lstm // 2,
                            num_layers=num_lstm_layers,
                            dropout=dropout_lstm,
                            bidirectional=True,
                            batch_first=True)

        self.linear = nn.Linear(num_hidden_units_lstm, self.config.num_labels)

        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_model.parameters():
                p.requires_grad = False

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                class_weights=None):

        output_bert = self.bert(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                head_mask=head_mask,
                                inputs_embeds=inputs_embeds)

        out_lstm, _ = self.lstm(output_bert[0])
        out_lstm = self.dropout(out_lstm)
        logits = self.linear(out_lstm)

        output_bert = (logits,) + output_bert[2:]  # add hidden states and attention if they are here
        if labels is not None:
            if class_weights is None:
                loss_fct = CrossEntropyLoss()
            else:
                loss_fct = CrossEntropyLoss(weight=class_weights)

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(active_loss,
                                            labels.view(-1),
                                            torch.tensor(loss_fct.ignore_index).type_as(labels))

                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            output_bert = (loss,) + output_bert

        return output_bert  # (loss), scores, (hidden_states), (attentions)


class BertForTokenClassificationCustom(nn.Module):
    def __init__(self,
                 model_name: str,
                 num_labels: int,
                 hidden_dropout_prob: float,
                 attention_probs_dropout_prob: float):
        super().__init__()
        self.bert = BertForTokenClassification.from_pretrained(model_name,
                                                               num_labels=num_labels,
                                                               hidden_dropout_prob=hidden_dropout_prob,
                                                               attention_probs_dropout_prob=attention_probs_dropout_prob)

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path: str,
                        num_labels: int,
                        hidden_dropout_prob: float,
                        attention_probs_dropout_prob: float):

        return BertForTokenClassificationCustom(model_name=pretrained_model_name_or_path,
                                                num_labels=num_labels,
                                                hidden_dropout_prob=hidden_dropout_prob,
                                                attention_probs_dropout_prob=attention_probs_dropout_prob)

    def forward(self, input_bert):
        outputs = self.bert(*input_bert)

        # Prepare output to be compatible with nn.module.CrossEntropyLoss()
        bert_shape = outputs[0].shape
        return outputs[0].view(bert_shape[0], bert_shape[-1], -1)

