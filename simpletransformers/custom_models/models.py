import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import (
    BertModel,
    BertPreTrainedModel,
)
from transformers.modeling_utils import PreTrainedModel, SequenceSummary




class BertForMultiTaskSequenceClassification(BertPreTrainedModel):
    """
    Bert model adapted for multi-task classification

    Currently 2 classification tasks with different numbers of classes
    """
    def __init__(self, config, weight=None):
        super(BertForMultiTaskSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.num_additional_labels = config.num_additional_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_1 = nn.Linear(config.hidden_size, self.config.num_labels)
        self.classifier_2 = nn.Linear(config.hidden_size, self.config.num_additional_labels)
        self.weight = weight

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
        additional_labels=None
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        # Complains if input_embeds is kept

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits_1 = self.classifier_1(pooled_output)
        logits_2 = self.classifier_2(pooled_output)

        outputs = (logits_1, logits_2) + outputs[2:]  # add hidden states and attention if they are here
        # by default they will not be returned
        # (need to set output_hidden_states and output_attentions to True in the BertConfig)

        if labels is not None:
            loss_fct_1 = CrossEntropyLoss(weight=self.weight)
            loss_fct_2 = CrossEntropyLoss() # for now will ignore the weights,
                                            # but in theory could be implemented
            loss_1 = loss_fct_1(logits_1.view(-1, self.num_labels), labels.view(-1))
            loss_2 = loss_fct_2(logits_2.view(-1, self.num_additional_labels), additional_labels.view(-1))
            loss = loss_1 + loss_2
            outputs = (loss,) + outputs

        return outputs  # (loss), logits_1, logits_2, (hidden_states), (attentions)
      












