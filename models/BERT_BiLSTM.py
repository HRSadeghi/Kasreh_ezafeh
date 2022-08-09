#Copyright 2022 Hamidreza Sadeghi. All rights reserved.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BERTBiLSTMTagger(nn.Module):
    r"""
    This is a class to define the Kasreh Ezafeh model. This model is created using BERT model and BiLSTM model.
    Args:
        bert_model (`transformers.models.bert.modeling_bert.BertForMaskedLM`):
            Pre-trained BERT model trained with the transformers library.
        no_of_bert_layer (`int`, *optional*, defaults to 7):
            Number of bert layers that is used in new model.
        bert_out_dim (`int`, *optional*, defaults to 768):
            The dimension of the pretrained BERT layer that we output from.
        lstm_dim (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
    Examples:
    ```python
    >>>from transformers import AutoTokenizer, AutoModelForMaskedLM, BertConfig
    >>>tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-zwnj-base")
    >>>config = BertConfig.from_pretrained("HooshvareLab/bert-fa-zwnj-base",
    ...                                    output_hidden_states=True)
    >>>model = AutoModelForMaskedLM.from_pretrained("HooshvareLab/bert-fa-zwnj-base", config=config)
    >>>bb_tagger_model = BERTBiLSTMTagger(bert_model = model)
    >>>bb_tagger_model = bb_tagger_model.to(device)
    >>>input = tokenizer(['اجرای طرح ترافیک از ساعت هشت سی تا شانزده'] , return_tensors='pt', padding=True)
    >>>output = bb_tagger_model(input)
    ```"""
    def __init__(self,
                 bert_model, 
                 no_of_bert_layers = 7,
                 bert_out_dim = 768, 
                 lstm_dim  = 128,  
                 num_classes = 3 
                ):
                 
        super(BERTBiLSTMTagger, self).__init__()
        
        self.no_of_bert_layer = no_of_bert_layers
        self.lstm = nn.LSTM(bert_out_dim, lstm_dim, bidirectional=True, num_layers=1)
        self.bert_model = bert_model
        self.linear = nn.Linear(2*lstm_dim, num_classes)
        self.dropout_layer = nn.Dropout(p=0.1)

        
        
    def forward(self, input):

        hidden = self.bert_model(**input)['hidden_states'][self.no_of_bert_layer]
        
        lstm_output, _ = self.lstm(hidden)
        lstm_output = self.dropout_layer(lstm_output)
        
        logits = self.linear(lstm_output)
        return logits
        

