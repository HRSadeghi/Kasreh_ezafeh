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


import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torchmetrics import MeanMetric
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertConfig






def accuracy_function(input, real_tar, pred):
    accuracies = torch.eq(real_tar, torch.argmax(pred, dim=2))

    mask = torch.logical_not(torch.eq(input['attention_mask'], 0))
    accuracies = torch.torch.logical_and(mask, accuracies)

    accuracies = accuracies.type(torch.FloatTensor)
    mask = mask.type(torch.FloatTensor)
    return torch.sum(accuracies)/torch.sum(mask)


def loss_function(input, 
                  real, 
                  pred, 
                  loss_object = loss_object):
  mask = torch.logical_not(torch.eq(input['attention_mask'], 0))
  loss_ = loss_object(pred, real)

  mask = mask.type(loss_.type())
  loss_ *= mask

  return torch.sum(loss_)/torch.sum(mask)




def train_step(model,
               input,
               tags, 
               optimizer, 
               loss_object, 
               train_loss = None, 
               train_accuracy = None):
    # Step 1. Remember that Pytorch accumulates gradients.
    # We need to clear them out before each instance
    model.zero_grad()

    # Step 3. Run our forward pass.
    tag_scores = model(input)
    tag_scores = torch.permute(tag_scores, [0,2,1])


    # Step 4. Compute the loss, gradients, and update the parameters by
    #  calling optimizer.step()
    loss = loss_function(input, 
                         tags, 
                         tag_scores, 
                         loss_object = loss_object)

    acc = accuracy_function(input, tags, torch.permute(tag_scores, [0,2,1]))

    if train_loss is not None:
        train_loss.update(loss.cpu().item())
    if train_accuracy is not None:
        train_accuracy.update(acc.cpu().item())

    loss.backward()
    optimizer.step()



def evaluate(dataLoader, model): 
    _loss = MeanMetric()
    _accuracy = MeanMetric()
    for batch, (input, tags) in enumerate(dataLoader):
        with torch.no_grad():
            tag_scores = model(input)
            tag_scores = torch.permute(tag_scores, [0,2,1])
            loss = loss_function(input, tags, tag_scores)
            acc = accuracy_function(input, tags, torch.permute(tag_scores, [0,2,1]))

            _loss.update(loss.cpu().item())
            _accuracy.update(acc.cpu().item())

    mean_loss = _loss.compute().cpu().item()
    mean_acc = _accuracy.compute().cpu().item()

    return mean_loss, mean_acc



def load_pretrained_bert_model(model_name = 'HooshvareLab/bert-fa-zwnj-base', 
                               output_hidden_states = True):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = BertConfig.from_pretrained(model_name,
                                        output_hidden_states=output_hidden_states)
    model = AutoModelForMaskedLM.from_pretrained(model_name, config=config)

    return tokenizer, model



def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device