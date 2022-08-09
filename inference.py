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


from utils.training_utils import load_pretrained_bert_model, get_device, train_step, evaluate
from utils.tag_mapping import get_tag2idx_idx2tag_dics
from models.BERT_BiLSTM import BERTBiLSTMTagger
from handlers.checkpoint_handler import load_checkpoint
import torch
import torch.optim as optim
import torch.nn as nn
import time
import argparse


def inference(input_sen,
              model,
              tokenizer,
              idx2tag,
              device = 'cpu'):

    input = tokenizer([input_sen] , return_tensors='pt', padding=True)
    input = input.to(device)

    input_ids = list(input['input_ids'][0].detach().cpu().numpy())
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    start = time.time()
    __o = model(input)

    duration = time.time() - start

    print(f'Duration_time: {duration:4f}')

    out_ids = torch.argmax(__o[0], -1).detach().cpu().numpy()
    out_labels = [idx2tag[x] for x in out_ids]

    for x,y in zip(input_tokens, out_labels):
        print(x,y)





def main():
    parser = argparse.ArgumentParser(description='Create a train command.')

    parser.add_argument('--input_sen', 
                        type=str,
                        default='',
                        help='A sentence in Persian language')

    parser.add_argument('--checkpoint_dir', 
                        type=str,
                        default='saved_checkpoints',
                        help='path to the checkpoint directory')

    parser.add_argument('--batch_size', 
                        type=int,
                        default=64,
                        help='path to the valid_data.txt file')

    parser.add_argument('--Pretrained_BERT_model_name', 
                        type=str,
                        default='HooshvareLab/bert-fa-zwnj-base',
                        help='The name of pretrained BERT model or a path to pretrained BERT model')

    args = parser.parse_args()



    device = get_device()
    _, idx2tag = get_tag2idx_idx2tag_dics()

    print('Loading tokenizer of pretrained BERT model ...')    
    tokenizer, bert_model = load_pretrained_bert_model(model_name = args.Pretrained_BERT_model_name)



    print('Loading model weights ...')   
    model = BERTBiLSTMTagger(bert_model = bert_model)
    model = model.to(device)

    to_load={
            'model_state_dict': model,
            'optimizer_state_dict': optimizer,
            }

    load_checkpoint(args.checkpoint_dir, to_load)


    inference(args.input_sen,
              model,
              tokenizer,
              idx2tag,
              device = device)






if __name__ == '__main__':
    main()   