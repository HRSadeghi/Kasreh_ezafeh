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


from utils.data_preprocessing import prepare_dataset_for_train
from utils.training_utils import load_pretrained_bert_model, get_device, train_step, evaluate
from utils.tag_mapping import get_tag2idx_idx2tag_dics, mapping_dic
from models.BERT_BiLSTM import BERTBiLSTMTagger
from data_loader.loader import Kasreh_DataLoader
from handlers.checkpoint_handler import save_checkpoint, load_checkpoint
from torchmetrics import MeanMetric
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn
import time
import argparse



def _evaluate(model, 
              test_dataLoader, 
              loss_object
              ):

    start = time.time()
    val_loss, val_acc = evaluate(test_dataLoader, model, loss_object)
    duration = time.time() - start

    return val_loss, val_acc, duration






def main():
    parser = argparse.ArgumentParser(description='Create a train command.')
    parser.add_argument('--test_file_path', 
                        type=str,
                        default='dataset/test_data.txt',
                        help='path to the test_data.txt file')
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

    parser.add_argument('--no_of_bert_layer', 
                        type=int,
                        default=7,
                        help='Number of bert layers that is used in new model')
    args = parser.parse_args()


    print('Preparing test dataset ...')    
    test_sens, test_tags = prepare_dataset_for_train(args.test_file_path)


    device = get_device()
    tag2idx, idx2tag = get_tag2idx_idx2tag_dics()

    print('Preparing dataloaders ...')   
    tokenizer, bert_model = load_pretrained_bert_model(model_name = args.Pretrained_BERT_model_name)


    test_dataLoader = Kasreh_DataLoader(test_sens, 
                            test_tags,
                            tokenizer = tokenizer, 
                            tag2idx = tag2idx,
                            mapping_dic = mapping_dic, 
                            device=device,
                            batch_size = args.batch_size)

    print('Creating BERT BiLSTM model ...')   
    model = BERTBiLSTMTagger(bert_model = bert_model, no_of_bert_layer = args.no_of_bert_layer)
    model = model.to(device)

    
    loss_object = nn.CrossEntropyLoss(reduction='none')

    print('Loading model weights ...')   
    to_load={
            'model_state_dict': model,
            }

    load_checkpoint(args.checkpoint_dir, to_load)


    print('Starting to train model ...')  
    val_loss, val_acc, duration = _evaluate(model, 
                                            test_dataLoader, 
                                            loss_object
                                            )
    print(f'Val_loss {val_loss:.4f} Val_accuracy {val_acc:.4f}')
    print(f'Duration_time: {duration:4f}')






if __name__ == '__main__':
    main()   