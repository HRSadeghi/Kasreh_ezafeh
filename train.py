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
from torchmetrics import MeanMetric
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn
import time
import argparse


def train(model, 
          train_dataLoader, 
          val_dataLoader, 
          optimizer, 
          loss_object,
          epochs,
          ):
    for epoch in range(epochs):
        start = time.time()

        train_loss = MeanMetric()
        train_accuracy = MeanMetric()

        for batch, (input, tags) in enumerate(train_dataLoader):
            train_step(model, 
                       input, 
                       tags, 
                       optimizer,
                       loss_object,
                       train_loss = train_loss,
                       train_accuracy = train_accuracy
                       )

            if batch % 100 == 0:
                print(f'Epoch {epoch + 1} Batch {batch} Train_loss {train_loss.compute().cpu().item():.4f} Train_accuracy {train_accuracy.compute().cpu().item():.4f}')


        val_loss, val_acc = evaluate(val_dataLoader, model)

        print(f'Epoch {epoch + 1} Batch {batch} Train_loss {train_loss.compute().cpu().item():.4f} Train_accuracy {train_accuracy.compute().cpu().item():.4f}    Val_loss {val_loss:.4f} Val_accuracy {val_acc:.4f}')
        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
        if (epoch + 1) % 1 == 0:
            #ckpt_save_path = ckpt_manager.save()
            ckpt_save_path = '---'
            print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')





def main():
    parser = argparse.ArgumentParser(description='Create a train command.')
    parser.add_argument('--train_file_path',
                        type=str, 
                        default='dataset/train_data.txt',
                        help='path to the train_data.txt file')
    parser.add_argument('--test_file_path', 
                        type=str,
                        default='dataset/test_data.txt',
                        help='path to the test_data.txt file')
    parser.add_argument('--valid_file_path', 
                        type=str,
                        default='',
                        help='path to the valid_data.txt file')

    parser.add_argument('--batch_size', 
                        type=int,
                        default=64,
                        help='path to the valid_data.txt file')

    parser.add_argument('--epochs', 
                        type=int,
                        default=2,
                        help='path to the valid_data.txt file')


    parser.add_argument('--valid_size', 
                        type=float,
                        default=0.1,
                        help='A float number between 0 and 1 for splitting validation sample from training sample.')

    parser.add_argument('--Pretrained_BERT_model_name', 
                        type=str,
                        default='HooshvareLab/bert-fa-zwnj-base',
                        help='The name of pretrained BERT model or a path to pretrained BERT model')
    args = parser.parse_args()



    print('Preparing training dataset ...')
    train_sens, train_tags = prepare_dataset_for_train(args.train_file_path)
    print('Preparing test dataset ...')    
    test_sens, test_tags = prepare_dataset_for_train(args.test_file_path)

    print('Preparing validation dataset ...')   
    if args.test_file_path != '':
        train_sens, val_sens, train_tags, val_tags = train_test_split(train_sens, train_tags, test_size=args.valid_size, random_state=42)
    else:
        val_sens, val_tags = prepare_dataset_for_train(args.valid_size)

    device = get_device()
    tag2idx, idx2tag = get_tag2idx_idx2tag_dics()

    print('Preparing dataloaders ...')   
    tokenizer, bert_model = load_pretrained_bert_model(model_name = args.Pretrained_BERT_model_name)



    train_dataLoader = Kasreh_DataLoader(train_sens, 
                           train_tags,
                           tokenizer = tokenizer, 
                           tag2idx = tag2idx,
                           mapping_dic = mapping_dic, 
                           device=device,
                           batch_size = args.batch_size)


    val_dataLoader = Kasreh_DataLoader(val_sens, 
                            val_tags,
                            tokenizer = tokenizer, 
                            tag2idx = tag2idx,
                            mapping_dic = mapping_dic, 
                            device=device,
                            batch_size = args.batch_size)


    test_dataLoader = Kasreh_DataLoader(test_sens, 
                            test_tags,
                            tokenizer = tokenizer, 
                            tag2idx = tag2idx,
                            mapping_dic = mapping_dic, 
                            device=device,
                            batch_size = args.batch_size)

    print('Creating BERT BiLSTM model ...')   
    model = BERTBiLSTMTagger(bert_model = bert_model)
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_object = nn.CrossEntropyLoss(reduction='none')


    print('Starting to train model ...')  
    train(model, 
          train_dataLoader, 
          val_dataLoader, 
          optimizer, 
          loss_object,
          args.epochs,
          )






if __name__ == '__main__':
    main()   