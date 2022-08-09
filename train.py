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
from utils.traning_utils import load_pretrained_bert_model, get_device
from utils.tag_mapping import get_tag2idx_idx2tag_dics, mapping_dic
from models.BERT_BiLSTM import BERTBiLSTMTagger
from data_loader.loader import Kasreh_DataLoader
from sklearn.model_selection import train_test_split
import argparse



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
        test_sens, test_tags = prepare_dataset_for_train(args.test_file_path)

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
                           batch_size = 64)


    val_dataLoader = Kasreh_DataLoader(val_sens, 
                            val_tags,
                            tokenizer = tokenizer, 
                            tag2idx = tag2idx,
                            mapping_dic = mapping_dic, 
                            device=device,
                            batch_size = 64)


    test_dataLoader = Kasreh_DataLoader(test_sens, 
                            test_tags,
                            tokenizer = tokenizer, 
                            tag2idx = tag2idx,
                            mapping_dic = mapping_dic, 
                            device=device,
                            batch_size = 64)

    print('Creating BERT BiLSTM model ...')   
    model = BERTBiLSTMTagger(bert_model = bert_model)
    model = model.to(device)




if __name__ == '__main__':
    main()   