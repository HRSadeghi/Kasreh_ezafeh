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


import logging
import re
from .tag_mapping import mapping_dic, get_tag2idx_idx2tag_dics
from .delimiters import delimiters

def read_Bijankhan_dataset(path):
    r"""
    A function to take a batch of input and output data corresponding to the input and output of the neural network used
    Args:
        path (`string`):
            Path to a .txt file related to the Bijankhan dataset.

    Returns:
        lines (`list`): 
            A list of tuples where each tuple has two elements. The first element of the tuple is a token and the second element is its corresponding tag.
    """
    with open(path) as f:
        lines = f.readlines()

    lines = [x.strip().split('\t') for x in lines]
    lines = [x for x in lines if len(x) == 2]

    return lines


def get_all_tags(data):
    r"""
    A function to extract all the tags in the dataset in order.
    Args:
        data (`int`):
            A list of tuples where each tuple has two elements. The first element of the tuple is a token and the second element is its corresponding tag.

    Returns:
        tags (`list`): 
            An ordered list of tags in the dataset.
    """
    
    tags = set()
    for x in data:
        try:
            tags.add(x[1])
        except:
            logging.warning('Each row in dataset must contain a word and a tag!')
    tags = sorted(list(tags))
        
    return tags



def map_tags(data, mapping_dic):
    r"""
    A function to convert any token into another token corresponding to a dictionary
    Args:
        data (`list`):
            A list of tuples where each tuple has two elements. The first element of the tuple is a token and the second element is its corresponding tag.
        mapping_dic (`dict`):
            A dictionary to convert a tag to another tag.
    """
    logging.info('Mapping tags ...')
    for x in data:
        x[1] = mapping_dic[x[1]]


def split_data(data, delimiters):
    r"""
    A function to divide the dataset into several sentences. At first, the entire dataset is a single sentence.
    Args:
        data (`list`):
            A list of tuples where each tuple has two elements. The first element of the tuple is a token and the second element is its corresponding tag.
        delimiters (`list`):
            A list of delimiters (string) to convert the entire dataset into multiple sentences.

    Returns:
        encoded_sens (`dict`): 
            A dictionary of BERT model inputs. Each item is a pytorch tensor.
        encoded_labels (`torch.Tensor`): 
            A pytorch tensor obtained based on output tags.
    """

    logging.info('Splitting sentences in dataset...')
    _all_sens, _all_tags = [x[0] for x in data], [x[1] for x in data]
    _sens, _tags = [], []

    _curr_sen, _curr_tag = [], []

    for i, x in enumerate(_all_sens):
        if x not in delimiters:
            _curr_sen.append(x)
            _curr_tag.append(_all_tags[i])
        else:
            _sens.append(_curr_sen)
            _tags.append(_curr_tag)
            _curr_sen, _curr_tag = [], []

    return _sens, _tags


def remove_sens_containing_english(sens, tags):
    r"""
    This function removes any sentence that contains a token containing an English character.

    Args:
        sens (`list`):
            A list of all sentences. Each sentence is in the form of a list of words.
        tags (`list`):
            A list of all tags. Each tag itself is a list that shows the tag of each corresponding word.

    Returns:
        _sens (`dict`): 
            A list of all new sentences. Each sentence is in the form of a list of words.
        _tags (`torch.Tensor`): 
            A list of all new tags. Each tag itself is a list that shows the tag of each corresponding word.
    """

    logging.info('Removing sentences containing english chars...')
    _sens, _tags = [], []
    for i, x in enumerate(sens):
        flag = True
        for y in x:
            if re.search(r'[a-zA-Z]+', y):
                flag = False
                break
        if flag:
            _sens.append(x)
            _tags.append(tags[i])
    return _sens, _tags


def punctuation(string): 
    r"""
    This function removes punctuations from a string.
    Args:
        sens (`string`):
            A character, word or sentence.

    Returns:
        out_string (`string`): 
            A new character, word or sentence with punctuation removed.
    """

    # punctuation marks 
    punctuations = '''!()-[]{};:'"\,<>./?@#$٫+=×%^&*_~…٬«»؛؟،ـ'''
  
    # traverse the given string and if any punctuation 
    # marks occur replace it with null 
    for x in string.lower(): 
        if x in punctuations: 
            string = string.replace(x, " ") 
  
    out_string = ' '.join(string.split())
    return out_string


def remove_punctuation(sens):
    r"""
    This function removes punctuations from each token.
    Args:
        sens (`list`):
            A list of all sentences. Each sentence is in the form of a list of words.

    Returns:
        _out_sens (`list`): 
            A list of all new sentences. Each sentence is in the form of a list of words.
    """

    _sens = []
    for i, x in enumerate(sens):
        _curr = []
        for y in x:
            t = punctuation(y)
            _curr.append(t)
        _sens.append(_curr)
    return _sens



def clean(sens):
    r"""
    This function replaces not-allowed characters from any token.
    Args:
        sens (`list`):
            A list of all sentences. Each sentence is in the form of a list of words.

    Returns:
        _out_sens (`list`): 
            A list of all new sentences. Each sentence is in the form of a list of words.
    """

    map = {'ً':'',
           'ٌ':'',
           'ء':' ',
           '\u200c':' ',
           }
    _out_sens = []
    for x in sens:
        _curr_sen = []
        for y in x:
            _t = y[:]
            for z in map.keys():
                _t = _t.replace(z, map[z])
            _t = ' '.join(_t.split())
            _curr_sen.append(_t)
        _out_sens.append(_curr_sen)
    return _out_sens


def remove_and_split_tokens(sens, tags, mapping_dic = None):
    r"""
    This function initially removes tokens that do not contain any allowed characters. Then, if there is a space in a token, it splits that token into the number of spaces.
    Args:
        sens (`list`):
            A list of all sentences. Each sentence is in the form of a list of words.
        tags (`list`):
            A list of all tags. Each tag itself is a list that shows the tag of each corresponding word.
        mapping_dic (`dict`, *optional*, defaults to `None`):
            A dictionary to convert a tag to another tag.

    Returns:
        _sens (`dict`): 
            A list of all new sentences. Each sentence is in the form of a list of words.
        _tags (`torch.Tensor`): 
            A list of all new tags. Each tag itself is a list that shows the tag of each corresponding word.
    """

    _sens, _tags = [], []
    for i,x in enumerate(sens):
        _curr_sen, curr_tag = [], []
        for j,y in enumerate(x):
            if y == '' or y == ' ' or y is None:
                continue
            _t = y.split(' ')
            _curr_sen += _t
            if mapping_dic is not None:
                curr_tag += [mapping_dic['O']]*(len(_t) - 1)
            else:
                curr_tag += ['O']*(len(_t) - 1)
            curr_tag += [tags[i][j]]
        _sens.append(_curr_sen)
        _tags.append(curr_tag)

    return _sens, _tags

def remove_empty_samples(sens, tags):
    r"""
    This function removes all sentences that do not contain any tokens.
    Args:
        sens (`list`):
            A list of all sentences. Each sentence is in the form of a list of words.
        tags (`list`):
            A list of all tags. Each tag itself is a list that shows the tag of each corresponding word.

    Returns:
        _sens (`dict`): 
            A list of all new sentences. Each sentence is in the form of a list of words.
        _tags (`torch.Tensor`): 
            A list of all new tags. Each tag itself is a list that shows the tag of each corresponding word.
    """

    _sens, _tags = [], []
    for i,x in enumerate(sens):
        if len(x) == 0 or x == [''] or x is None or x == [' ']:
            continue
        _sens.append(x)
        _tags.append(tags[i])
    return _sens, _tags


def remove_long_sens(sens, tags, max_len = 80):
    r"""
    This function removes sentences that exceed the maximum allowed length.

    Args:
        sens (`list`):
            A list of all sentences. Each sentence is in the form of a list of words.
        tags (`list`):
            A list of all tags. Each tag itself is a list that shows the tag of each corresponding word.
        max_len (`int`):
            Maximum length allowed.

    Returns:
        _sens (`dict`): 
            A list of all new sentences. Each sentence is in the form of a list of words.
        _tags (`torch.Tensor`): 
            A list of all new tags. Each tag itself is a list that shows the tag of each corresponding word.
    """
    _sens, _tags = [], []
    for x,y in zip(sens, tags):
        if len(x) < max_len:
            _sens.append(x)
            _tags.append(y)
    return _sens, _tags



def prepare_dataset_for_train(path):
    r"""
    This function removes sentences that exceed the maximum allowed length.

    Args:
        path (`string`):
            Path to a .txt file related to the Bijankhan dataset.

    Returns:
        _sens (`dict`): 
            A list of all necessary sentences for training. Each sentence is in the form of a list of words.
        _tags (`torch.Tensor`): 
            A list of all necessary tags for training. Each tag itself is a list that shows the tag of each corresponding word.
    """

    _section = read_Bijankhan_dataset(path)
    map_tags(_section, mapping_dic)

    _sens, _tags = split_data(_section, delimiters)
    _sens, _tags = remove_sens_containing_english(_sens, _tags)
    _sens = remove_punctuation(_sens)
    _sens = clean(_sens)
    _sens, _tags = remove_and_split_tokens(_sens, _tags, mapping_dic)
    _sens, _tags = remove_empty_samples(_sens, _tags)
    _sens, _tags = remove_long_sens(_sens, _tags)

    return _sens, _tags
