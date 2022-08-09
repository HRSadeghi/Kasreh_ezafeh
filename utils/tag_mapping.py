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

mapping_dic = {
                '@e':'e', 
                'O': 'O', 
                'e':'e', 
                've': 'ye', 
                'y': 'ye', 
                'ye': 'ye'
            }

def get_tag2idx_idx2tag_dics():
    r"""
        Denotes the number of batches per epoch.

        Returns:
            tag2idx (`dict`):
                A dictionary that maps each tag to an index.
            idx2tag (`dict`):
                A dictionary that maps each index to a tag.
        """
        
    tag_set = sorted(list(set(mapping_dic.values())))

    tag2idx = dict()
    idx2tag = dict()

    for i,x in enumerate(tag_set):
        tag2idx[x] = i
        idx2tag[i] = x
    return tag2idx, idx2tag