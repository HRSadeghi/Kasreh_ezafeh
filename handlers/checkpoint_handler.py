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


import os
import torch

def save_checkpoint(base_directory_path, 
                    to_save,
                    score_name,
                    score_strategy = 'max',
                    n_saved = 3,
                    filename_prefix = 'best',
                    ext = 'pt'
                    ):

    if not os.path.exists(base_directory_path):
        os.mkdir(base_directory_path)
    def read_files_in_directory():
        files = [file for file in os.listdir(base_directory_path) if file.endswith(f".{ext}")]
        scores = [float('.'.join(f.split('=')[1].split('.')[:-1])) for f in files]
        files_scores = {files[i]:scores[i] for i in range(len(files))}
        files_scores = sorted(files_scores.items(), key=lambda x: x[1])     
        if score_strategy == 'min':
            files_scores = files_scores[::-1]
        for i in range(max(0, len(files) - n_saved + 1)):
            _file_path = os.path.join(base_directory_path, files_scores[i][0])
            os.remove(_file_path)

        if len(files) == 0:
            global_step = 0
        else:
            global_step = max([int(f.split('_')[3]) for f in files]) + 1
        return global_step
            
    global_step = read_files_in_directory()
    file_name = f"{filename_prefix.replace('_','')}_global_time_{global_step}_{score_name}={to_save[score_name]}.{ext}"
    model_path = os.path.join(base_directory_path, file_name)
    torch.save(to_save, model_path)





def load_checkpoint(base_directory_path, 
                    to_load,
                    score_strategy = 'max',
                    ckpt_path = None,
                    filename_prefix = 'best',
                    ext = 'pt'
                    ):
    if ckpt_path is None:
        files = [file for file in os.listdir(base_directory_path) if file.endswith(f".{ext}")]
        files = [f for f in files if filename_prefix in f]
        scores = [float('.'.join(f.split('=')[1].split('.')[:-1])) for f in files]
        files_scores = {files[i]:scores[i] for i in range(len(files))}
        files_scores = sorted(files_scores.items(), key=lambda x: x[1])
        if score_strategy == 'max':
            ckpt_path = os.path.join(base_directory_path, files_scores[-1][0])
        else:
            ckpt_path = os.path.join(base_directory_path, files_scores[0][0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=torch.device(device))

    for k in checkpoint.keys():
        if 'state_dict' in k:
            if k in to_load.keys():
                to_load[k].load_state_dict(checkpoint[k])
        else:
            to_load[k] = checkpoint[k]

    return ckpt_path

            