from difflib import SequenceMatcher

import DataHolder_SQL

import torch
from tqdm import tqdm
tqdm.pandas()

from collections import OrderedDict
from transformers import TapexTokenizer
from korquad_eval import exact_match_score, f1_score

from Reward_Utils import get_reward_op as get_reward

import Chuncker

import random
import numpy as np

import Model_tapex as Model

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set the specific GPUs to use

chuncker = Chuncker.Chuncker()

seed = 100
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

data_holder = DataHolder_SQL.DataHolder()
data_holder.r_ix_rl_op = np.array(range(data_holder.input_ids_wtq_rl_op.shape[0]))

batch_size = 1
learning_rate = 0

loaded_state_dict = torch.load('wtq_output_tapex_sf_0.bin')
new_state_dict = OrderedDict()
for n, v in loaded_state_dict.items():
    name = n.replace("module.", "")  # .module이 중간에 포함된 형태라면 (".module","")로 치환
    new_state_dict[name] = v

device1 = torch.device("cuda" if torch.cuda.is_available() and not False else "cpu")
device2 = torch.device("cuda" if torch.cuda.is_available() and not False else "cpu")
print(device1, device2)
model1 = Model.TableBartModel()
model1.to(device1)
model1.load_state_dict(new_state_dict, strict=False)
model1.eval()

#loaded_state_dict2 = torch.load('wtq_output_tapex_agg.bin')
loaded_state_dict2 = torch.load('wtq_output_tapex_col_sel.bin')
new_state_dict2 = OrderedDict()
for n, v in loaded_state_dict2.items():
    name = n.replace("module.", "")  # .module이 중간에 포함된 형태라면 (".module","")로 치환
    new_state_dict2[name] = v

model2 = Model.TableBartModel()
model2.to(device1)
model2.load_state_dict(new_state_dict2, strict=False)
model2.eval()

tokenizer = TapexTokenizer.from_pretrained('microsoft/tapex-base-finetuned-wtq')

n_step = data_holder.input_ids_wtq_rl_op.shape[0] + 1

total_reward = 0
step = 0

file = open('trl_predictions_op', 'w', encoding='utf-8')

for s in range(n_step):
    batch = {}

    input_dict = data_holder.next_batch_wtq_rl_op(tokenizer=tokenizer, batch_size=batch_size)
    #input_dict = data_holder.next_batch_wtq_rl2(tokenizer=tokenizer, batch_size=batch_size)
    input_ids = input_dict['input_ids']
    input_ids = input_ids.to(device1)

    label_texts = input_dict['label_text']
    label_texts2 = input_dict['label_text2']
    input_texts = input_dict['input_table_text']
    query_texts = input_dict['query_texts']

    input_indices = input_dict['input_indices']
    print(s, input_indices, 'processing...')

    batch['input_ids'] = input_ids

    outputs = model1.bart_model.generate(input_ids, max_length=128)
    #outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)

    decoder_input_ids = np.array([[2, 0, 11311, 35]], dtype=np.int32)
    decoder_input_ids = torch.tensor(np.array(decoder_input_ids), dtype=torch.long)
    decoder_input_ids = decoder_input_ids.to(device1)

    outputs2 = model2.bart_model.generate(input_ids, decoder_input_ids=decoder_input_ids, max_length=128)
    outputs2 = tokenizer.batch_decode(outputs2, skip_special_tokens=True)
    """
    lines = input_dict['input_table_text'][0].split('[tr]')
    tks = lines[0].split('[td]')
    print(tks)
    tks = lines[1].split('[td]')
    print(tks)
    tks = lines[2].split('[td]')
    print(tks)
    print(query_texts)
    """

    lines = input_dict['input_table_text'][0].split('[tr]')
    tks = lines[0].split('[td]')

    for k in range(len(outputs)):
        try:
            col_word = outputs2[0].split(', row:')[0].replace('col:', '')

            col_scores = []
            for c in range(len(tks)):
                s = SequenceMatcher(None, tks[c], col_word)
                col_scores.append(s.ratio())

            col_index = col_scores.index(max(col_scores))
        except:
            col_index = 0

        tokens = tokenizer.convert_ids_to_tokens(outputs[0])

        predicted_text = ''
        for word in tokens:
            predicted_text += str(word).replace('Ġ', '') + ' '
        predicted_text = predicted_text.replace('  ', ' ')
        file.write(predicted_text + '\t' + str(col_index) + '\n')
