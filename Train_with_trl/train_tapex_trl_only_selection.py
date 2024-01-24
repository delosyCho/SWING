import DataHolder_SQL

import torch
from tqdm import tqdm
from collections import OrderedDict

tqdm.pandas()
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset

from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead, create_reference_model
from trl.core import LengthSampler

import Chuncker

from korquad_eval import exact_match_score, f1_score
import random
import numpy as np

from Reward_Utils import get_reward_count, get_reward_select

import os


def get_reward(input_table_text, answer_text, row_indices, col_idx, get_reward_sel=False):

    try:
        int(answer_text)
        is_number = True
    except:
        is_number = False

    if (answer_text != '1' and len(row_indices) == 1) or get_reward_sel is True:
        reward_value, agg_losses = get_reward_select(
            input_table_text=input_table_text,
            answer_text=answer_text,
            row_indices=row_indices,
            col_idx=col_idx
        )
    elif is_number is True:
        reward_value, agg_losses = get_reward_count(
            answer_text=answer_text,
            row_indices=row_indices,
        )
    else:
        reward_value, agg_losses = get_reward_select(
            input_table_text=input_table_text,
            answer_text=answer_text,
            row_indices=row_indices,
            col_idx=col_idx
        )

    return reward_value, agg_losses


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set the specific GPUs to use

chuncker = Chuncker.Chuncker()


seed = 100
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

data_holder = DataHolder_SQL.DataHolder()

batch_size = 9
batch_size_add = 3

learning_rate = 1e-8

config = PPOConfig(model_name="microsoft/tapex-base-finetuned-wtq",
                   learning_rate=learning_rate,
                   batch_size=batch_size + batch_size_add)
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none"}

loaded_state_dict = torch.load('wtq_output_tapex_sf_0.bin')
new_state_dict = OrderedDict()
for n, v in loaded_state_dict.items():
    name = n.replace("module.", "")
    name = name.replace("bart_model.", "pretrained_model.")  # .module이 중간에 포함된 형태라면 (".module","")로 치환
    new_state_dict[name] = v

model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(config.model_name)
model.load_state_dict(new_state_dict, strict=False)

ref_model = create_reference_model(model=model)

tokenizer = AutoTokenizer.from_pretrained(config.model_name)


ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=None, data_collator=None)
device = ppo_trainer.accelerator.device

print(device)
n_step = 5000

total_reward = 0
total_reward2 = 0

step = 0
step2 = 0

step_check1 = 0
step_check2 = 0

epoch = 0
epoch2 = 0

correction_check = [0, 0, 0, 0, 0, 0]
correction_check2 = [0, 0, 0, 0, 0, 0]

file = open('trl_predictions_nums', 'r', encoding='utf-8')
prediction_texts = file.read().split('\n')

file = open('trl_predictions_sel', 'r', encoding='utf-8')
prediction_texts2 = file.read().split('\n')

pseudo_labels = np.full(shape=[data_holder.input_ids_wtq_rl.shape[0], 2], dtype='<U200', fill_value='None')
pseudo_labels2 = np.full(shape=[data_holder.input_ids_wtq_s_.shape[0], 2], dtype='<U200', fill_value='None')

for s in range(n_step):
    if epoch > 10:
        np.save('pseudo_labels_count', pseudo_labels)
        np.save('pseudo_labels_sel', pseudo_labels2)
        exit(0)

    if step >= int(data_holder.input_ids_wtq_rl.shape[0] / batch_size) * batch_size:
        correction_check = [0, 0, 0, 0, 0, 0]
        step = 0
        total_reward = 0
        epoch += 1

        data_holder.b_ix = 0
        np.random.shuffle(data_holder.r_ix_rl)

        np.save('pseudo_labels_count', pseudo_labels)
        pseudo_label_count = 0
        for p in range(pseudo_labels.shape[0]):
            if pseudo_labels[p] != 'None':
                pseudo_label_count += 1
        print('count:', pseudo_label_count, '/', pseudo_labels.shape[0])

    if step2 >= int(data_holder.input_ids_wtq_s_.shape[0] / batch_size) * batch_size:
        correction_check2 = [0, 0, 0, 0, 0, 0]
        step2 = 0
        epoch2 += 1

        data_holder.b_ix2 = 0
        np.random.shuffle(data_holder.r_ix_rl)

        np.save('pseudo_labels_sel', pseudo_labels2)
        pseudo_label_count = 0
        for p in range(pseudo_labels2.shape[0]):
            if pseudo_labels2[p] != 'None':
                pseudo_label_count += 1
        print('sel:', pseudo_label_count, '/', pseudo_labels2.shape[0])

    batch = {}
    input_dict = data_holder.next_batch_wtq_rl2(tokenizer=tokenizer, batch_size=batch_size)
    input_dict2 = data_holder.next_batch_wtq_rl3(tokenizer=tokenizer, batch_size=batch_size_add)

    input_ids = torch.cat([input_dict['input_ids'], input_dict2['input_ids']], dim=0)
    input_ids = input_ids.to(device)

    label_texts = input_dict['label_text']
    label_texts.extend(input_dict2['label_text'])

    label_texts2 = input_dict['label_text2']
    label_texts2.extend(input_dict2['label_text2'])

    input_texts = input_dict['input_table_text']
    input_texts.extend(input_dict2['input_table_text'])

    query_texts = input_dict['query_text']
    query_texts.extend(input_dict2['query_text'])

    input_indices = input_dict['input_indices']
    input_indices.extend(input_dict2['input_indices'])

    batch['input_ids'] = input_ids
    query_tensors = []
    response_tensors = []
    response_texts = []
    prediction_tokens_list = []
    rewards = []

    for i, input_ids_tensor in enumerate(input_ids):
        test_row_indices = []
        if i < batch_size:
            tks = prediction_texts[input_indices[i]].split('\t')[0].split(' ')
            col_idx = prediction_texts[input_indices[i]].split('\t')[1]
        else:
            tks = prediction_texts2[input_indices[i]].split('\t')[0].split(' ')
            col_idx = prediction_texts2[input_indices[i]].split('\t')[1]
        col_idx = int(col_idx)

        for tk in tks:
            try:
                int(tk)
                test_row_indices.append(int(tk))
            except:
                None

        query_string = str(tokenizer.decode(input_ids_tensor, skip_special_tokens=True)).split('col :')[0]
        query_tensors.append(input_ids_tensor)

        response = ppo_trainer.generate(input_ids_tensor)
        if len(response[0]) > 128:
            response = response[:, 0:128]

        tokens = tokenizer.convert_ids_to_tokens(response[0])
        outputs = tokenizer.batch_decode(response, skip_special_tokens=True)
        response_tensors.append(response.squeeze())
        response_texts.append(outputs[0])

        row_indices = []
        for tk in tokens:
            try:
                int(tk.replace('Ġ', ''))
                row_indices.append(int(tk.replace('Ġ', '')))
            except:
                None

        get_reward_sel = False
        if i < batch_size:
            get_reward_sel = True

        reward, agg_loss = get_reward(
            answer_text=label_texts2[i],
            row_indices=row_indices,
            input_table_text=input_texts[i],
            col_idx=col_idx
        )
        test_reward, test_agg_loss = get_reward(
            answer_text=label_texts2[i],
            row_indices=test_row_indices,
            input_table_text=input_texts[i],
            col_idx=col_idx
        )

        rewards.append(torch.tensor(reward, dtype=torch.float).to(device))
        query_text = tokenizer.batch_decode(input_ids_tensor, skip_special_tokens=True)

        if reward != 3 and False:
            if label_texts[i] != 'none':
                inputs = tokenizer(answer=label_texts[i], return_tensors="np")
                ids = list(inputs['input_ids'][0, :])
                ids.insert(0, 2)
                label_ids = torch.tensor([ids], dtype=torch.long)
                label_ids = label_ids.to(device)

                query_tensors.append(input_ids_tensor)
                response_tensors.append(label_ids.squeeze())
                rewards.append(torch.tensor(3, dtype=torch.float).to(device))
        total_reward += reward

        if i < batch_size:
            step += 1
            if test_row_indices != row_indices:
                step_check1 += 1

                if reward < 0 and 0 > test_reward:
                    correction_check[2] += 1
                if reward > 0 > test_reward:
                    correction_check[3] += 1
                if reward < 0 < test_reward:
                    correction_check[4] += 1
                if reward > 0 and 0 < test_reward:
                    correction_check[5] += 1
            else:
                step_check2 += 1

                if reward > 0 and test_reward > 0:
                    correction_check[0] += 1
                elif reward < 0 and 0 > test_reward:
                    correction_check[1] += 1
        else:
            step2 += 1
            if test_row_indices != row_indices:
                if reward < 0 and 0 > test_reward:
                    correction_check2[2] += 1
                if reward > 0 > test_reward:
                    correction_check2[3] += 1
                if reward < 0 < test_reward:
                    correction_check2[4] += 1
                if reward > 0 and 0 < test_reward:
                    correction_check2[5] += 1
            else:
                if reward > 0 and test_reward > 0:
                    correction_check2[0] += 1
                if reward < 0 and 0 > test_reward:
                    correction_check2[1] += 1

        if step % 50 == 0 or step2 % 50 == 0 and (step > 0 and step2 > 0):
            print('count:', step, correction_check, 'select:', step2, correction_check2, step_check1, step_check2)

        if reward > 0 and (reward > 0 and test_reward > 0 and test_row_indices != row_indices) is False:
            lines = input_texts[i].split('[tr]')
            table_2d = []
            for line in lines:
                # print(line)
                tks = line.split('[td]')
                table_2d.append(tks)

            statement = ''
            for tk in tokens:
                try:
                    statement += str(int(tk.replace('Ġ', ''))) + ' '
                except:
                    continue
            statement = statement.strip()
            if max(row_indices) < len(table_2d) and len(row_indices) == len(set(row_indices)):
                if i < batch_size:
                    pseudo_labels[input_indices[i], 0] = statement
                    pseudo_labels[input_indices[i], 1] = str(epoch)

                else:
                    pseudo_labels2[input_indices[i], 0] = statement
                    pseudo_labels2[input_indices[i], 1] = str(epoch)

    if epoch == 0:
        continue

    if learning_rate == 0:
        continue
    batch["response"] = response_texts

    stats = ppo_trainer.step(
        query_tensors,
        response_tensors,
        rewards
    )

