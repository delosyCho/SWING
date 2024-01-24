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

import Model_tapex as Model


chuncker = Chuncker.Chuncker()


def get_reward(chuncker, question, input_table_text, answer_text, predicted_list):  # predicted will be the list of predicted token
    # print(finish, predicted_list)
    try:
        float(answer_text)
        is_number_answer = True
    except:
        is_number_answer = False

    answer_text = str(answer_text).replace(' | ', '|')
    answers = answer_text.replace(' | ', '|').split('|')

    reward = -1
    predicted_words = predicted_list.copy()

    lines = input_table_text.split('[tr]')
    table_2d = []
    for line in lines:
        #print(line)
        tks = line.split('[td]')
        table_2d.append(tks)
    #print()

    predicted_text = ''
    for word in predicted_words:
        predicted_text += str(word).replace('Ġ', '') + ' '
    predicted_text = predicted_text.replace('  ', ' ')

    tks = predicted_text.split(' ')
    row_indices = []
    for word in tks:
        try:
            row_indices.append(int(word))
        except:
            None

    for c in range(len(table_2d[0])):
        predicted_word = ''

        try:
            for r, row_idx in enumerate(row_indices):
                predicted_word += table_2d[row_idx][c]
        except:
            None

        if predicted_word == answer_text.replace('|', ''):
            reward = 1

    return reward


seed = 100
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

batch_size = 16
learning_rate = 0#5e-7

data_holder = DataHolder_SQL.DataHolder()
data_holder.r_ix_s_ = np.array(range(int(data_holder.input_ids_wtq_s_.shape[0] / batch_size) * batch_size))
np.random.shuffle(data_holder.r_ix_s_)

config = PPOConfig(model_name="microsoft/tapex-base-finetuned-wtq", learning_rate=learning_rate, batch_size=batch_size,
                   forward_batch_size=1)
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none"}

loaded_state_dict = torch.load('wtq_output_tapex_ext2.bin')
new_state_dict = OrderedDict()
for n, v in loaded_state_dict.items():
    name = n.replace("module.", "")
    name = name.replace("bart_model.", "pretrained_model.")  # .module이 중간에 포함된 형태라면 (".module","")로 치환
    new_state_dict[name] = v


device = torch.device("cuda:1" if torch.cuda.is_available() and not False else "cpu")

test_model = Model.TableBartModel()
test_model.to(device)
test_model.eval()

model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(config.model_name)
model.load_state_dict(new_state_dict, strict=False)

ref_model = create_reference_model(model=model)

tokenizer = AutoTokenizer.from_pretrained(config.model_name)

ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=None, data_collator=None)
device = ppo_trainer.accelerator.device

print(device)
n_step = 20000

total_reward = 0
step = 0

file = open('trl_predictions', 'r', encoding='utf-8')
prediction_texts = file.read().split('\n')

s_idx = 0

correction_rate = 0
correction_check = [0, 0, 0, 0, 0]

epoch = 0

for s in range(n_step):
    if epoch > 10:
        exit(0)

    if step >= int(data_holder.input_ids_wtq_s_.shape[0] / batch_size) * batch_size:
        print(epoch, ':', step, 'mean:', total_reward / step, correction_check)

        correction_rate = 0
        correction_check = [0, 0, 0, 0, 0]
        step = 0
        total_reward = 0

        epoch += 1

        data_holder.b_ix = 0
        np.random.shuffle(data_holder.r_ix_rl)

    batch = {}

    input_dict = data_holder.next_batch_wtq_rl3(tokenizer=tokenizer, batch_size=batch_size)
    input_ids = input_dict['input_ids']
    input_ids = input_ids.to(device)

    label_texts = input_dict['label_text']
    label_texts2 = input_dict['label_text2']
    input_texts = input_dict['input_table_text']

    input_indices = input_dict['input_indices']
    batch['input_ids'] = input_ids

    query_tensors = []
    response_tensors = []
    response_texts = []
    prediction_tokens_list = []
    rewards = []

    for i, input_ids_tensor in enumerate(input_ids):
        test_row_indices = []
        input_idx = prediction_texts[input_indices[i]].split('\t')[0]
        tks = prediction_texts[input_indices[i]].split('\t')[1].split(' ')
        for tk in tks:
            try:
                int(tk)
                test_row_indices.append(int(tk))
            except:
                None

        input_ids_tensor_ = torch.unsqueeze(input_ids_tensor, dim=0).to(device)

        query_string = str(tokenizer.decode(input_ids_tensor, skip_special_tokens=True)).split('col :')[0]
        query_tensors.append(input_ids_tensor)
        response = ppo_trainer.generate(input_ids_tensor)

        tokens = tokenizer.convert_ids_to_tokens(response[0])
        outputs = tokenizer.batch_decode(response, skip_special_tokens=True)
        response_tensors.append(response.squeeze())

        row_indices = []
        for tk in tokens:
            try:
                int(tk.replace('Ġ', ''))
                row_indices.append(int(tk.replace('Ġ', '')))
            except:
                None

        response_texts.append(outputs[0])
        prediction_tokens_list.append(tokens)

        reward = get_reward(
            question=query_string,
            chuncker=chuncker,
            input_table_text=input_texts[i],
            answer_text=label_texts2[i],
            predicted_list=tokens
        )

        test_reward = get_reward(
            question=query_string,
            chuncker=chuncker,
            input_table_text=input_texts[i],
            answer_text=label_texts2[i],
            predicted_list=tks
        )

        rewards.append(torch.tensor(reward, dtype=torch.float).to(device))

        total_reward += reward
        step += 1

        #if step == 600:
        #    exit(1)
        # print(query_text)

        if step % 10 == 0:
            print(input_idx, input_indices[i])
            print(epoch, ':', step, 'reward:', reward, 'mean:', total_reward / step, correction_check)

        if test_row_indices != row_indices:
            if reward < 0 and 0 > test_reward:
                correction_check[2] += 1
            if reward > 0 > test_reward:
                correction_check[3] += 1
            if reward < 0 < test_reward:
                correction_check[4] += 1

            correction_rate += (reward - test_reward)

            print('prediction reward:', reward, 'test reward:', test_reward, 'correction rate:', correction_rate / 2)
            print(label_texts2[i])
            print(outputs[0])
            print(query_string, input_idx, input_indices[i])
            print(test_row_indices, row_indices)
            print('-----')
        else:
            if reward > 0 and test_reward > 0:
                correction_check[0] += 1
            if reward < 0 and 0 > test_reward:
                correction_check[1] += 1

    if learning_rate == 0:
        continue
    batch["response"] = response_texts

    stats = ppo_trainer.step(
        query_tensors,
        response_tensors,
        rewards
    )

    #ppo_trainer.log_stats(stats, batch, rewards)

