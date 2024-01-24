from __future__ import absolute_import, division, print_function

import logging
import random
import sys

import numpy as np
import torch
from tqdm import tqdm, trange

from transformers import AdamW, get_linear_schedule_with_warmup, TapexTokenizer

import DataHolder_SQL
import Model_tapex as Model

from collections import OrderedDict


if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() and not False else "cpu")
    n_gpu = torch.cuda.device_count()
    print('n gpu:', n_gpu)

    seed = 1000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    phase = '6'
    phase_to_save = str(int(phase) + 1)

    # Prepare model
    data_holder = DataHolder_SQL.DataHolder(phase=phase, pseudo_path='pseudo_labels/')

    model = Model.TableBartModel(model_size='base')
    model.to(device)
    model.train()
    """
    loaded_state_dict = torch.load('tapex_aug2.bin')
    new_state_dict = OrderedDict()
    for n, v in loaded_state_dict.items():
        name = n.replace("module.", "")  # .module이 중간에 포함된 형태라면 (".module","")로 치환
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    """
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    tokenizer = TapexTokenizer.from_pretrained('microsoft/tapex-base-finetuned-wtq')

    num_params = count_parameters(model)
    logger.info("Total Parameter: %d" % num_params)

    # Prepare optimizer
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    epoch_num = 13
    data_holder.batch_size = 12

    num_step = int(data_holder.input_ids_wtq_s.shape[0] / data_holder.batch_size) * epoch_num

    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, eps=1e-6)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(num_step * 0.1), num_training_steps=num_step
    )

    model.train()
    epoch = 0

    max_grad_norm = 1.0

    for ep in trange(int(epoch_num), desc="Epoch"):
        total_loss = 0
        tr_step = 0

        num_step = int(data_holder.input_ids_wtq_s.shape[0] / data_holder.batch_size)

        for step in range(num_step):
            sub_idx = 5
            num_sel = 3
            num_of_op = 1

            batch = data_holder.next_batch_wtq_tapex_with_pseudo_label(tokenizer=tokenizer,
                                                                       sub_idx=sub_idx,
                                                                       num_sel=num_sel,
                                                                       num_of_op=num_of_op)

            if n_gpu == 1:
                batch = tuple(t.to(device) for t in batch)

            input_ids, attention_mask, token_type_ids, label_ids = batch
            loss = model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=None,
                         label_ids=label_ids)

            #print(loss)
            #input()

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

            tr_step += 1
            total_loss += loss.item()
            mean_loss = total_loss / tr_step
            print(ep, ':', step, loss.item(), mean_loss, '/', num_step)

        epoch += 1
        logger.info("** ** * Saving file * ** **")
        output_model_file = "pseudo_labels/self_training_%s.bin" % phase_to_save
        torch.save(model.state_dict(), output_model_file)


if __name__ == "__main__":
    main()