from transformers import TapasTokenizer, TapexTokenizer, T5Tokenizer

import numpy as np

import pandas as pd
from datasets import load_dataset


def is_num(str_word):
    try:
        float(str_word)
        return True
    except:
        return False


def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


def strip_word(word: str):
    return word.replace('\\\\n', ' ').replace('\\xa0', ' ')


if "__main__" == __name__:
    dataset_dev = load_dataset("wikitablequestions")['validation']
    tables_ = dataset_dev['table']
    questions_ = dataset_dev['question']
    answer_lists_ = dataset_dev['answers']

    dataset = load_dataset("wikitablequestions")['train']
    tables = dataset['table']
    questions = dataset['question']
    answer_lists = dataset['answers']

    questions.extend(questions_)
    tables.extend(tables_)
    answer_lists.extend(answer_lists_)

    tokenizer = TapexTokenizer.from_pretrained('microsoft/tapex-base-finetuned-wtq')
    tokenizer_tapas = TapasTokenizer.from_pretrained('google/tapas-base')

    count = 0
    count2 = 0
    count2_ = 0
    count3 = 0
    count_ = 0

    max_length = 1024
    answer_length = 64

    input_ids = np.zeros(shape=[len(questions), max_length], dtype=np.int32)
    attention_mask = np.zeros(shape=[len(questions), max_length], dtype=np.int32)
    token_type_ids = np.zeros(shape=[len(questions), max_length, 5], dtype=np.int32)

    decoder_ids = np.full(shape=[len(questions), answer_length], dtype=np.int32, fill_value=0)
    decoder_attention_mask = np.full(shape=[len(questions), answer_length], dtype=np.int32, fill_value=0)

    label_ids = np.full(shape=[len(questions), answer_length], dtype=np.int32, fill_value=-100)
    label_ids2 = np.full(shape=[len(questions), answer_length], dtype=np.int32, fill_value=-100)

    for i in range(len(questions)):
        table_dict = tables[i]
        table_2d = []
        table_2d.append(table_dict['header'])
        table_2d.extend(table_dict['rows'])

        num_col = len(table_2d[0])
        num_row = len(table_2d)
        original_num_row = len(table_2d)

        if num_row > 300:
            continue

        question = str(questions[i])
        answers = answer_lists[i]
        answer = answers[0]
        for a in range(1, len(answers)):
            answer += '|' + answers[a]
        answer_str = str(answer)

        """
        ###
        # make input text and 2d table list
        a = tokenizer.decode(inputs_tapex['input_ids'][0], skip_special_tokens=True)
        pattern = 'row \d*'

        sql = re.sub(pattern=pattern, repl='col', string=a)
        lines = sql.split(' col : ')
        lines.pop(0)

        table_2d = []
        for line in lines:
            line = line.replace(' |', '|').replace('| ', '|')
            tks = line.split('|')
            table_2d.append(tks)
        
        check = True
        for r in range(1, len(table_2d) - 1):
            if len(table_2d[r - 1]) != len(table_2d[r]):
                check = False
        if check is False:
            count2_ += 1
        """
        is_number_case = True
        whole_count_case = False

        try:
            answer_value = float(answer)
            if answer_value + 1 == original_num_row:
                if str(question).find('total') == -1 and str(question).find('table') == -1:
                    whole_count_case = True
                else:
                    count3 += 1
                    continue
            else:
                count3 += 1
                continue
        except:
            is_number_case = False

        supervision = True
        for word in answers:
            is_exist = False
            for tl in table_2d:
                for td in tl:
                    if str(td).lower() == str(word).lower():
                        is_exist = True

            if is_exist is False:
                supervision = False

        #if whole_count_case is True:
        #    continue

        if is_number_case is True:
            continue

        if supervision is False:
            count2 += 1
            continue

        is_okay = True
        selected_answer_positions = []
        for answer_text in answers:
            check_count = 0
            is_check = False
            for r, tl in enumerate(table_2d):
                for c, td in enumerate(tl):
                    if str(td).lower() == str(answer_text).lower():
                        is_check = True
                        selected_answer_positions.append((r, c))
                        check_count += 1
            #print(count)
            if check_count > 1:
                is_okay = False

            if is_check is False:
                is_okay = False

        if is_okay is False and whole_count_case is False:
            count2_ += 1
            continue

        table = {}
        for c in range(num_col):
            cols = []
            for r in range(1, len(table_2d)):
                col_word = str(table_2d[r][c])
                cols.append(strip_word(col_word))
            header_word = strip_word(str(table_2d[0][c]))
            table[header_word] = cols
        table = pd.DataFrame.from_dict(table)

        inputs_tapex = tokenizer(table, question,
                                 answer=answers,
                                 padding="max_length", return_tensors="np",
                                 max_length=1024, truncation=False)
        inputs = tokenizer_tapas(table=table, queries=question, padding="max_length", return_tensors="np")
        tokens = tokenizer.convert_ids_to_tokens(inputs_tapex['input_ids'][0])

        # check answers are in table
        if len(tokens) > max_length:
            continue

        token_type_inputs = np.zeros(shape=[len(table_2d), len(table_2d[0]), 3], dtype=np.int32)
        #print('length:', inputs['input_ids'].shape[1])
        for j in range(inputs['input_ids'].shape[1]):
            if inputs['token_type_ids'][0, j, 0] == 1:
                r_ix = inputs['token_type_ids'][0, j, 2]
                c_ix = inputs['token_type_ids'][0, j, 1] - 1
                token_type_inputs[r_ix, c_ix, 0] = inputs['token_type_ids'][0, j, 4]
                token_type_inputs[r_ix, c_ix, 1] = inputs['token_type_ids'][0, j, 5]
                token_type_inputs[r_ix, c_ix, 2] = inputs['token_type_ids'][0, j, 6]

        rows = []
        cols = []
        ranks = []
        ranks_inv = []
        numeric_relations = []

        c_ix = 0
        r_ix = 0

        input_text = ''

        for j in range(len(tokens)):
            if tokens[j] == 'Ġ|':
                c_ix += 1

            if tokens[j] == 'Ġrow':
                try:
                    row_ix = int(tokens[j + 1].replace('Ġ', ''))
                    if row_ix == r_ix + 1:
                        r_ix += 1
                        c_ix = 0

                except:
                    None

            try:
                ranks.append(token_type_inputs[r_ix, c_ix, 0])
                ranks_inv.append(token_type_inputs[r_ix, c_ix, 1])
                numeric_relations.append(token_type_inputs[r_ix, c_ix, 2])
                rows.append(r_ix)
                cols.append(c_ix)
            except:
                # check input sentence
                for tr in table_2d:
                    print(tr)
                print(len(table_2d), len(table_2d[0]), token_type_inputs.shape)
                print(r_ix, c_ix)
                print(tokens)
                input()
            input_text += '(' + tokens[j] + ',' + str(r_ix) + ',' + str(c_ix) + ')'

        length = len(tokens)
        if length > max_length:
            length = max_length

        input_ids[count] = inputs_tapex['input_ids'][0][0: max_length]
        attention_mask[count] = inputs_tapex['attention_mask'][0][0: max_length]
        token_type_ids[count, 0:length, 0] = cols[0:length]
        token_type_ids[count, 0:length, 1] = rows[0:length]
        token_type_ids[count, 0:length, 2] = ranks[0:length]
        token_type_ids[count, 0:length, 3] = ranks_inv[0:length]
        token_type_ids[count, 0:length, 4] = numeric_relations[0:length]
        #answer = answer.replace('|', ' | ')

        if whole_count_case is True:
            answer = 'col: ' + 'num row' + ' , row: '
            for j in range(1, len(table_2d)):
                answer += str(j)
                if j < len(table_2d) - 1:
                    answer += ', '
            answer += 'agg: count'
        else:
            try:
                answer = 'col: ' + table_2d[0][selected_answer_positions[0][1]] + ' , row: '
                for position in selected_answer_positions:
                    answer += '' + str(position[0]) + ', '
                answer += 'agg: se'
            except:
                continue

        inputs = tokenizer(answer=answer, return_tensors="np")
        ids = list(inputs['input_ids'][0, :])

        length = len(ids)
        if length > answer_length:
            length = answer_length

        label_ids[count, 0: length] = ids[0: length]
        answer_str = answer_str.replace('|', ' | ')
        inputs = tokenizer(answer=answer_str, return_tensors="np")
        ids = list(inputs['input_ids'][0, :])

        length = len(ids)
        if length > answer_length:
            length = answer_length

        label_ids2[count, 0: length] = ids[0: length]
        count += 1
        """
        if len(selected_answer_positions) > 1:
            print(answers, answer_str)
            print(selected_answer_positions)
            print('-------')
        """
        if count % 100 == 0:
            print(count, count2_, count2, count3, count_)
    base_file = 'tapex_inputs_large/'
    np.save(base_file + 'input_ids_wtq_tapex_selection', input_ids[0:count])
    np.save(base_file + 'attention_mask_wtq_tapex_selection', attention_mask[0:count])
    np.save(base_file + 'token_type_ids_wtq_tapex_selection', token_type_ids[0:count])
    np.save(base_file + 'label_ids_wtq_tapex_selection', label_ids[0:count])
    np.save(base_file + 'label_ids_wtq_tapex_selection2', label_ids2[0:count])

    print(count)