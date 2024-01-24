from korquad_eval import exact_match_score


def get_reward_tmp(question, input_table_text, answer_text, predicted_list):  # predicted will be the list of predicted token
    # print(finish, predicted_list)
    try:
        float(answer_text)
        is_number_answer = True
    except:
        is_number_answer = False

    reward = 0
    predicted_words = predicted_list.copy()

    lines = input_table_text.split('[tr]')
    table_2d = []
    for line in lines:
        #print(line)
        tks = line.split('[td]')
        table_2d.append(tks)
    #print()

    has_consecutive_columns = []
    for c in range(len(table_2d[0])):
        is_consecutive = True
        for r in range(1, len(table_2d) - 1):
            try:
                if abs(float(table_2d[r][c]) - table_2d[r + 1][c]) != 1:
                    is_consecutive = False
            except:
                is_consecutive = False
                break
        has_consecutive_columns.append(is_consecutive)

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

    em_scores = [0]

    for col_idx in range(len(table_2d[0])):
        try:
            prediction = ''

            for p, idx in enumerate(row_indices):
                if p > 0:
                    prediction += '|'
                prediction += table_2d[idx][col_idx]

            em_scores.append(exact_match_score(prediction, answer_text))
        except:
            None

    if max(em_scores) >= 1:
        return 1, []

    if is_number_answer is False:
        return -1, []

    agg_loss = [10]
    agg_losses = []

    count_loss = abs(len(row_indices) - float(answer_text))
    if question.find('total') != -1 or question.find('many') != -1 or question.find('number') != -1:
        agg_loss.append(count_loss)

    for col_idx in range(len(table_2d[0])):
        if len(row_indices) == 1:
            # select case
            try:
                sel_loss = abs(float(table_2d[row_indices[0]][col_idx].replace(',', '')) - float(answer_text))
                agg_loss.append(sel_loss)
            except:
                None

        if len(row_indices) == 2:
            if question.find('diff') != -1 or question.find('more') != -1:
                # difference case
                try:
                    diff_value = abs(float(table_2d[row_indices[0]][col_idx].replace(',', '')) -
                                     float(table_2d[row_indices[1]][col_idx].replace(',', '')))
                    diff_loss = abs(diff_value - float(answer_text))
                    agg_loss.append(diff_loss)
                    agg_losses.append(('diff loss:', diff_loss))
                except:
                    None

        if len(row_indices) > 1 and has_consecutive_columns[col_idx] is False:
            if question.find('average') != -1 or question.find('mean') != -1:
                # avg case
                try:
                    sum_value = 0
                    for row_idx in row_indices:
                        sum_value += float(table_2d[row_idx][col_idx].replace(',', ''))
                    avg_value = sum_value / len(row_indices)

                    avg_loss = abs(avg_value - float(answer_text))
                    agg_loss.append(avg_loss)
                    agg_losses.append(('avg loss:', avg_loss))
                except:
                    None

            if question.find('total') != -1 or question.find('sum') != -1 or question.find('aggregat') != -1:
                # sum case
                try:
                    sum_value = 0
                    for row_idx in row_indices:
                        sum_value += float(table_2d[row_idx][col_idx].replace(',', ''))
                    sum_loss = abs(sum_value - float(answer_text))

                    agg_loss.append(sum_loss)
                    agg_losses.append(('sum loss:', sum_loss))
                except:
                    None

    agg_loss = min(agg_loss)
    if agg_loss <= 0.01:
        return 1, agg_losses
    elif agg_loss <= 1.0:
        return -0.1, agg_losses
    elif agg_loss <= 3.0:
        return -0.3, agg_losses
    else:
        return -1, agg_losses


def get_reward_select(input_table_text, answer_text, row_indices, col_idx=None):  # predicted will be the list of predicted token
    # print(finish, predicted_list)
    lines = input_table_text.split('[tr]')
    table_2d = []
    for line in lines:
        tks = line.split('[td]')
        table_2d.append(tks)

    col_indices = []
    if col_idx is not None:
        col_indices.append(col_idx)
    else:
        col_indices.extend(list(range(table_2d[0])))

    has_consecutive_columns = []
    for c in range(len(table_2d[0])):
        is_consecutive = True
        for r in range(1, len(table_2d) - 1):
            try:
                if abs(float(table_2d[r][c]) - table_2d[r + 1][c]) != 1:
                    is_consecutive = False
            except:
                is_consecutive = False
                break
        has_consecutive_columns.append(is_consecutive)

    em_scores = [0]
    for col_idx in range(len(table_2d[0])):
        try:
            prediction = ''

            for p, idx in enumerate(row_indices):
                if p > 0:
                    prediction += '|'
                prediction += table_2d[idx][col_idx]

            em_scores.append(exact_match_score(prediction, answer_text))
        except:
            None

    if max(em_scores) >= 1:
        return 1, []

    return -1, []


def get_reward_count(answer_text, row_indices):  # predicted will be the list of predicted token
    agg_loss = [10]
    agg_losses = []

    count_loss = abs(len(row_indices) - float(answer_text))
    agg_loss.append(count_loss)

    agg_loss = min(agg_loss)
    if agg_loss <= 0.01:
        return 1, agg_losses
    elif agg_loss <= 1.0:
        return -0.1, agg_losses
    elif agg_loss <= 3.0:
        return -0.3, agg_losses
    else:
        return -1, agg_losses


def get_reward_op(question, input_table_text, answer_text, predicted_list):  # predicted will be the list of predicted token
    # print(finish, predicted_list)
    try:
        float(answer_text)
        is_number_answer = True
    except:
        is_number_answer = False

    reward = 0
    predicted_words = predicted_list.copy()

    lines = input_table_text.split('[tr]')
    table_2d = []
    for line in lines:
        #print(line)
        tks = line.split('[td]')
        table_2d.append(tks)
    #print()

    has_consecutive_columns = []
    for c in range(len(table_2d[0])):
        is_consecutive = True
        for r in range(1, len(table_2d) - 1):
            try:
                if abs(float(table_2d[r][c]) - table_2d[r + 1][c]) != 1:
                    is_consecutive = False
            except:
                is_consecutive = False
                break
        has_consecutive_columns.append(is_consecutive)

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

    em_scores = [0]

    for col_idx in range(len(table_2d[0])):
        try:
            prediction = ''

            for p, idx in enumerate(row_indices):
                if p > 0:
                    prediction += '|'
                prediction += table_2d[idx][col_idx]

            em_scores.append(exact_match_score(prediction, answer_text))
        except:
            None

    if max(em_scores) >= 1:
        return 1, []

    if is_number_answer is False:
        return -1, []

    agg_loss = [10]
    agg_losses = []

    diff_check = False

    count_loss = abs(len(row_indices) - float(answer_text))
    if question.find('total') != -1 or question.find('many') != -1 or question.find('number') != -1:
        agg_loss.append(count_loss)

    for col_idx in range(len(table_2d[0])):
        if len(row_indices) == 1:
            # select case
            try:
                sel_loss = abs(float(table_2d[row_indices[0]][col_idx].replace(',', '')) - float(answer_text))
                agg_loss.append(sel_loss)
            except:
                None

        if len(row_indices) == 2:
            if (question.find('diff') != -1 or question.find('more') != -1) and question.find('how many') == -1:
                # difference case
                if row_indices[0] == row_indices[1] and diff_check is False:
                    diff_check = True
                    for c1 in range(len(table_2d[0])):
                        for c2 in range(len(table_2d[0])):
                            if c1 == c2:
                                continue

                            try:
                                row_idx = row_indices[0]
                                value1 = float(table_2d[row_idx][c1].replace(',', ''))
                                value2 = float(table_2d[row_idx][c2].replace(',', ''))

                                diff_value = abs(value1 - value2)
                                diff_loss = abs(diff_value - float(answer_text))
                                agg_loss.append(diff_loss)
                                agg_losses.append(('diff loss:', diff_loss))
                            except:
                                None
                else:
                    try:
                        diff_value = abs(float(table_2d[row_indices[0]][col_idx].replace(',', '')) -
                                         float(table_2d[row_indices[1]][col_idx].replace(',', '')))
                        diff_loss = abs(diff_value - float(answer_text))
                        agg_loss.append(diff_loss)
                        agg_losses.append(('diff loss:', diff_loss))
                    except:
                        None

        if len(row_indices) > 1 and has_consecutive_columns[col_idx] is False:
            if question.find('average') != -1 or question.find('mean') != -1:
                # avg case
                try:
                    sum_value = 0
                    for row_idx in row_indices:
                        sum_value += float(table_2d[row_idx][col_idx].replace(',', ''))
                    avg_value = sum_value / len(row_indices)

                    avg_loss = abs(avg_value - float(answer_text))
                    agg_loss.append(avg_loss)
                    agg_losses.append(('avg loss:', avg_loss))
                except:
                    None

            if question.find('total') != -1 or question.find('sum') != -1 or question.find('aggregat') != -1:
                # sum case
                try:
                    sum_value = 0
                    for row_idx in row_indices:
                        sum_value += float(table_2d[row_idx][col_idx].replace(',', ''))
                    sum_loss = abs(sum_value - float(answer_text))

                    agg_loss.append(sum_loss)
                    agg_losses.append(('sum loss:', sum_loss))
                except:
                    None

    agg_loss = min(agg_loss)
    if agg_loss <= 0.01:
        return 1, agg_losses
    elif agg_loss <= 1.0:
        return -0.1, agg_losses
    elif agg_loss <= 3.0:
        return -0.3, agg_losses
    else:
        return -1, agg_losses