from collections import defaultdict
import json
import os
import csv
import pdb

with open("/data/ARIJIT/Documents/ConsistentVQA/data/commonsenseConData/ConVQA_verified.json") as f:
            convqa_verified = json.load(f)

with open("/data/DataSets/VQA/3000_ans_list.json") as f:
    vqa_ans_list = json.load(f)

with open("/data/DataSets/VQA/COCO_imid_to_VQA_qas.json") as f:
    vqa_qas = json.load(f)

common_answers = defaultdict(dict)
for im in vqa_qas:
    for q,a, _ in vqa_qas[im]:
        q_key = " ".join(q.strip().lower().split("?")[0].split(" ")[:3])
        common_answers[q_key][a] = 1


result_dir = "/data/ARIJIT/Documents/ConsistentVQA/data/MTURK_files/results_files/"
res_files = os.listdir(result_dir)
data_list = []
for mturk_file in res_files:
    with open(os.path.join("/data/ARIJIT/Documents/ConsistentVQA/data/MTURK_files/results_files/", mturk_file), "r") as f:
        data = csv.DictReader(f)
        for entry in data:
            data_list.append(entry)

im_qas = defaultdict(list)

for line in data_list[::-1]:

    approved = str(line['AssignmentStatus'])
    if approved != "Approved":
        continue;

    ques_choice = str(line['Answer.q_choice'])
    image = line['Input.image_0']

    set1 = []

    inp_ques = line['Input.question_' + ques_choice.split("|")[0].split("_")[-1] + '0']
    inp_ans = line['Input.answer_' + ques_choice.split("|")[0].split("_")[-1] + '0']

    set1.append((inp_ques, inp_ans))

    cons_ques_1 = line['Answer.ques_01'].strip().lower()
    cons_ans_1 = line['Answer.ans_01'].strip().lower()

    set1.append((cons_ques_1, cons_ans_1))

    cons_ques_2 = line['Answer.ques_02'].strip().lower()
    cons_ans_2 = line['Answer.ans_02'].strip().lower()

    set1.append((cons_ques_2, cons_ans_2))

    try:
        cons_ques_3 = line['Answer.ques_03'].strip().lower()
        cons_ans_3 = line['Answer.ans_03'].strip().lower()
        set1.append((cons_ques_3, cons_ans_3))

        cons_ques_4 = line['Answer.ques_04'].strip().lower()
        cons_ans_4 = line['Answer.ans_04'].strip().lower()
        set1.append((cons_ques_4, cons_ans_4))
    except:
        a=1 # do nothing

    im_qas[image].append(set1)

    try:
        ques_choice = str(line['Answer.q_choice'])
        image = line['Input.image_1']
        inp_ques = line['Input.question_' + ques_choice.split("|")[1].split("_")[-1] + '1']
        inp_ans = line['Input.answer_' + ques_choice.split("|")[1].split("_")[-1] + '1']
        set2 = []
        set2.append((inp_ques, inp_ans))

        cons_ques_1 = line['Answer.ques_11'].strip().lower()
        cons_ans_1 = line['Answer.ans_11'].strip().lower()

        set2.append((cons_ques_1, cons_ans_1))

        cons_ques_2 = line['Answer.ques_12'].strip().lower()
        cons_ans_2 = line['Answer.ans_12'].strip().lower()

        set2.append((cons_ques_2, cons_ans_2))
        #pdb.set_trace()
        im_qas[image].append(set2)
    except:
        a=1

#pdb.set_trace()
#im_qas = defaultdict(list)
#replace stuff with verified stuff if verified.

for hitid in convqa_verified:
    image = convqa_verified[hitid]['image']
    im_qas[image] = []

for hitid in convqa_verified:
    image = convqa_verified[hitid]['image']
    qas = convqa_verified[hitid]['ogConQA']
    ratings = convqa_verified[hitid]['ratings']
    rate_nums = {"intelligent": 10, "simple": 5, "not": 0, "incorrect": 0}
    q_ratings = []
    ratings = list(zip(*ratings))
    q_ratings.append(10)
    keep_qas = [qas[0]]
    for rate_entry, qa in zip(ratings, qas[1:]):
        #rating = [rate_nums[i] for i in rate_entry]
        #rating = np.average(rating)
        #if rating>=5 and rating<=9:
        rating = max(set(rate_entry), key=rate_entry.count)
        if rating not in ["not", "incorrect"]:
            keep_qas.append(qa)

    im_qas[image].append(keep_qas)

#pdb.set_trace()
#clean answers and questions, format. 
new_im_qas = dict()
for im in im_qas:
    con_qas_sets = im_qas[im]
    all_clean_qas = []
    for con_qas in con_qas_sets:
        q0, a0 = con_qas[0]
        q0 = q0.strip().lower().split("?")[0]
        a0 = a0.strip().lower().split(".")[0]
        clean_qas =[[q0,a0]]
        for ques, ans in con_qas[1:]:
            ans = ans.lower().strip().split(",")[0].split("?")[0].split(".")[0]
            if ans in vqa_ans_list:
                clean_ans = ans
            else:
                a_ws = ans.split(" ")
                if len(a_ws)>3:
                    for a_w in a_ws:
                        if a_w[::-1] in vqa_ans_list:
                            clean_ans = a_w
                            break
                        continue
                    #clean_ans = a_w
                else:
                    continue

            clean_qas.append((ques.lower().split("?")[0].split(".")[0], clean_ans))
        all_clean_qas.append(clean_qas)
    new_im_qas[im] = all_clean_qas
#pdb.set_trace()
im_qas = new_im_qas

#pdb.set_trace()
with open("data/consistentVQAsets/updated_commonsense_conVQA_consistent.json", "w") as f:
    json.dump(im_qas, f)