import os
import json
import pdb

if __name__=="__main__":

    convqa = json.load(open("/dataSRI/ARIJIT/attention_improve/data/consistentVQASets/updated_commonsense_conVQA_consistent.json"))

    #precomputed attention data
    base_path = "data/precomputed_attention_colorcrippled/val"
    atten_data = os.listdir(base_path)

    imq2conqa = dict()
    for im_file in convqa:
        imid = im_file.split("_")[-1].split(".")[0]

        for entry in convqa[im_file]:
            vqaq= entry[0][0].split("?")[0].lower()
            imq2conqa[imid+"_"+vqaq]=convqa[im_file]


    convqa_humanatt = []
    for att_file in atten_data:
        coco_id, question, answer, gt_a, max_s, attn = json.load(open(os.path.join(base_path, att_file)))
        question = question.split("?")[0].lower()
        imid = str(coco_id).zfill(12)
        pdb.set_trace()
        if imid+"_"+question in imq2conqa:
            convqa_humanatt.append(att_file)

    with open("data/consistentVQASets/ConVQA_HumanAtt_IntersectionQI.json", "w") as f:
        json.dump(convqa_humanatt, f)


