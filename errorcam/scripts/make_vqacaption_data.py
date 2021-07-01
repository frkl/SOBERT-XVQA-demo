from collections import defaultdict
import json
import os

if __name__=="__main__":

    coco_data_path = "/data/DataSets/COCO/"

    vqa_data_path = "/data/DataSets/VQA"

    splits=["train", "val"]

    

    format_data = {'images':[]}

    for split in splits:
        vqa_data = json.load(open(os.path.join(vqa_data_path, "OpenEnded_mscoco_"+split+"2014_questions.json")))
        imid2ques = defaultdict(list)
        for entry in vqa_data['questions']:
            tokens = entry['question'].lower().split("?")[0].split(" ")
            imid2ques[entry['image_id']].append({'tokens':tokens})

        for entry in vqa_data['questions']:
            entry_dict = dict()
            entry_dict['cocoid'] = entry['image_id']
            entry_dict['filename'] = "COCO_"+split+"2014_"+str(entry['image_id']).zfill(12)+".jpg"
            entry_dict['filepath'] = split+"2014"
            entry_dict['sentences'] = imid2ques[entry['image_id']]
            entry_dict['split'] = split
            format_data['images'].append(entry_dict)

    with open("models/ImageCaptioningPytorch/data/dataset_vqa.json", "w") as f:
        json.dump(format_data, f)

    

    
        

