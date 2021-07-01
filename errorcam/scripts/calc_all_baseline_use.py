import sys
sys.path.append("../")
import eval
import cv2



def format_paths(path):
    atten_ims = os.listdir(path)

    all_attens = []
    for att_im in atten_ims:
        att_d = cv2.imread(os.path.join(path, att_im), cv2.IMREAD_GRAYSCALE)
        att_d = cv2.resize(att_d, (7,7))

        qid = att_im.split(".")[0]


        all_attens.append((qid, att_d))
    
    return all_attens

def extract_ans_gtas

if __name__=="__main__":

    #san 2 attentions
    san2_path1 = "../data/san2-1-val"
    san2_path2 = "../data/san2-2-val"

    hiecoatt_path_p = "../data/p_att"
    hiecoatt_path_q = "../data/q_att"
    hiecoatt_path_w = "../data/w_att"

    #format data
    san_attns = format_paths(san2_path1) + format_paths(san2_path2)
    hiecoatt_attns = format_paths(hiecoatt_path_p) + format_paths(hiecoatt_path_q) + format_paths(hiecoatt_path_w)



    

    
    
