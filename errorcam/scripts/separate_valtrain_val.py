# separate val attention images into val-train and val splits. 
import os

if __name__=="__main__":

    if False:
        val_path = "data/precomputed_attention_colorcrippled/"
        val_ims = os.listdir(os.path.join(val_path, "val"))

        if not os.path.exists(os.path.join(val_path,"val_train")):
            os.mkdir(os.path.join(val_path,"val_train"))

        val_train_ims = val_ims[:1000]

        for ims in val_train_ims:
            os.rename(os.path.join(os.path.join(val_path, "val"), ims), os.path.join(os.path.join(val_path,"val_train"), ims))

    
    if True: #make sure we have exactly the same val_train and val images as colorcrippled set.
        val_im_fs = os.listdir("data/precomputed_attention_colorcrippled/val_train")

        val_path = "data/precomputed_attention_fullsmall/"
        val_ims = os.listdir(os.path.join(val_path, "val"))

        if not os.path.exists(os.path.join(val_path,"val_train")):
            os.mkdir(os.path.join(val_path,"val_train"))

        for ims in val_im_fs:
            os.rename(os.path.join(os.path.join(val_path, "val"), ims), os.path.join(os.path.join(val_path,"val_train"), ims))

    
