echo 'Please refer to https://github.com/JiahuiYu/generative_inpainting and https://github.com/zzzace2000/FIDO-saliency for licenses of respective components.'  



#TODO: Download VQA model checkpoints from Dropbox
wget https://www.dropbox.com/s/satczbns26q5nfa/models.zip?dl=1 -O models.zip
unzip models.zip -d ./res/models/

#TODO: Download errorcam model checkpoints from Dropbox
wget https://www.dropbox.com/s/n06utaqo5ftusbn/errorcam_checkpoints.zip?dl=1 -O errorcam_checkpoints.zip
unzip errorcam_checkpoints.zip -d ./errorcam/

#Download pretrained model from bottom-up-attention repo 
wget https://storage.googleapis.com/up-down-attention/resnet101_faster_rcnn_final.caffemodel ./bottom-up-attention/data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel 

git clone https://github.com/zzzace2000/FIDO-saliency ./FIDO-saliency
mv FIDO-saliency/arch/* ./arch/
mv FIDO-saliency/exp ./

git clone https://github.com/zzzace2000/generative_inpainting ./generative_inpainting
cd ./generative_inpainting
 
mkdir model_logs  
cd model_logs 
mkdir release_imagenet_256 
cd release_imagenet_256 

#Download the pretrained inpainting model 
../../../googledown.sh 15rNo7ZpxVlbQn9-UzKJ6o4BfP6pQj8jg checkpoint
../../../googledown.sh 1e3A7BgFHyTh9VuNqPj4hxJbSC2OV096W snap-0.index
../../../googledown.sh 18gjYI44AOxN1zv8vcNDY5jhtcgABkJmm snap-0.meta
../../../googledown.sh 1Ns-uezoLdVOOAKma2xWWBPXS4QOpxSYh snap-0.data-00000-of-00001
 
