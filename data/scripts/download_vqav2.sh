mkdir VQA_v2
cd VQA_v2

mkdir annotations
cd annotations
wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Train_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Val_mscoco.zip
unzip v2_Annotations_Train_mscoco.zip
unzip v2_Annotations_Val_mscoco.zip
rm v2_Annotations_Train_mscoco.zip
rm v2_Annotations_Val_mscoco.zip
cd ..

mkdir questions
cd questions
wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Val_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Test_mscoco.zip
unzip v2_Questions_Train_mscoco.zip
unzip v2_Questions_Val_mscoco.zip
unzip v2_Questions_Test_mscoco.zip
cd ..

mkdir images
cd images
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
wget http://msvocds.blob.core.windows.net/coco2015/test2015.zip
unzip train2014.zip
unzip val2014.zip
unzip test2015.zip
rm train2014.zip
rm val2014.zip
rm test2015.zip
cd ..

mkdir bottom_up_attention_36
cd bottom_up_attention_36
wget https://storage.googleapis.com/bottom-up-attention/trainval_36.zip
unzip trainval_36.zip
mv trainval_36 trainval
wget https://storage.googleapis.com/bottom-up-attention/test2015_36.zip
unzip test2015_36.zip
mv test2015_36 test2015
cd ..

mkdir bottom_up_attention_10_100
cd bottom_up_attention_10_100
wget https://storage.googleapis.com/bottom-up-attention/trainval.zip
unzip trainval.zip
wget https://storage.googleapis.com/bottom-up-attention/test2015.zip
unzip test2015.zip
cd ..
