mkdir VisualGenome
cd VisualGenome

mkdir annotations
cd annotations
wget http://visualgenome.org/static/data/dataset/image_data.json.zip
unzip image_data.json.zip
rm image_data.json.zip
wget http://visualgenome.org/static/data/dataset/objects.json.zip
unzip objects.json.zip
rm objects.json.zip
wget http://visualgenome.org/static/data/dataset/relationships.json.zip
unzip relationships.json.zip
rm relationships.json.zip
wget http://visualgenome.org/static/data/dataset/region_descriptions.json.zip
unzip region_descriptions.json.zip
rm region_descriptions.json.zip
wget http://visualgenome.org/static/data/dataset/attributes.json.zip
unzip attributes.json.zip
rm attributes.json.zip

cd ..
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
unzip images.zip
unzip images2.zip
rm images.zip
rm images2.zip
