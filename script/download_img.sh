# make dataset dir
mkdir -p ./dataset


# download imagenet training data: you can use aria2 for faster download, or simply use wget
cd ./dataset
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar 

# apt-get update
# apt install aria2
# aria2c -x16 -s16 -k1M https://academictorrents.com/download/a306397ccf9c2ead27155983c254227c0fd938e2.torrent

# extract the training data
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..