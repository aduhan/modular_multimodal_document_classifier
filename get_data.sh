# Create a folder, cd into it, download the data, extract it and remove the tar file

mkdir -p data
cd data

wget 'https://huggingface.co/datasets/rvl_cdip/resolve/main/data/rvl-cdip.tar.gz'
tar xvf rvl-cdip.tar.gz

rm rvl-cdip.tar.gz