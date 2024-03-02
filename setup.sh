# Only for USC HPC Servers
cd ~

# change this to USC Net ID
USERNAME="tereeves"

#ignore this
RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
YELLOW=$(tput setaf 3)
RESET=$(tput sgr0)

echo "${RED}Using username:${RESET} '$USERNAME'. ${RED}this should be exactly your username no spaces${RESET}" 
echo "${RED}ADDING CONDA{$RESET}"
mkdir -p /scratch1/$USERNAME/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /scratch1/$USERNAME/miniconda3/miniconda.sh
bash /scratch1/$USERNAME/miniconda3/miniconda.sh -b -u -p /scratch1/$USERNAME/miniconda3
rm -rf /scratch1/$USERNAME/miniconda3/miniconda.sh
echo "${RED}Copying CONDA this will take a while{$RESET}"
rsync -ah --progress /scratch1/$USERNAME/miniconda3/ ~/miniconda3

~/miniconda3/bin/conda init bash

bash
echo "${RED}Installing pytorch. This will also take a while{$RESET}"
conda create -n mm 

conda activate mm

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

echo "${GREEN} IF you made it this far it means everything probably worked. 'salloc' a gpu and check if torch is available"

# rm -rf /scratch1/$USERNAME/miniconda3/



# rsync -ah --progress /project/msoleyma_1026/DAIC/ /scratch1/$USERNAME/DAIC

# for f in /scratch1/$USERNAME/DAIC/data/*.tar; do tar -xzvf "$f" -C /scratch1/$USERNAME/DAIC/data/; done

# rm -rf /scratch1/$USERNAME/DAIC/data/*.tar



