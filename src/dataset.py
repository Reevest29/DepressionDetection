import os
import librosa
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch

DAIC = "/scratch1/tereeves/DAIC"

def get_labels(DAIC_loc):
    train_labels = pd.read_csv(os.path.join(DAIC_loc,"labels","train_split.csv"))
    dev_labels = pd.read_csv(os.path.join(DAIC_loc,"labels","dev_split.csv"))
    test_labels = pd.read_csv(os.path.join(DAIC_loc,"labels","test_split.csv"))
    return [train_labels,dev_labels,test_labels]
    

def create_spectrograms(labels,DAIC_loc):
    for x in tqdm(labels.values[:,0]):
        path = os.path.join(DAIC_loc,"data",f"{x}_P",f"{x}_AUDIO.wav")
        audio_data, sample_rate = librosa.load(path)
        spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        new_path = os.path.join(DAIC_loc,"data",f"{x}_P",f"{x}_spectrogram.png")
        np.save(new_path,spectrogram)


class DAICSpectrogramDataset(Dataset):
    """Spectrogram dataset"""

    def __init__(self, DAIC_loc,train,max_len):
        # get labels and patient nums for split
        train_split,dev_split,_ = get_labels(DAIC_loc)
        data = train_split if train else dev_split
        self.labels = data.loc[:,"PCL-C (PTSD)"]
        self.patient_nums = data.iloc[:,0]
        
        
        
        self.DAIC_loc = DAIC_loc # data location
        self.max_len = max_len # max length of spectogram segment in time axis

    def __len__(self):
        return len(self.patient_nums)

    def __getitem__(self, idx):
        pat_num = self.patient_nums[idx] # get patient number
        label = self.labels[idx] # get binary depression label

        # Use patient num to load spetrogram
        spec_path = os.path.join(self.DAIC_loc,"data",f"{pat_num}_P",f"{pat_num}_spectrogram.png.npy")
        spectrogram = np.load(spec_path)

        #split variable len spectrogram into even subsections
        split_idxs = np.arange(start=self.max_len,stop=spectrogram.shape[1],step=self.max_len)
        spec_splits = np.array_split(spectrogram,split_idxs,axis=1)
        
        # pad last spectrogram if short
        pad_len = self.max_len - spec_splits[-1].shape[1]
        spec_splits[-1] = np.pad(spec_splits[-1], ((0,0),(0,pad_len)), 'mean') 

        return spec_splits, label


def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    # import pdb;pdb.set_trace()
    ## get sequence lengths
    lengths = torch.tensor([ len(t) for t in batch ]).to(device)
    ## padd
    sequences = [ torch.tensor(np.array(x)) for (x,_) in batch ]
    labels = [ y for (_,y) in batch ]
    sequences = torch.nn.utils.rnn.pad_sequence(sequences,batch_first=True)
    ## compute mask
    mask = (sequences != 0).to(device)
    return sequences, lengths, mask, labels


    


if __name__ == "__main__":
    # labels = get_labels(DAIC)
    # for i, label in enumerate(labels):
    #     print(f"{i+1} / 3")
    #     create_spectrograms(label,DAIC)
    device = "cpu"
    spectrogram_dataset = DAICSpectrogramDataset(DAIC,True,100)
    train_dataloader = DataLoader(spectrogram_dataset, batch_size=5, shuffle=True,collate_fn=collate_fn_padd)

    sequences, lengths, mask, labels = next(iter(train_dataloader))
    print(sequences.shape)
    
