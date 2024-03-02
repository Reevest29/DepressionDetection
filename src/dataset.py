import os
import librosa
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

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


class DIACSpectrogramDataset(Dataset):
    """Spectrogram dataset"""

    def __init__(self, DIAC_loc,train,max_len):
        # get labels and patient nums for split
        train_split,dev_split,_ = get_labels(DIAC_loc)
        data = train_split if train else dev_split
        self.labels = data.loc[:,"PCL-C (PTSD)"]
        self.patient_nums = data.iloc[:,0]
        
        
        self.DIAC_loc = DIAC_loc # data location
        self.max_len = max_len # max length of spectogram segment in time axis

    def __len__(self):
        return len(self.patient_nums)

    def __getitem__(self, idx):
        pat_num = self.patient_nums[idx]
        label = self.labels[idx]
        spec_path = os.path.join(self.DAIC_loc,"data",f"{pat_num}_P",f"{pat_num}_spectrogram.png")
        spectrogram = np.load(spec_path)
        spec_splits = np.split(spectrogram,self.max_len)

        return spec_splits, label





    


if __name__ == "__main__":
    # labels = get_labels(DAIC)
    # for i, label in enumerate(labels):
    #     print(f"{i+1} / 3")
    #     create_spectrograms(label,DAIC)

    spectrogram_dataset = DIACSpectrogramDataset(DAIC,True,100)
    print(spectrogram_dataset[0].shape)