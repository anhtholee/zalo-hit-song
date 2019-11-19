import librosa
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm
import pydub
import os
from PIL import Image
import pylab
import librosa.display
import matplotlib.pyplot as plt
from matplotlib import cm

# https://stackoverflow.com/questions/53633177/how-to-read-a-mp3-audio-file-into-a-numpy-array-save-a-numpy-array-to-mp3?noredirect=1&lq=1
def mp3read(f, normalised=False, sr=22500):
    """Read Mp3 file into a numpy array

    Args:
        f (str): file path
        normalised (bool, optional): whether to normalise the data
        sr (int, optional): Sample rate

    Returns:
        Numpy array: Array of the mp3 file
    """
    a = pydub.AudioSegment.from_mp3(f).set_frame_rate(sr)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        # y = y.reshape((-1, 2))
        y = np.mean(y.reshape((-1, 2)), axis=1)
    if normalised:
        return np.float32(y) / 2 ** 15, a.frame_rate
    else:
        return np.float32(y), a.frame_rate

# Credit goes to: https://gist.github.com/bmcfee/1f66825cef2eb34c839b42dddbad49fd
def ks_key(X):
    """Estimate the key from a pitch class distribution
    
    Args:
        X (np.ndarray, shape=(12,)): Pitch-class energy distribution.
        
    Returns:
        Str: The estimated key
    """
    X = scipy.stats.zscore(X)

    # Coefficients from Kumhansl and Schmuckler
    # as reported here: http://rnhart.net/articles/key-finding/
    major = np.asarray(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    )
    major = scipy.stats.zscore(major)

    minor = np.asarray(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    )
    minor = scipy.stats.zscore(minor)

    # Generate all rotations of major
    major = scipy.linalg.circulant(major)
    minor = scipy.linalg.circulant(minor)

    keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    key_id = np.argmax(np.concatenate((major.T.dot(X), minor.T.dot(X))))
    third = "major" if key_id <= 11 else "minor"
    return keys[key_id % 12] + third


def extract_features(filepath):
    """Extract useful features from an audio file

    Args:
        filepath (str): file path

    Returns:
        DataFrame: a dataframe with 1 row containing the generated features
    """
    try:
        sr = 22050  # sample rate
        n_mfcc = 20  # number of MFCC features
        # Load audio file
        song_id = os.path.basename(filepath).split('.')[0]
        audio, sample_rate = mp3read(filepath, sr=sr)
        duration = audio.shape[0] / sample_rate

        # S = librosa.feature.melspectrogram(audio, sr=sample_rate, n_mels=128)

        # Convert to log scale (dB)
        # log_S = librosa.power_to_db(S, ref=np.max)

        # ==== Tempo ====
        # tempo = librosa.beat.tempo(audio, sr=sample_rate)[0]

        # ==== Zero Crossing Rate ====
        # zrate = librosa.feature.zero_crossing_rate(audio)

        # ==== Spectral centroid ====
        # spec_centroid = librosa.feature.spectral_centroid(
            # y=audio, sr=sample_rate)

        # ==== Spectral roll off ====
        # spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)

        # ==== Spectral flux ====

        # ==== Chroma features ====
        # chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        # if duration < 120:
        #     chroma_cq = librosa.feature.chroma_cqt(y=audio, sr=sample_rate)
        # else:
        #     chroma_cq = librosa.feature.chroma_cqt(y=audio[:sample_rate*120], sr=sample_rate)
        # # Pitch class distribution
        # song_chroma = chroma_cq.mean(axis=1)

        # ==== MFCC features ====
        mfccs = librosa.feature.mfcc(S=log_S, sr=sample_rate, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)

        # Create a DataFrame containing all agg features
        df = pd.concat((
            pd.DataFrame(mfccs_mean.reshape(-1, n_mfcc),
                         columns=['mfcc_mean_' + str(i) for i in range(n_mfcc)]),
            pd.DataFrame(mfccs_std.reshape(-1, n_mfcc),
                         columns=['mfcc_std_' + str(i) for i in range(n_mfcc)]),
        ), axis=1)
        # df = pd.DataFrame([song_id], columns=['id'])
        df['id'] = song_id
        # df['zrate_mean'] = np.mean(zrate)
        # df['zrate_std'] = np.std(zrate)
        # df['zrate_skew'] = scipy.stats.skew(zrate, axis=1)[0]
        # df['spec_centroid_mean'] = np.mean(spec_centroid)
        # df['spec_centroid_std'] = np.std(spec_centroid)
        # df['spec_centroid_skew'] = scipy.stats.skew(spec_centroid, axis=1)[0]
        # df['spec_rolloff_mean'] = np.mean(spec_rolloff)
        # df['spec_rolloff_std'] = np.std(spec_rolloff)
        # df['spec_rolloff_skew'] = scipy.stats.skew(spec_rolloff, axis=1)[0]
        # df['tempo'] = tempo
        # df['duration'] = duration
        # df['key'] = ks_key(song_chroma)


    except Exception as e:
        print("Error encountered with file {}:  {}".format(filepath, e))
        return None

    return df.set_index('id').reset_index()

def extract_spectrogram(output_path, filepath):
    try:
        sr = 12000  # sample rate
        n_mfcc = 20  # number of MFCC features
        # Load audio file
        song_id = os.path.basename(filepath).split('.')[0]
        audio, sample_rate = mp3read(filepath, sr=sr)
        # duration = audio.shape[0] / sample_rate
        # print(audio.shape[0])
        # print(duration)

         # Let's make and display a mel-scaled power (energy-squared) spectrogram
        S = librosa.feature.melspectrogram(audio, sr=sample_rate, 
            fmax = sample_rate/2, # Maximum frequency to be used on the on the MEL scale
            n_fft=2048, 
            hop_length=512, 
            n_mels = 96, # As per the Google Large-scale audio CNN paper
            power = 2) # Power = 2 refers to squared amplitude)

        # Convert to log scale (dB). We'll use the peak power as reference.
        log_S = librosa.power_to_db(S, ref=np.max)
        # mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=20)
        # print(log_S.shape)
        # Save to image
        if not os.path.exists('imgs/' + output_path):
            os.makedirs('imgs/' + output_path)
        # if output == 'train':
        output_name = os.path.join('imgs', output_path, song_id + '.jpg')
        # np.save(output_name, S)
        # im = Image.fromarray(np.uint8(log_S * 255) , 'L')
        # im.thumbnail((1024, 1024), Image.ANTIALIAS)
        # im.save(output_name)
        # Plotting the spectrogram and save as JPG without axes (just the image)
        # pylab.figure(figsize=(3,2))
        # pylab.axis('off') 
        # pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
        fig = plt.figure(frameon=False)
        fig.set_size_inches(0.96,0.96)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        librosa.display.specshow(log_S, cmap=cm.jet)
        # pylab.savefig(output_name, bbox_inches=None, pad_inches=0)
        # pylab.close()
        fig.savefig(output_name)
        plt.close()

    except Exception as e:
        print("Error encountered with file {}:  {}".format(filepath, e))
        return None