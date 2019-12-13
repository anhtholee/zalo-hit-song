# Model for Zalo AI Challenge 2019's Hit Song Prediction competition
## Overview
*(taken from the competition site)*
> Hit Song Science is an active research topic in Music Information Retrieval, aiming to predict whether a given song will become a chart-topping hit. Hit prediction is useful to musicians, labels, and music vendors because popular songs generate larger revenues and allow artists to share their message with a broad audience.

> In this track, you are challenged to predict hit rank of a song. You are provided with 9K Vietnamese songs uploaded to Zing MP3 in 2017 and 2018, each song will have:
> - The audio file in .mp3 format;
> - Its metadata (title, composer, singer); and
> - Its ranking from 1 to 10, based on its position in Zing MP3 top charts for a six-month period after uploading date.
> Your task is to predict this rank based on the song content and song information. Although the true rank is an Integer number, your predicted rank can be any Real number. The result is evaluated based on the difference between the predicted rank and the true rank.

> Notes
> - You can only use the provided training dataset. External data samples are prohibited.
> - We reserve the right to inspect submissions and source codes to maintain the integrity of the challenge.

## Data
*(taken from the competition site)*

The provided data consists of two archives of audio files (.mp3 format) and .tsv files with metadata.
- `train.zip` and `test.zip`: the audio files composing the train and public test dataset. (9078 tracks for train and 1118 tracks for public test).
- `train_info.tsv`, `test_info.tsv`: metadata of each song, including: ID, Title, Artist Name, Artist ID, Composer Name, Composers ID, Release Time.
- `train_rank.csv`: the csv file contains the true rank of each song in the train set.
- `sample_submission.csv`: an example of submission for the public test. The submission file should be in .csv format, with 1118 rows and WITHOUT header. The score rounding to 4 decimal places.

## Approach
### Features
I had not the time to experiment mwith actual audio content (spectrogram, MFCC features, etc.). Therefore only the metadata (both from the `csv` file and the `mp3` files) are used. Apart from the metadata features, there are some engineered ones:
- Datetime components (month, year, hour, day of week, etc.) are extracted and each are converted into cyclical components (sine & cosine).
- `artist_is_composer`: whether the artist also composed the song
- `hot_artist`, `hot_composer`: whether the artist/composer of the song is 'hot' (based on average ranking in the training set)
- `n_artists`, `n_composers`: number of artists, composers featured in that song.
- `is_cover`, `is_remix`, `is_beat`, `is_ost`: whether the song title contains the corresponding words.
- Embedding features: Generated from neural network, the idea is from this original paper: [Entity Embeddings of Categorical Variables](http://arxiv.org/abs/1604.06737), which was inspired by Word2Vec's.

### Modelling
Final submission is weighted average of 5 models:
- XGBoost
- CatBoost
- Naive (predict the mean of artist/composers/album/genre ranking for each song)
- MLP (2 hidden layers of 64 & 32 nodes, categorical features represented as embedding matrices)
- XGBoost with embedding features generated from MLP training.

## How to create the submission file (tested on `python 3.7`)
### Package installation
Run `pip install -r requirements.txt`.
### Data Preparation
- Download all `csv` files from the competition site and put them into folder `data` in this working directory.
- Download all `mp3` files from the competition site, extract and put them into `audio-data/train` and `audio-data/test` accordingly.
- To re-generate the metadata info, run the script `gen_metadata.py`. Example:
```shell
python3 gen_metadata.py --audio_dir=./audio-data/train --output_dir=./audio-features --output_name=train_song_metadata.csv
```
and similarly for the test/private audios:
```shell
python3 gen_metadata.py --audio_dir=./audio-data/test --output_dir=./audio-features --output_name=test_song_metadata.csv
python3 gen_metadata.py --audio_dir=./audio-data/private --output_dir=./audio-features --output_name=private_song_metadata.csv
```
Depending on the test set filename, replace `test_info.tsv` to `private_info.tsv` or vice-versa in all scripts (`xgb_cb_naive.py`, `nn_main.py`, `xgb_embedding.py`, `ensemble.py`), the same goes for the extra metadata file (either `private_song_metadata.csv` or `test_song_metadata.csv`)

### Generate embedding features
Run `python3 nn_main.py` to generate predictions as well as embedding matrices for categorical features, using neural net.

### Create predictions
```shell
python3 xgb_cb_naive.py && python3 xgb_embedding.py
```

### Ensemble the predictions (averaging)
```shell
python3 ensemble.py
```

The result would be in `submissions/final_submission.csv`
