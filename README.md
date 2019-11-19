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
*(taken from competition site*

The data provided consists of two archives of audio files (.mp3 format) and .tsv files with metadata.
- `train.zip` and `test.zip`: the audio files composing the train and public test dataset. (9078 tracks for train and 1118 tracks for public test).
- `train_info.tsv`, `test_info.tsv`: metadata of each song, including: ID, Title, Artist Name, Artist ID, Composer Name, Composers ID, Release Time.
- `train_rank.csv`: the csv file contains the true rank of each song in the train set.
- `sample_submission.csv`: an example of submission for the public test. The submission file should be in .csv format, with 1118 rows and WITHOUT header. The score rounding to 4 decimal places.

## Approach

## How to create the submission file (tested on `python 3.7`)
### Package installation
Run `pip install -r requirements.txt` or `python3 -m pip install -r requirements.txt` (globally or in a virtual enviroment).
### Data Preparation
- Download all `csv` files from the competition site and put them into folder `data` in this working directory.
- Download all `mp3` files from the competition site, extract and put them into `audio-data/train` and `audio-data/test` accordingly.
- To re-generate the metadata info, run the script `gen_metadata.py`. Example:
```shell
python3 gen_metadata.py --audio_dir=./audio-data/train --output_dir=./audio-features --output_name=train_song_metadata.csv
```
and similarly for the test audios:
```shell
python3 gen_metadata.py --audio_dir=./audio-data/test --output_dir=./audio-features --output_name=test_song_metadata.csv
```

### Generate embedding features
Run `python3 nn_main.py` to generate embedding matrices for categorical features.

### Create predictions
```shell
python3 predict_ml.py && python3 main_with_embedding.py
```

### Ensemble the predictions
```shell
python3 ensemble.py
```