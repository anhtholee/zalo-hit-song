"""
============================
Helper function for audio metadata extraction - for Zalo AI challenge
============================
Author: Le Anh Tho
"""
import pandas as pd
from mutagen.easyid3 import EasyID3
import os
from mutagen.mp3 import MP3

def get_metadata(filepath):
	"""Extract metadata (album, genre, duration) from an audio file
	
	Args:
	    filepath (str): file path

	Returns:
		DataFrame: a dataframe with 1 row containing the metadata
	"""
	song_id = os.path.basename(filepath).split('.')[0]
	
	try:
		id3_ = EasyID3(filepath)
	except Exception as e:
		print("Error encountered with file {}:  {}".format(filepath, e))
		return None
	try:
		mp3_ = MP3(filepath) 
	except Exception as e:
		print("Error encountered with file {}:  {}".format(filepath, e))
		return None
	df = pd.DataFrame([song_id], columns=['id'])
	df['album'] = id3_.get('album', ['unknown'])[0]
	df['genre'] = id3_.get('genre', ['unknown'])[0]
	df['duration'] = mp3_.info.length
	return df.set_index('id').reset_index()