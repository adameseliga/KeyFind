from torch.utils.data import Dataset
import evaluate
import torch
import datasets
import warnings
import librosa as lbr
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import logging
import glob
import json
import concurrent.futures
import re

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
accuracy = evaluate.load("accuracy")
note_mapping = {
    "C#": "Db",
    "D#": "Eb",
    "F#": "Gb",
    "G#": "Ab",
    "A#": "Bb",
    "C": "C",
    "D": "D",
    "E": "E",
    "F": "F",
    "G": "G",
    "A": "A",
    "B": "B",
    "Db": "Db",
    "Eb": "Eb",
    "Gb": "Gb",
    "Ab": "Ab",
    "Bb": "Bb"
}
SR = 16000
chroma_meta = []
# Dataset Class
class ChromagramDataset(Dataset):
    def __init__(self, chromagrams, labels):
        self.chromagrams = chromagrams
        self.labels = labels

    def __len__(self):
        return len(self.chromagrams)

    def __getitem__(self, idx):
        chromagram = torch.tensor(self.chromagrams[idx], dtype=torch.float32, device='cuda')
        label = torch.tensor(self.labels[idx], dtype=torch.int8, device='cuda')
        return {"input_values": chromagram, "labels": label}
    
# Giant Steps Key Dataset master processing function
def parse_df(df):
    audio_data = []
    for _, row in df.iterrows():
        song_name = f"{row['TRACK']}.LOFI.wav"
        path = f'/home/aseliga/Documents/repos/keyidentifier/giantsteps-key-dataset-master/audio/{song_name}'
        # chromagram
        audio = audio_to_array(path)
        #logging.info(f'dtype:   {type(audio)}\nshape:   {audio.shape}\nrow: {i} {type(audio[0])}')
        # intent_class = label2id[row['BEATPORT KEY']]
        song_data = {'input_values': audio, 'label': row['BEATPORT KEY']}
        audio_data.append(song_data)
    return audio_data

# Parse raw audio into np ndarray
def audio_to_array(path):
    y, _ =  lbr.load(path, sr=SR)
    y = lbr.feature.chroma_stft(y=y, sr=SR).tolist() # Parse(time,amplitude)
    logging.info(f'LENGTH:  {len(y)}') 
    y = np.asarray(y)
    logging.info(f'SHAPE:   {y.shape}')
    return pad_or_truncate(y)

# Preprocess JAAH
def preprocess_JAAH(annotations, features):
    # Parses label for every example
    def get_annotations(annotations=annotations):
        file_paths = glob.glob(f'{annotations}/*')
        labels = []
        for path in file_paths:
            with open(path, 'r') as json_file:
                data = json.load(json_file)
            labels.append(data['key'])
            logging.info(f'LABEL shape | JAAH:   {type(data["key"])}')
        return np.array(labels)
    # Parses Chroma features for every example
    def get_features(features=features):
        file_paths = glob.glob(f'{features}/*')
        features = []
        for path in file_paths:
            df = pd.read_csv(path, engine='c').iloc[:, 1:]
            features.append(pad_or_truncate(df.to_numpy(dtype='float64').T))
            logging.info(f'FEATURE shape | JAAH:   {df.shape}')
            chroma_meta.append(df.shape)
        return features

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        logging.info('Grabbing features and labels...')
        t1 = executor.submit(get_annotations, annotations)
        t2 = executor.submit(get_features, features)
        labels = t1.result()
        input_values = t2.result()
        logging.info('Done!')
    

    df = datasets.Dataset.from_dict({'input_values': input_values, 'label': labels})
    '''
    for i, arr in enumerate(input_values):2
        logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
        logging.info(f'dtype:   {type(arr)}\nshape:   {arr.shape}\nrow: {i}')
        intent_class = label2id[labels[i]]
        song_data = {'audio': {'array': arr}, 'label':int(intent_class)}
        audio_data.append(song_data)
    '''
    return df

# Get the CrossEra Chromagrams
def get_crossera_features(path):
    features = {}
    df = pd.read_csv(path, header=None, engine='c')
    mask = df[1] == 0
    indices = df[mask].index.tolist()
    for i, row in enumerate(indices):
        song_name = df.loc[row,0].split('/')[1]
        if i < len(indices)-1:
            chroma_data = df.iloc[row:indices[i+1], [2,3,4,5,6,7,8,9,10,11,12,13]]
        else:
            chroma_data = df.iloc[row:, [2,3,4,5,6,7,8,9,10,11,12,13]]

        chroma_data = chroma_data.to_numpy(dtype='float64')
        features[song_name] = pad_or_truncate(chroma_data.T)
        chroma_meta.append(features[song_name].shape)

    return features
    
# Get the CrossEra labels (key signatures)
def get_crossera_labels(path):
    labels = {}
    df = pd.read_csv(path, engine='c')
    
    for _, row in df.iterrows():
        key_signature = f'{row["Key"]} {str(row["Mode"]).lower()}'
        song_name = row['Filename']
        labels[song_name] = key_signature
    return labels

# Preprocess CrossEra - returns Dataset of features and labels
def preprocess_crossera(features, labels):
    audio_data = []
    for song_name in labels.keys():
        try:
            song_data = {'input_values': features[song_name], 'label': labels[song_name]}
            audio_data.append(song_data)
        except KeyError:
            print(f'Key not found: {song_name}')
    
    #logging.info(f'shape:   {audio_data}')
    audio_data = datasets.Dataset.from_list(audio_data)
    # logging.info(f'shape:   {np.shape(audio_data)}')
    return audio_data

# One-hot encoding for labels
def one_h(toks: list):
    label2id, id2label = dict(), dict()
    for i, tok in enumerate(toks):
        label2id[str(tok)] = i
        id2label[str(i)] = tok
    return label2id, id2label

# Convert audio file to spectrogram
def parse_audio(path, SR=16000):
    y, _ =  lbr.load(path, sr=SR)
    y = lbr.feature.chroma_stft(y=y, sr=SR).tolist() # Parse(time,amplitude)
    
    return np.ndarray.flatten(pad_or_truncate(y))

def map_labels(examples):
    label_set = list(set(examples['label']))
    label2id, _ = one_h(label_set)
    labels = [int(label2id[y]) for y in examples['label']]
    print(f"First few new labels: {labels[:5]}")
    return {'new_label': labels}
# Map sharp keys to flats
def map_keys(examples):
    keys = [f"{note_mapping[x.split(' ')[0]]} {x.split(' ')[1]}" for x in examples['label']]
    return {"label": keys}

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

def pad_or_truncate(array, y_length=200):
    """
    Pads or truncates the array to the target length along its first dimension.

    :param array: A numpy array of shape (x, 12).
    :param target_length: The target length of the first dimension.
    :return: A numpy array of shape (target_length, 12).
    """
    current_length = array.shape[1]
    
    if current_length < y_length:
        # Pad the array
        padding = np.zeros((y_length - current_length, 12))
        array = np.concatenate((array, padding), axis=1)
    elif current_length > y_length:
        # Truncate the array
        array = array[:y_length, :]
    pad_width = (58, 58), (0, 0)  #((top_pad, bottom_pad), (0, 0))

    return np.pad(array, pad_width, mode='constant', constant_values=0).T

def preprocess_chromagrams(examples, feature_extractor):

    audio_arrays = [np.array(x).flatten() for x in examples["input_values"]]

    inputs = feature_extractor(
           audio_arrays, sampling_rate=SR, max_length=16000, truncation=True, padding=True)
        
    return inputs
def find_inhomogeneous_shapes(dataset, column_name):
    shapes = set()
    inhomogeneous_examples = []
    i = 0
    for example in dataset[column_name]:
        i += 1
        logging.info(f'feature {i}')
        # Convert to a numpy array for consistent shape handling
        shape = len(example)
        shapes.add(shape)

        # Assuming you're looking for elements that don't have a uniform shape
        if len(shapes) > 1:
            inhomogeneous_examples.append((shape, example))

    return shapes, len(inhomogeneous_examples)

if __name__ == '__main__':
    
    logging.info('Preprocessing GiantSteps...')
    with open('/home/aseliga/Documents/repos/keyidentifier/giantsteps-key-dataset-master/sources.csv', 'r') as f:
        df = pd.read_csv(f,usecols=['TRACK','BEATPORT KEY'])
    giantsteps_data = datasets.Dataset.from_list(parse_df(df))
    giantsteps_data.to_parquet('giantsteps.parquet')
    
    
    
    #giantsteps_data = datasets.Dataset.from_parquet('giantsteps.parquet')
    

    logging.info('Preprocessing JAAH...')
    jaah_annotations = '/home/aseliga/Documents/repos/keyidentifier/JAAH-master/annotations'
    jaah_features = '/home/aseliga/Documents/repos/keyidentifier/JAAH-master/feature'
    jaah_data = preprocess_JAAH(jaah_annotations, jaah_features)
    
    
    logging.info('Preprocessing CrossEra...')
    orchestra_baroque = get_crossera_features('cross-era/annotations/chroma-nnls_orchestra_baroque.csv')
    orchestra_classical = get_crossera_features('cross-era/annotations/chroma-nnls_orchestra_classical.csv')
    orchestra_romantic = get_crossera_features('cross-era/annotations/chroma-nnls_orchestra_romantic.csv')
    piano_baroque = get_crossera_features('cross-era/annotations/chroma-nnls_piano_baroque.csv')
    piano_classical = get_crossera_features('cross-era/annotations/chroma-nnls_piano_classical.csv')
    piano_romantic = get_crossera_features('cross-era/annotations/chroma-nnls_piano_romantic.csv')
    features = [orchestra_baroque,orchestra_classical,orchestra_romantic,piano_baroque,piano_classical,piano_romantic]
    labels = get_crossera_labels('cross-era/cross-era_annotations.csv')
    audio_data = []
    for feature in features:
        audio_data.extend(preprocess_crossera(feature, labels))
    audio_data = datasets.Dataset.from_list(audio_data)

    logging.info('Preprocessing finished! Building Dataset... \n'+
                 f'{audio_data}|{giantsteps_data}|{jaah_data}')
    encoded_data = datasets.concatenate_datasets([giantsteps_data,jaah_data,audio_data])
    
    # encoded_data = datasets.concatenate_datasets([giantsteps_data,jaah_data])
    # encoded_data = giantsteps_data
    # Filter labels to match pattern
    pattern = re.compile(r'^[A-G](#|b)? (major|minor)$')
    encoded_data = encoded_data.filter(lambda row: pattern.match(row['label']) is not None)
    encoded_data = encoded_data.map(map_keys, batched=True, batch_size=100)
    encoded_data.to_parquet('encoded_datasets.parquet')
    
    #encoded_df = pd.read_parquet('encoded_datasets.parquet').to_numpy()
    #np.save('encoded_datasets.npy', encoded_df)

    #encoded_data.to_parquet('encoded_datasets.parquet')
    logging.info('Dataset built!')
