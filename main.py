from transformers import ASTFeatureExtractor, TrainingArguments, Trainer, AutoModelForAudioClassification, DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_int8_training
from huggingface_hub import login
#from sklearn.model_selection import train_test_split
import librosa as lbr
import numpy as np
import pandas as pd
import torch
import datasets
import evaluate
import warnings
import logging
warnings.filterwarnings('ignore')
#huggingface.login()
SR = 16000 # Sampling Rate
feature_extractor = ASTFeatureExtractor('MIT/ast-finetuned-audioset-10-10-0.4593')

# Static methods
def parse_df(df):
    audio_data = []
    for i, row in df.iterrows():
        song_name = f"{row['TRACK']}.LOFI.wav"
        path = f'/home/aseliga/Documents/repos/keyidentifier/giantsteps-key-dataset-master/audio/{song_name}'
        # chromagram
        audio = audio_to_array(path)
        logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
        logging.info(f'dtype:   {type(audio)}\nshape:   {audio.shape}\nrow: {i}')
        # intent_class = label2id[row['BEATPORT KEY']]
        song_data = {'audio': {'array':audio, 'path': path, 'sampling_rate': SR}, 'label': row['BEATPORT KEY']}
        audio_data.append(song_data)
    return audio_data

# One hot encoding for labels
def one_h(toks: list):
    label2id, id2label = dict(), dict()
    for i, tok in enumerate(toks):
        label2id[tok] = str(i)
        id2label[str(i)] = tok
    return label2id, id2label
    
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

def audio_to_array(path):
    y, _ =  lbr.load(path, sr=SR)
    y = lbr.feature.chroma_stft(y=y, sr=SR)  # Parse(time,amplitude)
    return np.ndarray.flatten(y)

def preprocess_function(examples):
    audio_arrays = [x for x in examples["input_values"]]
    inputs = feature_extractor(
        audio_arrays, sampling_rate=SR, truncation=True)
    
    return inputs

def map_labels(examples):
    labels = [int(label2id[y]) for y in examples['label']]
    return {'label': labels}

if __name__ == '__main__':
    
    '''
    with open('/home/aseliga/Documents/repos/keyidentifier/giantsteps-key-dataset-master/sources.csv', 'r') as f:
        df = pd.read_csv(f,usecols=['TRACK','BEATPORT KEY'])
        print(df)
    '''
    
    
    
    # Map label names to integer
    # labels = df['BEATPORT KEY'].unique()
    # 1-Hot encode set of labels
    # label2id, id2label = one_h(labels)
    

    # Parse audio data
    #encoded_data = datasets.Dataset.from_list(parse_df(df))
    
    
    #encoded_data = datasets.Dataset.from_parquet('/home/aseliga/Documents/repos/keyidentifier/encoded_data.parquet')
    #jaah = datasets.Dataset.from_parquet('/home/aseliga/Documents/repos/keyidentifier/encoded_JAAH.parquet')
    #cross_era = datasets.Dataset.from_parquet('/home/aseliga/Documents/repos/keyidentifier/encoded_crossera.parquet')
    #encoded_data = datasets.concatenate_datasets([encoded_data,jaah,cross_era])
    encoded_data = datasets.Dataset.from_parquet('encoded_datasets.parquet')
    
    flat_dataset = encoded_data.flatten()
    column_values = flat_dataset['label']
    labels = list(set(column_values))
    print(labels, len(labels))
    label2id, id2label = one_h(labels)

    #encoded_data = encoded_data.add_column('int8_values', int8_values)
    #encoded_data = encoded_data.remove_columns('label')
    #encoded_data = encoded_data.rename_column("int8_values", "label")
    #encoded_data.to_parquet('encoded_data.parquet')
    encoded_data = encoded_data.map(preprocess_function, batched=True, batch_size = 100)
    encoded_data = encoded_data.map(map_labels, batched=True, batch_size=100)
    print(encoded_data['label'])
    encoded_data = encoded_data.train_test_split(test_size=.2)
    
    
    num_labels = len(id2label)
    
    model = AutoModelForAudioClassification.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593',num_labels=num_labels, label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True)
    accuracy = evaluate.load("accuracy")

    training_args = TrainingArguments(
        output_dir="/home/aseliga/Documents/repos/keyidentifier/keyFinder",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-6,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=8,
        num_train_epochs=500,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        bf16=True,
    )
    lora_config = LoraConfig(
        r=20,
        lora_alpha=38,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_data['train'],
        eval_dataset=encoded_data['test'],
        compute_metrics=compute_metrics
    )
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()


    trainer.train() # go crazy
