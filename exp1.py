from transformers import TrainingArguments, Trainer, ASTForAudioClassification, ASTFeatureExtractor
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_int8_training
import datasets
import warnings
import logging
import torch
import sys
from utils import preprocess_chromagrams, map_labels, one_h, compute_metrics



# Constants
warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f'{DEVICE}')
SR = 16000

model_checkpoint = 'MIT/ast-finetuned-audioset-10-10-0.4593'
FEATURE_EXTRACTOR = ASTFeatureExtractor(sys.argv[0])
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)




if __name__ == '__main__':
    logging.info('Parsing dataset...')

    encoded_data = datasets.Dataset.from_parquet('/home/aseliga/Documents/repos/keyidentifier/encoded_datasets.parquet')

    
    logging.info('Preprocessing features...')
    encoded_data = encoded_data.map(preprocess_chromagrams, batched=True, batch_size=200, fn_kwargs={'feature_extractor': FEATURE_EXTRACTOR})
    logging.info('Preprocessing labels...')
    
    encoded_data = encoded_data.map(map_labels, batched=True, batch_size=200)
    logging.info(f'{encoded_data.column_names}')
    encoded_data = encoded_data.remove_columns('label')
    encoded_data = encoded_data.rename_column('new_label', 'label')

  
    logging.info('Storing data...')
    torch.cuda.empty_cache()
    logging.info(f'{encoded_data}')
    label2id, id2label = one_h(list(set(encoded_data['label'])))
    encoded_data = encoded_data.train_test_split(test_size=.5)

 
    train_dataset = encoded_data['train']
    val_dataset = encoded_data['test']

    
    
    logging.info('Configuring hyperparameters...')
    """
    config = ASTConfig(
        hidden_size=768,  # Default value, can be adjusted based on model capacity requirements
        num_hidden_layers=16,  # Standard number of layers, can be reduced for smaller datasets or limited resources
        num_attention_heads=16,  # Standard number of attention heads
        intermediate_size=3072,  # Typical size for the feed-forward layer in transformers
        hidden_act="gelu",  # Standard activation function, 'gelu' is commonly used
        hidden_dropout_prob=0.1,  # Can be adjusted to prevent overfitting; start with 0.1
        attention_probs_dropout_prob=0.1,  # Similar to hidden_dropout_prob, can be adjusted based on your dataset
        initializer_range=0.02,  # Standard initializer range for transformer models
        layer_norm_eps=1e-12,  # Standard epsilon value for layer normalization
        patch_size=8,  # Adjust based on the resolution of your chromagrams, may require experimentation
        qkv_bias=True,  # Standard to include bias in queries, keys, and values
        frequency_stride=12,  # Set to 1 for chromagrams as they have 12 pitch classes
        time_stride=12,  # Can be adjusted based on the temporal resolution of your chromagrams
        max_length=200,  # Should match the temporal dimension of your chromagrams
        num_mel_bins=16
    )
    
    """
    # build new model class
    model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", num_labels=24, label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True) 

    training_args = TrainingArguments(
        output_dir="./keyFinder",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-6,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,
        num_train_epochs=500,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        bf16=True,
        logging_dir='./logs'
    )
    lora_config = LoraConfig(
        r=20,
        lora_alpha=38,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="all",
        task_type=TaskType.FEATURE_EXTRACTION
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        compute_metrics=compute_metrics
    )
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()



    


    logging.info('Starting Training...')
    trainer.train() # <-- training function from class CustomTrainer()
    
    
    
