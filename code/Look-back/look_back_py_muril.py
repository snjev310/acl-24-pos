# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import brown
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from sklearn.metrics import confusion_matrix
from datasets import Dataset
import warnings
import os
warnings.filterwarnings('ignore')
import random
random.seed(28)


# %%
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# %%
df = pd.read_excel('Data_merged_final_2_jan_31_jan_5_feb_update.xlsx')
df.head(1)


# %%
def combined_words(token,label):
    word_list = []
    token_list = []
    pos_list = []
    op_list = []
    temp_list = []
    for item in zip(token,label):
        word_list.append(item)
    for item in word_list:
        temp_list.append(item)
        if item[0]=='।' or item[0]=='?':
            op_list.append(temp_list)
            temp_list=[]
    for sentences in op_list:
        sent = []
        pos = []
        for item in sentences:
            sent.append(str(item[0]))
            #print(sent)
            pos.append(list(item)[1].upper())
        token_list.append(sent)
        pos_list.append(pos)
    df = pd.DataFrame(list(zip(token_list,pos_list)),columns=['tokens','pos'])
    return df


# %%
hindi_df = combined_words(df['hindi_token'],df['hindi_upos'])
print("Hindi Corpus: ",hindi_df.shape)
angika_df = combined_words(df['angika_token'],df['angika_upos'])
print("Angika Corpus: ",angika_df.shape)
magahi_df = combined_words(df['magahi_token'],df['magahi_upos'])
print("Magahi Corpus: ",magahi_df.shape)
bhojpuri_df = combined_words(df['bhojpuri_token'],df['bhojpuri_upos'])
print("Bhojpuri Corpus: ",bhojpuri_df.shape)


# %%
def replace_upos_pos(row):
    upos = []
    for item in row:
        item = item.strip().lower()
        if item=='noun' or item=='NOUN':
            item = 0
        elif item == 'sym':
            item = 4
        elif item == 'adp' or item=='ado':
            item = 2
        elif item == 'num' or item == 'NUM':
            item = 3
        elif item == 'punct':
            item = 1
        elif item=='sconj':
            item=5
        elif item=='adj':
            item=6
        elif item=='part':
            item=7
        elif item=='det' or item=='DET':
            item=8
        elif item=='cconj' or item == 'CONJ' or item=='conj':
            item= 9
        elif item=='propn' or item == 'PROPN':
            item = 10
        elif item=='pron' or item =='PRON':
            item = 11
        elif item=='unk' or item=='UNK':
            item = 12
        elif item=='X' or item =='x':
            item = 13
        elif item=='adv' or item=='ADV':
            item = 14
        elif item == 'intj' or item=='INTJ':
            item = 15
        elif item == 'verb' or item == 'VERB':
            item = 16
        elif item == 'VAUX' or item=='vaux' or item=='aux':
            item = 17
        upos.append(item)
    return upos 


# %%
hindi_df['upos'] = hindi_df['pos'].apply(replace_upos_pos)
angika_df['upos'] = angika_df['pos'].apply(replace_upos_pos)
magahi_df['upos'] = magahi_df['pos'].apply(replace_upos_pos)
bhojpuri_df['upos'] = bhojpuri_df['pos'].apply(replace_upos_pos)

# %%
dataset_hi_test = Dataset.from_pandas(hindi_df)
dataset_ang_test = Dataset.from_pandas(angika_df)
dataset_mag_test = Dataset.from_pandas(magahi_df)
dataset_bho_test = Dataset.from_pandas(bhojpuri_df)

# %%
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased",strip_accents=False)


# %%
from transformers import BertTokenizerFast 
from transformers import DataCollatorForTokenClassification 
from transformers import AutoModelForTokenClassification


# %%
import datasets
from datasets import load_dataset
dataset_hi_train = load_dataset("universal_dependencies", "hi_hdtb")


# %%
def tokenize_and_align_labels(example, label_all_tokens=True):
    tokenized_input = tokenizer(example['tokens'], truncation=True, is_split_into_words=True)
    #print('examples: ',tokenized_input)

    labels = []
    for i,label in enumerate(example['upos']):
        word_ids = tokenized_input.word_ids(batch_index=i)

        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                #set -100 as the label for these special tokens
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                #of current word_idx is != previous then its the most regular case and add the corresponding token
                label_ids.append(label[word_idx])
            else:
                #to take care of sub-words which have the same word_idx set -100 as well for them, but only if label_all_tokens==False
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_input['labels'] = labels
    return tokenized_input


# %%
tokenized_datasets_hi_train = dataset_hi_train.map(tokenize_and_align_labels,batched=True)

tokenized_datasets_hi_test = dataset_hi_test.map(tokenize_and_align_labels,batched=True)
tokenized_datasets_ang_test = dataset_ang_test.map(tokenize_and_align_labels,batched=True)
tokenized_datasets_mag_test = dataset_mag_test.map(tokenize_and_align_labels,batched=True)
tokenized_datasets_bho_test = dataset_bho_test.map(tokenize_and_align_labels,batched=True)


# %%
label_list = dataset_hi_train["train"].features["upos"].feature.names


# %%
model = AutoModelForTokenClassification.from_pretrained("google/muril-base-cased",num_labels=18)


# %%
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
"test-upos",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
)


# %%
data_collator = DataCollatorForTokenClassification(tokenizer)
metric = datasets.load_metric("seqeval")


# %%
def compute_metrics(eval_preds):
    pred_logits,labels = eval_preds

    pred_logits = np.argmax(pred_logits,axis=2)

    predictions = [
        [label_list[eval_preds] for (eval_preds,l) in zip(prediction,label) if l!= -100] for prediction,label in zip(pred_logits,labels)
    ]

    true_labels = [
        [label_list[l] for (eval_preds,l) in zip(prediction,label) if l!= -100] for prediction,label in zip(pred_logits,labels)
    ]

    results = metric.compute(predictions=predictions,references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
        "true_labels":true_labels,
        "predictions":predictions
    }


# %%
def training_(train_data,eval_data):
    trainer = Trainer( 
    model,
    args,
   train_dataset=train_data,
   eval_dataset=eval_data,
   data_collator=data_collator,
   tokenizer=tokenizer,
   compute_metrics=compute_metrics
)
    return trainer.train()


# %%
train = training_(tokenized_datasets_hi_train['train'],tokenized_datasets_hi_train['validation'])


# %%
def evaluate_(eval_data):
    trainer = Trainer( 
    model, 
    args, 
   #train_dataset=tokenized_datasets["test"], 
   eval_dataset=eval_data, 
   data_collator=data_collator, 
   tokenizer=tokenizer, 
   compute_metrics=compute_metrics 
)
    return trainer.evaluate()


# %%
hindi_eval_result = evaluate_(tokenized_datasets_hi_test)
magahi_eval_result = evaluate_(tokenized_datasets_mag_test)
angika_eval_result = evaluate_(tokenized_datasets_ang_test)
bhojpuri_eval_result = evaluate_(tokenized_datasets_bho_test)

# %%
print('-'*10,"Hindi Results",'-'*10)
print("Precision: ",hindi_eval_result['eval_precision'],"\nRecall: ",hindi_eval_result['eval_recall'],"\nF1-Score",hindi_eval_result['eval_f1'])
print('-'*10,"Magahi Results",'-'*10)
print("Precision: ",magahi_eval_result['eval_precision'],"\nRecall: ",magahi_eval_result['eval_recall'],"\nF1-Score",magahi_eval_result['eval_f1'])
print('-'*10,"Angika Results",'-'*10)
print("Precision: ",angika_eval_result['eval_precision'],"\nRecall: ",angika_eval_result['eval_recall'],"\nF1-Score",angika_eval_result['eval_f1'])
print('-'*10,"Bhojpuri Results",'-'*10)
print("Precision: ",bhojpuri_eval_result['eval_precision'],"\nRecall: ",bhojpuri_eval_result['eval_recall'],"\nF1-Score",bhojpuri_eval_result['eval_f1'])

# %%
hindi_true_label = hindi_eval_result['eval_true_labels']
hindi_predicted_label = hindi_eval_result['eval_predictions']

angika_true_label = angika_eval_result['eval_true_labels']
angika_predicted_label = angika_eval_result['eval_predictions']

magahi_true_label = magahi_eval_result['eval_true_labels']
magahi_predicted_label = magahi_eval_result['eval_predictions']

bhojpuri_true_label = bhojpuri_eval_result['eval_true_labels']
bhojpuri_predicted_label = bhojpuri_eval_result['eval_predictions']

# %%
print(tokenized_datasets_hi_test['tokens'][0])
print(hindi_true_label[0])
print(hindi_predicted_label[0])


# %%
hindi_token_new = []
for i in range(len(hindi_df)):
    hindi_token_new.append(tokenizer.convert_ids_to_tokens(tokenized_datasets_hi_test["input_ids"][i]))
    
angika_token_new = []
for j in range(len(angika_df)):
    angika_token_new.append(tokenizer.convert_ids_to_tokens(tokenized_datasets_ang_test["input_ids"][j]))

magahi_token_new = []
for k in range(len(magahi_df)):
    magahi_token_new.append(tokenizer.convert_ids_to_tokens(tokenized_datasets_mag_test["input_ids"][k]))

bhojpuri_token_new = []
for k in range(len(bhojpuri_df)):
    bhojpuri_token_new.append(tokenizer.convert_ids_to_tokens(tokenized_datasets_bho_test["input_ids"][k]))



# %%
hindi_token = sum(hindi_token_new,[])
hindi_token = [x for x in hindi_token if x != '[CLS]' and x!='[SEP]']

angika_token = sum(angika_token_new,[])
angika_token = [x for x in angika_token if x != '[CLS]' and x!='[SEP]']

magahi_token = sum(magahi_token_new,[])
magahi_token = [x for x in magahi_token if x != '[CLS]' and x!='[SEP]']

bhojpuri_token = sum(bhojpuri_token_new,[])
bhojpuri_token = [x for x in bhojpuri_token if x != '[CLS]' and x!='[SEP]']

hindi_true_label = sum(hindi_true_label,[])
hindi_predicted_label = sum(hindi_predicted_label,[])

angika_true_label = sum(angika_true_label,[])
angika_predicted_label = sum(angika_predicted_label,[])

magahi_true_label = sum(magahi_true_label,[])
magahi_predicted_label = sum(magahi_predicted_label,[])

bhojpuri_true_label = sum(bhojpuri_true_label,[])
bhojpuri_predicted_label = sum(bhojpuri_predicted_label,[])

# %%
def look_back(token,true_label,predicted_label):
    swapped_predicted = []
    for i in range(len(token)):
        #print(token[i])
        if '##' in token[i]:
            if '##' in token[i-1]:
                if '##' in token[i-2]:
                    swapped_predicted.append(predicted_label[i-3])
                else:
                    swapped_predicted.append(predicted_label[i-2])
            else:
                swapped_predicted.append(predicted_label[i-1])
        else:
            swapped_predicted.append(predicted_label[i])
    return swapped_predicted


# %%
swapped_predicted_look_back_hindi = look_back(hindi_token,hindi_true_label,hindi_predicted_label)


swapped_predicted_look_back_angika = look_back(angika_token,angika_true_label,angika_predicted_label)
swapped_predicted_look_back_magahi = look_back(magahi_token,magahi_true_label,magahi_predicted_label)
swapped_predicted_look_back_bhojpuri = look_back(bhojpuri_token,bhojpuri_true_label,bhojpuri_predicted_label)


# %%
from sklearn.metrics import classification_report
print("Hindi Look Back Prediction")
print(classification_report(hindi_true_label,swapped_predicted_look_back_hindi))
print("Angika Look Back Prediction")
print(classification_report(angika_true_label,swapped_predicted_look_back_angika))
print("*"*10)
print("Magahi Look Back Prediction")
print(classification_report(magahi_true_label,swapped_predicted_look_back_magahi))

print("Bhojpuri Look Back Prediction")
print(classification_report(bhojpuri_true_label,swapped_predicted_look_back_bhojpuri))

