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

# %% [markdown]
# 
# %%
# %%
# %%
# %%
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


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
tokenizer = AutoTokenizer.from_pretrained("google/rembert",strip_accents=False)


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
model = AutoModelForTokenClassification.from_pretrained('google/rembert',num_labels=18)


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
hindi_true_label = hindi_eval_result['eval_true_labels']
hindi_predicted_label = hindi_eval_result['eval_predictions']

angika_true_label = angika_eval_result['eval_true_labels']
angika_predicted_label = angika_eval_result['eval_predictions']

magahi_true_label = magahi_eval_result['eval_true_labels']
magahi_predicted_label = magahi_eval_result['eval_predictions']

bhojpuri_true_label = bhojpuri_eval_result['eval_true_labels']
bhojpuri_predicted_label = bhojpuri_eval_result['eval_predictions']
# %%
print("Length of hindi true equal predicted labels: ",len(hindi_predicted_label),len(hindi_predicted_label)==len(hindi_true_label))
print("Length of angika true equal predicted labels: ",len(angika_predicted_label),len(angika_predicted_label)==len(angika_true_label))
print("Length of magahi true equal predicted labels: ",len(magahi_predicted_label),len(magahi_predicted_label)==len(magahi_true_label))
print("Length of bhojpuri true equal predicted labels: ",len(bhojpuri_predicted_label),len(bhojpuri_predicted_label)==len(bhojpuri_true_label))


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
hi_to_len = []
for i in range(len(hindi_token_new)):
    hi_to_len.append(len(hindi_token_new[i]))
    
hi_tr_len = []
for i in range(len(hindi_token_new)):
    hi_tr_len.append(len(hindi_true_label[i]))

ang_to_len = []
for i in range(len(angika_token_new)):
    ang_to_len.append(len(angika_token_new[i]))
    
ang_tr_len = []
for i in range(len(angika_token_new)):
    ang_tr_len.append(len(angika_true_label[i]))

mag_to_len = []
for i in range(len(magahi_token_new)):
    mag_to_len.append(len(magahi_token_new[i]))
    
mag_tr_len = []
for i in range(len(magahi_token_new)):
    mag_tr_len.append(len(magahi_true_label[i]))
    
bho_to_len = []
for i in range(len(bhojpuri_token_new)):
    mag_to_len.append(len(bhojpuri_token_new[i]))
    
bho_tr_len = []
for i in range(len(bhojpuri_token_new)):
    mag_tr_len.append(len(bhojpuri_true_label[i]))

# %%
count_1 = 0
for item in zip(ang_to_len,ang_tr_len):
    count_1 += 1
    if item[0]-item[1]==2:
        pass
    else:
        print(item,count_1)


# %%
hindi_token = sum(hindi_token_new,[])
hindi_token = [x for x in hindi_token if x != '[CLS]' and x!='[SEP]']

angika_token = sum(angika_token_new,[])
angika_token = [x for x in angika_token if x != '[CLS]' and x!='[SEP]']

magahi_token = sum(magahi_token_new,[])
magahi_token = [x for x in magahi_token if x != '[CLS]' and x!='[SEP]']

bhojpuri_token = sum(bhojpuri_token_new,[])
bhojpuri_token = [x for x in bhojpuri_token if x != '[CLS]' and x!='[SEP]']
# %%
hindi_true_label = sum(hindi_true_label,[])
hindi_predicted_label = sum(hindi_predicted_label,[])

angika_true_label = sum(angika_true_label,[])
angika_predicted_label = sum(angika_predicted_label,[])

magahi_true_label = sum(magahi_true_label,[])
magahi_predicted_label = sum(magahi_predicted_label,[])

bhojpuri_true_label = sum(bhojpuri_true_label,[])
bhojpuri_predicted_label = sum(bhojpuri_predicted_label,[])
# %%
print(len(hindi_token))
print(len(hindi_true_label))
print(len(hindi_predicted_label))

print(len(angika_token))
print(len(angika_true_label))
print(len(angika_predicted_label))

print(len(magahi_token))
print(len(magahi_true_label))
print(len(magahi_predicted_label))

print(len(bhojpuri_token))
print(len(bhojpuri_true_label))
# %%
def compute_metrics(eval_preds):
    pred_logits,labels = eval_preds
    pred_logits_max = np.max(pred_logits,axis=2)
    return {
        "pred_logits":pred_logits,
        "labels":labels,
        "pred_logits_max":pred_logits_max
        }


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
hindi_eval_result_score = evaluate_(tokenized_datasets_hi_test)
angika_eval_result_score = evaluate_(tokenized_datasets_ang_test)
magahi_eval_result_score = evaluate_(tokenized_datasets_mag_test)
bhojpuri_eval_result_score = evaluate_(tokenized_datasets_bho_test)


# %%
def max_val_and_label(df,lang_eval_result_score):
    label_ = []
    maximum_ = []
    for i in range(len(df)):
        for index,item in enumerate(lang_eval_result_score['eval_labels'][i]):
            if item !=-100 :
                #print(angika_eval_result['eval_pred_logits'][i][index])
                lab = np.argmax(lang_eval_result_score['eval_pred_logits'][i][index])
                label_.append(lab)
                #print("label: ",lab)
                pred_mat = (lang_eval_result_score['eval_pred_logits'][i][index])
                #print("Maximum: ",maxi)
                maximum_.append(list(pred_mat))
                #max_ = angika_eval_result['eval_pred_logits'][i][index]/np.max(angika_eval_result['eval_pred_logits'][i][index])
                #print("Norm: ",max_)
    return label_,maximum_


# %%
hindi_pred_label_,hindi_pred_label_max_score = max_val_and_label(hindi_df,hindi_eval_result_score)

angika_pred_label_,angika_pred_label_max_score = max_val_and_label(angika_df,angika_eval_result_score)
magahi_pred_label_,magahi_pred_label_max_score = max_val_and_label(magahi_df,magahi_eval_result_score)
bhojpuri_pred_label_,bhojpuri_pred_label_max_score = max_val_and_label(bhojpuri_df,bhojpuri_eval_result_score)


# %%
print(len(hindi_pred_label_)==len(hindi_pred_label_max_score))
print(len(angika_pred_label_)==len(angika_pred_label_max_score))
print(len(magahi_pred_label_)==len(magahi_pred_label_max_score))
print("Bhojpuri: ",len(bhojpuri_pred_label_)==len(bhojpuri_pred_label_max_score))


# %%
print(len(hindi_pred_label_max_score)==len(hindi_token))
print(len(angika_pred_label_max_score)==len(angika_token))
print(len(magahi_pred_label_max_score)==len(magahi_token))


# %%
def label_softmax_score(eval_label,eval_pred_logits):
    label_ = []
    maximum_ = []
    size = eval_label.shape[0]
    for i in range(size):
        for index,item in enumerate(eval_label[i]):
            if item != -100:
                lab = np.argmax(eval_pred_logits[i][index])
                label_.append(lab)
                pred_mat = eval_pred_logits[i][index]
                maximum_.append(list(pred_mat))
    return maximum_,label_


# %%
hi_label_score_list,hi_arg_max_label = label_softmax_score(hindi_eval_result_score['eval_labels'],hindi_eval_result_score['eval_pred_logits'])

ang_label_score_list,ang_arg_max_label = label_softmax_score(angika_eval_result_score['eval_labels'],angika_eval_result_score['eval_pred_logits'])

mag_label_score_list,mag_arg_max_label = label_softmax_score(magahi_eval_result_score['eval_labels'],magahi_eval_result_score['eval_pred_logits'])

bho_label_score_list,bho_arg_max_label = label_softmax_score(bhojpuri_eval_result_score['eval_labels'],bhojpuri_eval_result_score['eval_pred_logits'])

# %%
hindi_data = {'hindi_token':hindi_token,'hindi_true_label':hindi_true_label,'hindi_predicted_label':hindi_predicted_label,
'hi_label_score_list':hi_label_score_list,'hi_arg_max_label':hi_arg_max_label}
df_hindi_max_score = pd.DataFrame(hindi_data)

angika_data = {'angika_token':angika_token,'angika_true_label':angika_true_label,'angika_predicted_label':angika_predicted_label,
'ang_label_score_list':ang_label_score_list,'ang_arg_max_label':ang_arg_max_label}
df_angika_max_score = pd.DataFrame(angika_data)


magahi_data = {'magahi_token':magahi_token,'magahi_true_label':magahi_true_label,'magahi_predicted_label':magahi_predicted_label,
'magahi_label_score_list':mag_label_score_list,'magahi_arg_max_label':mag_arg_max_label}
df_magahi_max_score = pd.DataFrame(magahi_data)

bhojpuri_data = {'bhojpuri_token':bhojpuri_token,'bhojpuri_true_label':bhojpuri_true_label,'bhojpuri_predicted_label':bhojpuri_predicted_label,
'bhojpuri_label_score_list':bho_label_score_list,'bhojpuri_arg_max_label':bho_arg_max_label}
df_bhojpuri_max_score = pd.DataFrame(bhojpuri_data)
# %%
df_bhojpuri_max_score.head()


# %%
hi_token = list(df_hindi_max_score['hindi_token'])
hi_truth = list(df_hindi_max_score['hindi_true_label'])
hi_predicted = list(df_hindi_max_score['hindi_predicted_label'])
hi_label = list(df_hindi_max_score['hi_arg_max_label'])
hi_max_list = list(df_hindi_max_score['hi_label_score_list'])

ang_token = list(df_angika_max_score['angika_token'])
ang_truth = list(df_angika_max_score['angika_true_label'])
ang_predicted = list(df_angika_max_score['angika_predicted_label'])
ang_label = list(df_angika_max_score['ang_arg_max_label'])
ang_max_list = list(df_angika_max_score['ang_label_score_list'])

mag_token = list(df_magahi_max_score['magahi_token'])
mag_truth = list(df_magahi_max_score['magahi_true_label'])
mag_predicted = list(df_magahi_max_score['magahi_predicted_label'])
mag_label = list(df_magahi_max_score['magahi_arg_max_label'])
mag_max_list = list(df_magahi_max_score['magahi_label_score_list'])

bho_token = list(df_bhojpuri_max_score['bhojpuri_token'])
bho_truth = list(df_bhojpuri_max_score['bhojpuri_true_label'])
bho_predicted = list(df_bhojpuri_max_score['bhojpuri_predicted_label'])
bho_label = list(df_bhojpuri_max_score['bhojpuri_arg_max_label'])
bho_max_list = list(df_bhojpuri_max_score['bhojpuri_label_score_list'])
# %%
print(len(hi_token)==len(hi_max_list))
print(len(hi_token)==len(hi_label))

print(len(ang_token)==len(ang_max_list))
print(len(ang_token)==len(ang_label))

print(len(mag_token)==len(mag_max_list))
print(len(mag_token)==len(mag_label))


# %%
def underscore_position(token):
    my_list = token
    underscore_pos = []
    for i in range(len(my_list) - 5):
        if my_list[i]=='▁':
            my_list[i+1] = my_list[i]+my_list[i + 1]
            underscore_pos.append(i)
    return my_list,underscore_pos


# %%
updated_hi_token,hi_position = underscore_position(hi_token)
updated_ang_token,ang_position = underscore_position(ang_token)
updated_mag_token,mag_position = underscore_position(mag_token)
updated_bho_token,bho_position = underscore_position(bho_token)

# %%
print(len(updated_hi_token)==len(hi_max_list))
print(len(hi_token)==len(hi_label))
print(len(updated_hi_token)==len(hi_label))


# %%
print(len(updated_hi_token)==len(hi_truth))
print(len(updated_hi_token)==len(hi_max_list))
print(len(updated_hi_token)==len(hi_label))


# %%
def final_allignment_pre_processing(token,true_label,pred_label,underscore_pos,label,max_list):
    my_list = token
    true_list = true_label
    pred_list = pred_label
    label_ = label
    max_list_ = max_list
    positions_to_remove = underscore_pos
    for position in positions_to_remove:
        if 0 <= position < len(my_list):
            removed_element = my_list.pop(position)
            true_list.pop(position)
            pred_list.pop(position)
            label_.pop(position)
            max_list_.pop(position)
    return my_list,true_list,pred_list,label_,max_list_


# %%
final_hindi_token,final_hindi_true_label,final_hindi_pred_label,final_hindi_label,final_hindi_max_list = final_allignment_pre_processing(updated_hi_token,hi_truth,hi_predicted,hi_position,hi_label,hi_max_list)


# %%
print(len(final_hindi_token))
print(len(final_hindi_true_label))
print(len(final_hindi_pred_label))
print(len(final_hindi_label))
print(len(final_hindi_max_list))


# %%
final_angika_token,final_angika_true_label,final_angika_pred_label,final_angika_label,final_angika_max_list = final_allignment_pre_processing(updated_ang_token,ang_truth,ang_predicted,ang_position,ang_label,ang_max_list)


# %%
print(len(final_angika_token))
print(len(final_angika_true_label))
print(len(final_angika_pred_label))
print(len(final_angika_label))
print(len(final_angika_max_list))


# %%
final_magahi_token,final_magahi_true_label,final_magahi_pred_label,final_magahi_label,final_magahi_max_list = final_allignment_pre_processing(updated_mag_token,mag_truth,mag_predicted,mag_position,mag_label,mag_max_list)


# %%
print(len(final_magahi_token))
print(len(final_magahi_true_label))
print(len(final_magahi_pred_label))
print(len(final_magahi_label))
print(len(final_magahi_max_list))

final_bhojpuri_token,final_bhojpuri_true_label,final_bhojpuri_pred_label,final_bhojpuri_label,final_bhojpuri_max_list = final_allignment_pre_processing(updated_bho_token,bho_truth,bho_predicted,bho_position,bho_label,bho_max_list)




# %%
print(len(final_bhojpuri_token))
print(len(final_bhojpuri_true_label))
print(len(final_bhojpuri_pred_label))
print(len(final_bhojpuri_label))
print(len(final_bhojpuri_max_list))


# %%
label_list_original = ['NOUN','PUNCT','ADP','NUM','SYM','SCONJ','ADJ','PART','DET','CCONJ','PROPN','PRON','X','_','ADV','INTJ','VERB','AUX']


# %%
def max_1(index,max_list_1,max_list_2):
    return np.argmax(np.add(max_list_1,max_list_2))
def max_2(index,max_list_1,max_list_2,max_list_3):
    return np.argmax(np.add(np.add(max_list_1,max_list_2),max_list_3))
def max_3(index,max_list_1,max_list_2,max_list_3,max_list_4):
    return np.argmax(np.add(np.add(max_list_1,max_list_2),np.add(max_list_3,max_list_4)))
def max_4(index,max_list_1,max_list_2,max_list_3,max_list_4,max_list_5):
    return np.argmax(np.add(np.add(np.add(max_list_1,max_list_2),np.add(max_list_3,max_list_4)),max_list_5))


# %%
def look_back(token,truth,predicted,label,max_list):
    swapped_predicted = predicted
    #print(len(swapped_predicted))
    for i in range(len(token)-2):
        if '▁' in token[i] and '▁' not in token[i+1] and '▁' in token[i+2]:
            index_1 = max_1(i,max_list[i],max_list[i+1])
            #print(i,index_1,max_list[i],max_list[i+1])
            #print("Token: ",token[i],token[i+1],token[i+2])
            swapped_predicted[i] = label_list_original[index_1]
            swapped_predicted[i+1] = label_list_original[index_1]
            #print("True: ",truth)
            #print("Swapped: ",swapped_predicted)
        elif '▁' in token[i] and '▁' not in token[i+1] and '▁' not in token[i+2] and '▁' in token[i+3]:
            index_2 = max_2(i,max_list[i],max_list[i+1],max_list[i+2])
            #print(i,index_2,max_list[i],max_list[i+1],max_list[i+2])
            #print("Token: ",token[i],token[i+1],token[i+2],token[i+3])
            swapped_predicted[i] = label_list_original[index_2]
            swapped_predicted[i+1] = label_list_original[index_2]
            swapped_predicted[i+2] = label_list_original[index_2]
            #print("Swapped: ",swapped_predicted)
        elif '▁'  in token[i] and '▁' not in token[i+1] and '▁' not in token[i+2] and '▁' not in token[i+3] and '▁' in token[i+4]:
            index_3 = max_3(i,max_list[i],max_list[i+1],max_list[i+2],max_list[i+3])
            #print(i,index_3,max_list[i],max_list[i+1],max_list[i+2],max_list[i+3])
            #print("Token: ",token[i],token[i+1],token[i+2],token[i+3],token[i+4])
            swapped_predicted[i] = label_list_original[index_3]
            swapped_predicted[i+1] = label_list_original[index_3]
            swapped_predicted[i+2] = label_list_original[index_3]
            swapped_predicted[i+3] = label_list_original[index_3]
            #print("Swapped: ",swapped_predicted)
        elif '▁'  in token[i] and '▁' not in token[i+1] and '▁' not in token[i+2] and '▁' not in token[i+3] and '▁' not in token[i+4] and '▁' in token[i+5]:
            index_4 = max_4(i,max_list[i],max_list[i+1],max_list[i+2],max_list[i+3],max_list[i+4])
            swapped_predicted[i] = label_list_original[index_4]
            swapped_predicted[i+1] = label_list_original[index_4]
            swapped_predicted[i+2] = label_list_original[index_4]
            swapped_predicted[i+3] = label_list_original[index_4]
            swapped_predicted[i+4] = label_list_original[index_4]
        else:
            pass
    return swapped_predicted


# %%
hi_swapped_pred = look_back(final_hindi_token,final_hindi_true_label,final_hindi_pred_label,final_hindi_label,final_hindi_max_list)

ang_swapped_pred = look_back(final_angika_token,final_angika_true_label,final_angika_pred_label,final_angika_label,final_angika_max_list)

mag_swapped_pred = look_back(final_magahi_token,final_magahi_true_label,final_magahi_pred_label,final_magahi_label,final_magahi_max_list)

bho_swapped_pred = look_back(final_bhojpuri_token,final_bhojpuri_true_label,final_bhojpuri_pred_label,final_bhojpuri_label,final_bhojpuri_max_list)

# %%
from sklearn.metrics import classification_report
print("Hindi Lookback score Result")
print(classification_report(hi_truth,hi_swapped_pred))


# %%
print("Angika Lookback score Result")
print(classification_report(ang_truth,ang_swapped_pred))


# %%
print("Magahi Lookback score Result")
print(classification_report(mag_truth,mag_swapped_pred))


# %%
print("Bhojpuri Lookback score Result")
print(classification_report(bho_truth,bho_swapped_pred))



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



