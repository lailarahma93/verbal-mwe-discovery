import pandas as pd
import gensim
from collections import Counter
import spacy
nlp = spacy.load("/Users/laila/Downloads/Models/Spacy2.1.x/NER/id_ud-tag-dep-ner-1.0.0/id_ud-tag-dep-ner/id_ud-tag-dep-ner-1.0.0")
df = pd.read_csv('vmwe_candidates.csv')

# Loading fastText Indonesian word vectors using gensim
dsm_file = '/Users/laila/Downloads/cc.id.300.vec.gz'
ds_model = gensim.models.KeyedVectors.load_word2vec_format(dsm_file, limit = 1000000)
print('Done!')

#the size of the model
vocab_size = len(ds_model.key_to_index)
print('The model covers', vocab_size, 'words')

#Calculate cosine similarity between head and modifier
mod_list = df['Modifier'].tolist()
mod_pos = []
for word in mod_list:
    doc = nlp(word)
    for token in doc:
        mod_pos.append(token.pos_)

df['POS_mod'] = mod_pos

cos_sim=[]
for verb,mod in zip(df['Verb'],df['Modifier']):
    try:
        cos_sim_value = ds_model.similarity(verb, mod)
        cos_sim_value = cos_sim_value.item()
        cos_sim_value = round(cos_sim_value, 2)
        cos_sim.append(cos_sim_value)
    except KeyError:
        cos_sim.append('N/A') #1655 N/A
print(len(cos_sim))
df['Cos_sim'] = cos_sim

#Calculate cosine similarity between MWE and head
df["NoSpaceMWE"] = df["Verb"] + df["Modifier"]
x=[]
for mwe,verb in zip(df['NoSpaceMWE'],df['Verb']):
    try:
        x_value = ds_model.similarity(mwe, verb)
        x_value = x_value.item()
        x_value = round(x_value, 2)
        x.append(x_value)
    except KeyError:
        x.append('N/A') 
print(len(x))
df['Cos_sim_MWExVerb'] = x

#Calculate cosine similarity between MWE and modifier
y=[]
for mwe,mod in zip(df['NoSpaceMWE'],df['Modifier']):
    try:
        y_value = ds_model.similarity(mwe, mod)
        y_value = y_value.item()
        y_value = round(y_value, 2)
        y.append(y_value)
    except KeyError:
        y.append('N/A') 
print(len(y))
df['Cos_sim_MWExMod'] = y

#Mutual Information
with open('Corpus.txt') as f:
    text = list(f)
    normalized = [x.lower() for x in text]
    words = normalized[0].split()
    sentences = normalized[0].split(" . ")

#Counting word frequency in the corpus
freq = {}
frequency = Counter(words)
for word in frequency:
    freq[word] = frequency[word]

#Convert the MWE candidates into a list
df["MWE"] = df["Verb"] + ' ' + df["Modifier"]
mwe_list = df['MWE'].tolist()
print(mwe_list[:10])

#Generate bigram from the corpus and count the frequency
bigrams = [b for l in sentences for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
# print(bigrams[:10])
bigram_list = []
for i in bigrams:
    string = " ".join(i)
    bigram_list.append(string)
print(bigram_list[:10])
bigram_dict = {}
bigram_freq = Counter(bigram_list)
for word in bigram_list:
    bigram_dict[word] = bigram_freq[word]

#Counting Mutual Information
n = 256622 #corpus size in token
fAB = []
for i in mwe_list:
    try:
        number = (bigram_dict[i]/n) + .00001
        fAB.append(number)
    except KeyError:
        number = (1/n) + .00001
        fAB.append(number)

#Convert the verb and modifier columns into lists
verb_list = df['Verb'].tolist()
mod_list = df['Modifier'].tolist()

#Counting freqs of verbs and mods
fA = []
for i in verb_list:
    fA_list = (frequency[i]/n) + .00001
    fA.append(fA_list)

fB = []
for i in mod_list:
    fB_list = (frequency[i]/n) + .00001
    fB.append(fB_list)

to_count = [list(x) for x in zip(fAB,fA,fB)]

MI = []
for i in to_count:
    score = i[0]/(i[1]*i[2])
    score = round(score)
    MI.append(score)

df['MI'] = MI

#Remove candidates with cos_sim <0.4 and PMI <100
df = df.dropna()
df = df[df['Cos_sim'] >= 0.4]
df = df[df['MI'] >= 150]

#check numbers of unique MWE
mwe = df['MWE'].tolist()
print(len(set(mwe)))

#save the final result as csv file
df.to_csv('vmwe_result.csv')
