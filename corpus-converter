import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import spacy


corpus_file = '/Users/laila/Downloads/Indonesian_Manually_Tagged_Corpus.tsv'

#read the corpus
df = pd.read_csv(corpus_file, sep='\\t', engine='python', header=None)
df.columns =['Token', 'PoS']

#convert the corpus into list of sentences
wordlist = df["Token"].tolist()
print(wordlist[:10])

#save it as txt file
with open("/Users/laila/Downloads/Corpus.txt", "w") as output:
    output.write(' '.join(map(str, wordlist)))
