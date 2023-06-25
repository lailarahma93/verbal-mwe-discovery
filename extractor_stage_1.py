import pandas as pd
import spacy
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
nlp = spacy.load("/Users/laila/Downloads/Models/Spacy2.1.x/NER/id_ud-tag-dep-ner-1.0.0/id_ud-tag-dep-ner/id_ud-tag-dep-ner-1.0.0")

with open('Corpus.txt') as f:
    text = list(f)
    text = text[0].split(" . ")
    normalized_text = [x.lower() for x in text]

def extract_mwe(data):
    mwe_candidates = []
    pos_excl = ['NUM', "PUNCT"]
    for sentence in data:
        doc = nlp(sentence)
        for token in doc:
            if token.pos_ == 'VERB' and (len(doc) - 1) > token.i and doc[token.i + 1].pos_ not in pos_excl:
                candidates = (token.sent, token.text, doc[token.i + 1].text)
                mwe_candidates.append(candidates)
    return mwe_candidates


def main():
    candidates = extract_mwe(normalized_text)
    print(candidates[0][1])
    print("raw candidates:", len(candidates))
    df = pd.DataFrame(candidates, columns=['Sentence','Verb','Modifier'])
    
    #Lemmatizing the heads
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    df['Stem'] = df['Verb'].apply(lambda x: stemmer.stem(x))
    df['Affix'] = df.apply(lambda x: x['Verb'].replace(x['Stem'], ' '), axis=1)
 
    #Removing [confixed head]-modifier and head-suffix-modifier
    df_filter1 = df[df["Affix"].str.contains(" kan| i| an| nya| lah") == False]
    df_filter1 = df_filter1[~df_filter1['Verb'].isin(df_filter1['Affix'])]
    
    #Lemmatizing the modifiers
    df_filter1['Stem_mod'] = df_filter1['Modifier'].apply(lambda x: stemmer.stem(x))
    df_filter1['Affix_mod'] = df_filter1.apply(lambda x: x['Modifier'].replace(x['Stem_mod'], ' '), axis=1)

    #Removing head-[confixed modifier], head-prefix-modifier, and head-modifier-suffix
    df_filter2 = df_filter1[df_filter1['Modifier'].isin(df_filter1['Stem_mod'])]

    #save the result as csv file
    df_filter2.to_csv('vmwe_candidates.csv')

    
main()
