# Manipualtion des données
import pandas as pd
import sklearn
# Représentation graphique
# Netooyage du texte
import re
from bs4 import BeautifulSoup
# NLP
import nltk
import spacy
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import joblib

global nlp
nlp = spacy.load("en_core_web_sm")

def preprocessing(texte, rejoin=True):
        """
        Normalise et nettoie le texte de la liste de mots fournie.
        :param liste_mots: la liste de mots à normaliser et nettoyer
        :return: le texte nettoyé et normalisé
        """
        # Convertir le texte en minuscules
        texte = texte.lower()

        #Conservation des mots c# et c++
        texte =texte.replace('c#','csharpxxx').replace('c #','csharpxxx').replace('c ++','cplusxxx').replace('c++','cplusxxx')

        # BeautifulSoup pour supprimer les balises HTML et les entités
        texte = BeautifulSoup(texte, "lxml").get_text()

        # Expression régulière pour supprimer les URLs, les caractères non-alphanumériques
        # et les nombres qui apparaissent au début ou à la fin d'un mot
        texte_nettoye = re.sub(r"[^a-zA-Z\s]+|(http\S+)|(www\.\S+)|\d+", ' ', texte)

        #----Tokenisation = réduction des phrases en mots élémentaires
        texte_tokenise = word_tokenize(texte_nettoye)



        stop_w = list(set(stopwords.words('english')))

        #----Filtre des stop words
        texte_filtered_w = [w for w in texte_tokenise if not w in stop_w]
        texte_filtered_w2 = [w for w in texte_filtered_w if len(w) > 2]

        for i in range(len(texte_filtered_w2)):
                if "sharpxxx" or "cplusxxx" in words_list[i]:
                    texte_filtered_w2[i] = texte_filtered_w2[i].replace("sharpxxx", "#")
                    texte_filtered_w2[i] = texte_filtered_w2[i].replace("plusxxx", "++")



        #----Lemmatizer (base d'un mot)
        # Mapping POS tags to WordNet POS tags
        wordnet_map = {'N': wordnet.NOUN, 'V': wordnet.VERB, 'R': wordnet.ADV, 'J': wordnet.ADJ}

        lemmatizer = WordNetLemmatizer()
        pos_tags = nltk.pos_tag(texte_filtered_w2)
        final_texte = [lemmatizer.lemmatize(word, pos=wordnet_map.get(pos[0].upper(), wordnet.NOUN)) for word, pos in pos_tags]


        if rejoin :
            # On renvoie une chaîne de caractère (en combinant tous les mots texte)
            return ' '.join(final_texte)
        return final_texte

#This function will remove the unecessary part of speech and keep only necessary part of speech given by pos_list
def final_preprocessing(texte):
    texte = nlp(texte)
    doc = texte
    pos_list = ["NOUN","PROPN"]
    text_list = []
    for token in doc:
            if(token.pos_ in pos_list):
                 text_list.append(token.text)
    join_text = " ".join(text_list)
    join_text = join_text.lower().replace("c #", "c#").replace("c ++", "c++").replace("#","c# ").replace("cc#","c#")
    return join_text


def predict_tags (texte) :
    tfidf_vectorizer = joblib.load('tfidf_vectorizer_100_words.joblib')
    multilabel_binarizer = joblib.load("multilabel_100_words.joblib")
    model = joblib.load("logit_tdidf_100_words.joblib")
    texte = [texte]
    texte_tfidf = tfidf_vectorizer.transform(texte)

    feature_names_tfidf = tfidf_vectorizer.get_feature_names()
    X_tfidf_input = pd.DataFrame(texte_tfidf.toarray(), columns=feature_names_tfidf)

    prediction =model.predict(X_tfidf_input)
    tags_predict = multilabel_binarizer.inverse_transform(prediction)
    tags= list({tag for tag_list in tags_predict for tag in tag_list if (len(tag_list) != 0)})

    if len(tags) == 0:
        tags = "Pas de tags prédits pour ce problème"
    return tags

