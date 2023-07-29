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
"""nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')"""
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


texte = """How to allow Python.app to firewall on Mac OS X? <p>When I run a python application on Mac, it shows many dialogs about want "Python.app" to accept incoming network connections.</p>

<p>Even I Allow it many times, it shows again and again.</p>

<p>How to allow it one time and not show any more?</p>

<p><a href="https://i.stack.imgur.com/AOWy3.png" rel="noreferrer"><img src="https://i.stack.imgur.com/AOWy3.png" alt="enter image description here"></a></p>

<hr>

<h1>Edit</h1>

<p>I found this question:
<a href="https://stackoverflow.com/questions/19688841/add-python-to-os-x-firewall-options">Add Python to OS X Firewall Options?</a></p>

<p>I followed the accepted answer to do but finally when I run <code>codesign -s "My Signing Identity" -f $(which python)</code>, it said:</p>

<pre><code>/usr/bin/python: replacing existing signature
error: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/codesign_allocate: can't create output file: /usr/bin/python.cstemp (Operation not permitted)
/usr/bin/python: the codesign_allocate helper tool cannot be found or used
</code></pre>

<p>How to do next?</p>  C# """

texte1 = """ excel cell date apache apache file date switch cell case cell type value value cell break value type return value cell type row c row c c string date date date date fix problem',
 'validation error path require document mongodb path email path require path email user schema var new schema number type string type string email type string admin date date string save user object var user email password field null email null admin save user function err err console log user err user user return validation error value',
 'plan application wonder thanks',
 'execute laravel command version laravel follow command git bash php user table table user trigger follow error input file number thing find site work suggestion work',
 'position property custom desire font size color code self self self uicolor self font self uicolor alpha self self navigation bar bar button result text center origin label effect' """


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

"""
texte =preprocessing(texte, rejoin=True)
print(texte)
print('-----------------------------------------------------------------------------------------------------------------------------------------------------------')
text2=final_preprocessing(texte)
print(text2)
print([text2])
print('-------------------------- Final-----------------------------------------------------------')
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
#print (predict_tags (text2))

multilabel_binarizer = joblib.load("multilabel.joblib")
model = joblib.load("logit_tdidf.joblib")
#X_test = joblib.load("X_tfidf_test.joblib")
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')



#1----Phase de processing du texte et de création de la matrice bag of words pour le texe entré
texte_tfidf = tfidf_vectorizer.transform([text2])  # Transformation du texte avec le modèle
feature_names_tfidf = tfidf_vectorizer.get_feature_names()  # Récupération des features du tfidf
X_tfidf_input = pd.DataFrame(texte_tfidf.toarray(), columns=feature_names_tfidf)  # Trnsformation du texte entré en dataframe

print(X_tfidf_input)  # Affichage du texte

#2--- Une fois la matrice bag of words du texte créé, on donne le texte au modèle pour faire des prédictions
prediction = model.predict(X_tfidf_input)  #Prédiction par le modèle, le modèle renvoie des entiers au lieu des vrais tags
tags_predict = multilabel_binarizer.inverse_transform(prediction) # récupéaration des vrais
tags = list({tag for tag_list in tags_predict for tag in tag_list if (len(tag_list) != 0)})
print('---------------------------------------------------Tags---------------------------------------------------------')
print(tags_predict)





#feature_names_tfidf = tfidf_vectorizer.get_feature_names()
#X_tfidf_input = pd.DataFrame(text3_tfidf.todense(), columns=feature_names_tfidf)

#print(X_tfidf_input.shape)


"""

import numpy as np

print("Version de NumPy :", np.__version__)