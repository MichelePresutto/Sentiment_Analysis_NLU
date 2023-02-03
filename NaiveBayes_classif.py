import sys
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from nltk.corpus import movie_reviews, subjectivity

def lol2str(lol):
    return " ".join([w for sent in lol for w in sent])

# from list to string
def list2str(l):
    return ' '.join(w for w in l)


def remove_objective_sents(classifier, vectorizer, doc): 
    filt_sent = []
    doc = [list2str(p) for p in doc]
    vectors = vectorizer.transform(doc)
    subj_prediction = classifier.predict(vectors)
    # apppend only subjective sentence (subj_prediction = 1)
    for d, est in zip(doc, subj_prediction):
        if est==1:
            filt_sent.append(d)
    filt_doc = list2str(filt_sent)
    return filt_doc

# Subjectivity/Objectivity Identification
def subjectivity_classifier():

    # initialize classifier and vectorizer for subjectivity classification
    vectorizer = CountVectorizer()
    classifier = MultinomialNB()

    # data is divided into objective and subjective sentences
    obj = subjectivity.sents(categories='obj')
    subj = subjectivity.sents(categories='subj')

    # preprocess the dataset
    corpus = [list2str(d) for d in obj] + [list2str(d) for d in subj]
    vectors = vectorizer.fit_transform(corpus)
    labels = [0] * len(obj) + [1] * len(subj)

    # evaluation
    accuracy = cross_validate(classifier, vectors, labels, cv=StratifiedKFold(n_splits=10) , scoring=['accuracy'])
    avg_accuracy = sum(accuracy['test_accuracy'])/len(accuracy['test_accuracy'])


    # train again the classifier to return
    classifier.fit(vectors, labels)

    return avg_accuracy, classifier, vectorizer


# Polarity Classification: positive or negative.
def polarity_classifier(subj_classifier, subj_vectorizer, filter):
   
    # initialize classifier and vectorizer for Polairty classification
    vectorizer = CountVectorizer()
    classifier = MultinomialNB()

    # data is divided into negative and positive sentences
    neg = movie_reviews.paras(categories='neg')
    pos = movie_reviews.paras(categories='pos')

    # remove objective sentences, better performances
    if filter:
        corpus = []
        doc = neg + pos
        for d in doc:
            corpus.append(remove_objective_sents(subj_classifier, subj_vectorizer, d))
    else:
        corpus = [lol2str(d) for d in neg] + [lol2str(d) for d in pos]

    # preprocess the dataset
    vectors = vectorizer.fit_transform(corpus)
    labels = [0] * len(neg) + [1] * len(pos)

    # evaluation 
    accuracy = cross_validate(classifier, vectors, labels, cv=StratifiedKFold(n_splits=10) , scoring=['accuracy'])
    avg_accuracy = sum(accuracy['test_accuracy'])/len(accuracy['test_accuracy'])

    return avg_accuracy



    
print("Baseline:")

# subjectivity classifier using simple Naive Bayes Classification
subj_accuracy, subj_classifier, subj_vectorizer = subjectivity_classifier()
print(f"Accuracy on Naive Bayes subjectivity classification: {round(subj_accuracy*100, 4)}")

# polarity classifier using simple Naive Bayes Classification and removing objective sentences (filter set to True by default)
polarity_accuracy = polarity_classifier(subj_classifier, subj_vectorizer, True)
print(f"Accuracy on Naive Bayes polarity classification: {round(polarity_accuracy*100, 4)}")

