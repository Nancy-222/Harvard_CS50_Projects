import nltk
import sys
from nltk.tokenize import word_tokenize
import string
import os
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    dict_map={}
    for filename in os.listdir(directory):
            with open(os.path.join(directory, filename),encoding="utf-8") as f:
                read=f.read()
                dict_map[filename]=read
    return dict_map


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    list_words=[]
    document_lowercase=document.lower()
    result=word_tokenize(document_lowercase)
    punctuation=string.punctuation
    stopwords=nltk.corpus.stopwords.words("english")
    for word in result:
        if word not in punctuation and word not in stopwords:
            list_words.append(word)
    return list_words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    number_of_documents=len(documents)
    idfs = dict()
    one_word=set()
    for words in documents.values():
        for word in words:
            one_word.add(word)
    for word in one_word:
        doc_count=0
        for word_doc in documents.values():
            if word in word_doc:
                doc_count+=1
        idf = math.log(number_of_documents /doc_count)
        idfs[word] = idf
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tfidfs=dict()
    list=[]
    for filename in files:
        tfidfs[filename]=[]
        idf_value=0
        for word in query:
            tf=files[filename].count(word)
            idf=idfs.get(word,0)
            tfidfs[filename].append((word, tf * idf))
            idf_value+=tf * idf
        list.append((idf_value,filename))
    list.sort(key=lambda tfidf: tfidf[0], reverse=True)
    top_files = []  
    for idf_value, filename in list[:n]:
        top_files.append(filename)
    return top_files 


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    list=[]
    for sentence,words in sentences.items():
        matching_word_measure=0
        query_term_density=0
        for word in query:
            if word in words:
                matching_word_measure+=idfs.get(word,0)
                query_term_density+=1
        query_term_density=query_term_density/len(words)
        list.append((sentence,matching_word_measure,query_term_density))
    list.sort(key=lambda tfidf: (tfidf[1],tfidf[2]), reverse=True)
    top_sentences = []  
    for i in range(n):
        tuple=list[i]
        sentence=tuple[0]
        top_sentences.append(sentence)
    return top_sentences 


if __name__ == "__main__":
    main()
