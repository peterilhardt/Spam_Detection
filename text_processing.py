## Functions for processing text data, including cleaning, displaying Word Clouds, and 
## converting to a document-term matrix (Bag of Words)

import re
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd


def clean_text(text):
    """
    Takes a string of article text and cleans it by:

    1. Removing numbers and words with numbers in them
    2. Removing all punctuation
    3. Making all letters lowercase
    4. Removing all non-English language characters (e.g. letters with 
       accents, Chinese symbols, etc.)
    5. Removing one- and two-letter words
    6. Removing newline characters
    7. Stripping whitespace

    Parameters
    ----------
    text: string
        Article text

    Returns
    -------
    string
        Article text cleaned
    """
    
    # removes numbers and words with numbers
    remove_numbers = lambda s: re.sub(r'\w*\d+\w*', ' ', s)  

    # removes !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
    remove_punctuation = lambda s: re.sub(r'[%s]' % \
                            re.escape(string.punctuation), ' ', s)  

    # makes all letters lowercase
    lowercase = lambda s: s.lower()  

    # removes non-english language characters
    ascii_only = lambda s: re.sub(r'[^\x00-\x7F]+', ' ', s)    

    # removes 1 and 2 letter words
    remove_short_words = lambda s: re.sub(r'\W*\b\w{1,2}\b', '', s)

    clean_txt = lowercase(remove_numbers(remove_punctuation(ascii_only(text))))
    clean_txt = remove_short_words(clean_txt)
    
    # return cleaned text with newlines and leading/trailing spaces removed
    return clean_txt.replace('\n', ' ').strip()


def display_wordcloud(data, title = None, file = None, bg_color = 'black', 
                      stopwords = None, max_words = 200, max_font_size = 40, 
                      scale = 3, random_state = None, prefer_horizontal = 0.9):
    """
    Generates a WordCloud from text data using the wordcloud module. It will 
    display 'max_words' words from the input text as a WordCloud image, which 
    randomly displays the most frequently occurring words in a horizontal or 
    vertical arrangement, making more frequent words larger in size. It is a 
    useful visualization to quickly show the most common words in text data. 
    
    Parameters
    ----------
    data: single string OR list of strings OR other object with multiple strings (e.g. array)
        The text data to generate a WordCloud from
    title: string, default: None
        Optional title for the image
    file: string, default: None
        Filename for optionally saving the WordCloud as image output
    bg_color: string, default: 'black'
        Background color for the WordCloud (typically 'white' or 'black')
    stopwords: list or set of strings, default: None
        Words to avoid including in the WordCloud
    max_words: int, default: 200
        Maximum number of words to include in the WordCloud
    max_font_size: int, default: 40
        Maximum font size to make a word in the WordCloud
    scale: int, default: 3
        Scaling factor between computation and drawing
    random_state: int, default: None
    prefer_horizontal: float in range(0, 1), default: 0.9
        Fraction of words to display horizontally instead of vertically
        
    Returns
    -------
    matplotlib image
        WordCloud of the input text data 
    """

    wordcloud = WordCloud(
        background_color = bg_color,
        stopwords = stopwords,
        max_words = max_words,
        max_font_size = max_font_size, 
        scale = scale,
        random_state = random_state,
        prefer_horizontal = prefer_horizontal
    ).generate(str(data))

    fig = plt.figure(1, figsize = (12, 12))
    plt.axis('off')

    if title: 
        fig.suptitle(title, fontsize = 20, fontweight = 'bold')
        fig.subplots_adjust(top = 1.38)

    plt.imshow(wordcloud)

    if file:
        plt.savefig(file, dpi = 100, bbox_inches = 'tight')
    plt.show()
    plt.close()


def doc_term_matrix(text, vectorizer = 'CV', stop_words = 'english'):
    """
    Generates a document-term matrix for a series of documents using either
    a Count Vectorizer or TF-IDF Vectorizer. A document-term matrix has the
    documents as rows and all words that appear in a document as columns.

    Parameters
    ----------
    text: Series
        The documents to be converted to a document-term matrix
    vectorizer: string: 'CV' or 'TFIDF', default: 'CV'
        Whether CountVectorizer or TfidfVectorizer should be used, respectively
    stop_words: string, default: 'english'
        Optional stopwords to exclude from the document-term matrix

    Returns
    -------
    DataFrame
        The document-term matrix
    """

    if vectorizer == 'CV':
        vec = CountVectorizer(stop_words = stop_words)
    elif vectorizer == 'TFIDF':
        vec = TfidfVectorizer(stop_words = stop_words)

    fit = vec.fit_transform(text)
    df = pd.DataFrame(fit.toarray(), columns = vec.get_feature_names())
    return df


