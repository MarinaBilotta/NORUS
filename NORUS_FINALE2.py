import os
import math
import glob
import PyPDF2
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt

def compare_articles(input_article_text, existing_articles, keyword_list):
    stemmer = PorterStemmer()

    input_tokens = tokenize(input_article_text)
    keywords = set([stemmer.stem(word.lower()) for word in keyword_list])
    keyword_count = sum(1 for word in input_tokens if stemmer.stem(word.lower()) in keywords)
    text_length = len(input_tokens)  # Lunghezza del testo dell'articolo di input

    # Calcola la frequenza delle parole chiave nell'articolo di input
    keyword_frequency = sum(input_tokens.count(stemmer.stem(word.lower())) for word in keyword_list)

    # Trova le parole chiave effettive nell'articolo di input e conteggia le occorrenze
    found_keywords = {}
    for word in input_tokens:
        stemmed_word = stemmer.stem(word.lower())
        if stemmed_word in keywords:
            if stemmed_word not in found_keywords:
                found_keywords[stemmed_word] = 1
            else:
                found_keywords[stemmed_word] += 1

    similarities = [calculate_similarity_cosine(input_article_text, article, stemmer, keywords) for article in existing_articles]

    original = all(similarity < 0.25 for similarity in similarities)

    return found_keywords, text_length, keyword_frequency, original

def calculate_similarity_cosine(paper1, paper2, stemmer, keywords):
    array1 = vectorize(paper1, stemmer, keywords)
    array2 = vectorize(paper2, stemmer, keywords)

    dot_product = 0
    norma_array1 = 0
    norma_array2 = 0

    common = common_words(array1, array2)
    if len(common) == 0:
        return 0

    for word in common:
        dot_product += array1[word] * array2[word]
        norma_array1 += array1[word] ** 2
        norma_array2 += array2[word] ** 2

    if norma_array1 == 0 or norma_array2 == 0:
        return 0

    similarity = dot_product / (math.sqrt(norma_array1) * math.sqrt(norma_array2))
    return similarity

def common_words(vector1, vector2):
    return set(vector1.keys()) & set(vector2.keys())

def vectorize(paper, stemmer, keywords):
    array = {}
    words = tokenize(paper)

    for word in words:
        word_stem = stemmer.stem(word.lower())
        if word_stem in keywords:
            if word_stem not in array:
                array[word_stem] = 1
            else:
                array[word_stem] += 1

    return array

def tokenize(paper):
    return paper.split()

def load_existing_articles(directory_path):
    existing_articles = []
    for file_path in glob.glob(os.path.join(directory_path, "*.pdf")):
        print("Lettura file:", file_path)
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            existing_articles.append(text)
    return existing_articles

def extract_text_from_pdf(pdf_file_path):
    with open(pdf_file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Impostazioni percorso cartelle
input_article_directory_path = r"C:\Users\marin\Desktop\INPUT_ARTICLE"
existing_articles_directory_path = r"C:\Users\marin\Desktop\PAGINA 1 EXCEL DOI"

# Variabili di input
input_article_filename = "38.pdf"
input_article_path = os.path.join(input_article_directory_path, input_article_filename)
keyword_list = ["Glyphosate", "cell", "herbicide", "molecular", "docking"]

# Caricamento degli articoli esistenti
existing_articles = load_existing_articles(existing_articles_directory_path)

# Estrazione del testo dall'articolo di input PDF
input_article_text = extract_text_from_pdf(input_article_path)

# Calcolo dei risultati
found_keywords, text_length, keyword_frequency, original = compare_articles(input_article_text, existing_articles, keyword_list)

# Numero totale di parole chiave trovate
total_keywords = sum(found_keywords.values())

# Visualizzazione delle parole chiave e dei loro conteggi
print("Parole chiave trovate nell'articolo:")
for keyword, count in found_keywords.items():
    print(f"{keyword}: {count}")

print("Numero totale di parole chiave:", total_keywords)

# Creazione del primo grafico (lilla) - Originalità e Numero di Parole Chiave
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.bar(["Originality"], [int(original)], color='purple')
plt.title('Originality')
plt.ylabel('Value')

plt.subplot(1, 2, 2)
plt.bar(["Keyword Count"], [total_keywords], color='purple')
plt.title('Keyword Count')
plt.ylabel('Value')

plt.tight_layout()

# Messaggio di output sotto forma di frase testuale
if original:
    originality_message = "L'articolo è considerato originale."
else:
    originality_message = "L'articolo non è considerato originale."

print(originality_message)

# Creazione del secondo grafico (arancione) - Frequenza di Occorrenza delle Keywords e Lunghezza del Testo
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.bar(["Keyword Frequency"], [keyword_frequency], color='orange')
plt.title('Keyword Frequency')
plt.ylabel('Value')

plt.subplot(1, 2, 2)
plt.bar(["Text Length"], [text_length], color='orange')
plt.title('Text Length')
plt.ylabel('Value')

plt.tight_layout()

# Messaggio di output sotto forma di frase testuale
print("Lunghezza del testo dell'articolo:", text_length)
print("Frequenza di occorrenza delle parole chiave nell'articolo:", keyword_frequency)

plt.show()