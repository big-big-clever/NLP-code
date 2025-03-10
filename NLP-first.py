import nltk
from nltk.corpus import gutenberg
from collections import Counter
import math

# 下载Gutenberg Corpus
nltk.download('gutenberg')

# 加载Gutenberg Corpus中的文本
text = gutenberg.raw('austen-emma.txt')  # 以《Emma》为例

# 字符级熵计算
def calculate_char_entropy(text):
    char_counts = Counter(text)
    total_chars = sum(char_counts.values())
    entropy = 0.0
    for count in char_counts.values():
        probability = count / total_chars
        entropy -= probability * math.log2(probability)
    return entropy

# 单词级熵计算
def calculate_word_entropy(text):
    words = nltk.word_tokenize(text)
    word_counts = Counter(words)
    total_words = sum(word_counts.values())
    entropy = 0.0
    for count in word_counts.values():
        probability = count / total_words
        entropy -= probability * math.log2(probability)
    return entropy

# 计算字符级熵
char_entropy = calculate_char_entropy(text)
print(f"字符级熵: {char_entropy:.4f} bits per character")

# 计算单词级熵
word_entropy = calculate_word_entropy(text)
print(f"单词级熵: {word_entropy:.4f} bits per word")
