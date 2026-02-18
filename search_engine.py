#!/usr/bin/env python3
"""
CS 5180 - Assignment 1 - Question 8
Implements 
-tokenization 
-surface-level normalization
-stopping 
-stemming
-build vocabulary: unigrams and then bigrams
-binary vectors
-score via dot product and rank documents
"""

import re
from typing import Dict, List, Tuple

# Stopping
STOPWORDS = {
  #pronouns
  "i", "me", "my", "mine"
  "you", "him", "his"
  "she", "her", "hers",
  "it", "its"
  "they", "them", "their", "theirs", 
  "we", "us", "our", "ours",
  #conjunctions
  "and", "or", "but", "nor", "so", "yet",
  #articles 
  "a", "an", "the",

def tokenize (text: str) -> List[str]:
  """
  Tokenization + surface-level normalization
  """
  return re.findall(r"[a-z]+", text.lower())

def stem(word: str) -> str:
    """
    Simple rule-based stemmer
    """
    #plural/3rd-person singular cases
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"          # e.g., "studies" -> "study"
    if len(word) > 3 and word.endswith("es"):
        return word[:-2]                # e.g., "watches" -> "watch"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]                # e.g., "dogs" -> "dog", "loves" -> "love"
    return word

def preprocess(text: str) -> List[str]:
    """
    Full preprocessing 
    """
    tokens = tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    tokens = [stem(t) for t in tokens]
    return tokens

def make_bigrams(tokens: List[str]) -> List[str]:
    return [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]

def build_vocabulary(query_tokens: List[str], doc_tokens_list: List[List[str]]) -> List[str]:
    #Unigrams
    unigram_set = set(query_tokens)
    for dtoks in doc_tokens_list:
        unigram_set.update(dtoks)
    #Bigrams
    query_bigrams = make_bigrams(query_tokens)
    bigram_set = set(query_bigrams)
    for dtoks in doc_tokens_list:
        bigram_set.update(make_bigrams(dtoks))
    unigrams = sorted(unigram_set)
    bigrams = sorted(bigram_set)
    #Ordering
    return unigrams + bigrams

def vectorize(tokens: List[str], vocab: List[str]) -> List[int]:
    """
    Binary vector
    """
    unigram_terms = set(tokens)
    bigram_terms = set(make_bigrams(tokens))
    present = unigram_terms.union(bigram_terms)
    return [1 if term in present else 0 for term in vocab]

def dot(u: List[int], v: List[int]) -> int:
    return sum(a * b for a, b in zip(u, v))

def main() -> None:
    q = "I love dogs"
    docs = {
        "d1": "I love a dog and a cat.",
        "d2": "She loves her cat and dogs.",
        "d3": "They love their cat.",
    }
    q_tokens = preprocess(q)
    doc_tokens = {name: preprocess(text) for name, text in docs.items()}

    vocab = build_vocabulary(q_tokens, list(doc_tokens.values()))

    q_vec = vectorize(q_tokens, vocab)
    d_vecs = {name: vectorize(toks, vocab) for name, toks in doc_tokens.items()}
    scores = {name: dot(q_vec, vec) for name, vec in d_vecs.items()}

    #Ranking
    ranking = sorted(scores.items(), key=lambda x: (-x[1], x[0]))

    #Print results
    print("Query tokens:", q_tokens)
    for name in sorted(docs.keys()):
        print(f"{name} tokens:", doc_tokens[name])
    print("\nVocabulary (unigrams then bigrams):")
    for i, term in enumerate(vocab):
        print(f"{i:02d}: {term}")
    print("\nBinary vectors:")
    print("q :", q_vec)
    for name in sorted(docs.keys()):
        print(f"{name}:", d_vecs[name])
    print("\nScores:")
    for name in sorted(scores.keys()):
        print(f"Score({name}) = {scores[name]}")
    print("\nRanking:")
    print(" > ".join([f"{name} ({score})" for name, score in ranking]))

if __name__ == "__main__":
    main()
