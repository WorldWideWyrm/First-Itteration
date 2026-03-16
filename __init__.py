from transformers import GPT2TokenizerFast
import Stemmer
import numpy as np
import os
import pickle

stemmer = Stemmer.Stemmer("english")
tokenizer = GPT2TokenizerFast.from_pretrained("Xenova/gpt-4", local_files_only=True)

text = input("input: ")

stemmed = " ".join(stemmer.stemWords(text.split()))

tokens = tokenizer.encode("User:" + stemmed + " Model:")

print("stemmed:", stemmed)
print("tokens:", tokens)

