from transformers import GPT2TokenizerFast
import Stemmer
import numpy as np
import os
import pickle
import math

d_model = 512
n=6
h=8



def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=1, keepdims=True)

def load_model():
    with open(os.path.join(os.getcwd(), "model.pkl"), 'rb') as file:
        model_data = pickle.load(file)
    return model_data

def token_vectors(model, tokens):
    return np.array([model['dictonary_vectors'][t] for t in tokens])

def PE(pos, i, d_model):
    if i % 2 == 0:
        return math.sin(pos / (10000 ** (i / d_model)))
    else:
        return math.cos(pos / (10000 ** ((i - 1) / d_model)))


def positional_encoding(tokens, start_pos, d_model):
    seq_len = tokens.shape[0]
    pe = np.zeros((seq_len, d_model))

    for pos in range(seq_len):
        actual_pos = start_pos + pos
        for i in range(d_model):
            pe[pos, i] = PE(actual_pos, i, d_model)

    return tokens + pe, seq_len + 1

def input_encoding(model, tokens, d_model, n, h):
    tokens_vektors = tokens
    for i in range(n):
        temp_matrice = multiheaded_attention(model['multihead_matrices_input'][i],tokens_vektors, d_model, h)
        temp_matrice = temp_matrice @ model['W_O_input'][i]
        tokens_vektors = add_norm(tokens_vektors, temp_matrice)
        temp_matrice = feedfarward(model['feedforward_matrices_input'][i],tokens_vektors)
        tokens_vektors = add_norm(tokens_vektors, temp_matrice)
    return tokens_vektors

def add_norm(tokens_vektors, temp_matric,  eps=1e-6):
    # Residual connection
    added = tokens_vektors + temp_matric

    # Layer normalization
    mean = np.mean(added, axis=1, keepdims=True)
    var = np.mean((added - mean) ** 2, axis=1, keepdims=True)

    return (added - mean) / np.sqrt(var + eps)



def multiheaded_attention(attention_model, tokens, d_model, h):
    wqkv = tokens @ attention_model
    d_k = d_model // h

    # Step 2: split Q, K, V
    q, k, v = np.split(wqkv, 3, axis=1)

    # split heads
    def split_heads(x):
        return x.reshape(x.shape[0], h, d_k)

    q = split_heads(q)
    k = split_heads(k)
    v = split_heads(v)

    outputs = []

    # attention per head
    for i in range(h):
        qi = q[:, i, :]
        ki = k[:, i, :]
        vi = v[:, i, :]
        outputs.append((softmax((qi@ ki.T)/math.sqrt(d_k)))@vi)
    return np.concatenate(outputs, axis=1)

def feedfarward(feedfarward_model, tokens_vektors):
    W1 = feedfarward_model["W1"]
    W2 = feedfarward_model["W2"]

    # First linear
    hidden = tokens_vektors @ W1

    # Activation (ReLU)
    hidden = np.maximum(0, hidden)

    # Second linear
    output = hidden @ W2

    return output

def output_decifiring(output_tokens, model, input_matrice, d_model, n, h, next_start_pos):
    
    return n
    

stemmer = Stemmer.Stemmer("english")
tokenizer = GPT2TokenizerFast.from_pretrained("Xenova/gpt-4", local_files_only=True)

text = input("input: ")

stemmed = " ".join(stemmer.stemWords(text.split()))

tokens = tokenizer.encode("User:" + stemmed + " Model:")

print("stemmed:", stemmed)
print("tokens:", tokens)


model = load_model()

tokens = token_vectors(model, tokens)

print(tokens.shape)

tokens, next_start_pos = positional_encoding(tokens, 0, d_model)

input_matrice = input_encoding(model, tokens, d_model, n, h)

output_tokens = []

output_tokens.append(model['dictonary_vectors'][output_decifiring(output_tokens, model, input_matrice, d_model, n, h, next_start_pos)])




