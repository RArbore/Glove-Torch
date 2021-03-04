import torch
from tqdm import tqdm

dims = 300

f = open("glove.6B."+str(dims)+"d.txt", "r")
lines = f.read().split("\n")

words = []
word_vectors = torch.zeros((400000, dims))

for line in tqdm(lines):
    if not line == "":
        components = line.split(" ")
        words.append(components[0])
        for i in range(dims):
            word_vectors[len(words)-1, i] = float(components[i+1])

print(word_vectors[0:10, 0:10])
torch.save(word_vectors, "GLOVE_"+str(dims)+"D_VEC.pt")