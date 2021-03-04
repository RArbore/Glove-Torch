import nltk
import torch
from tqdm import tqdm

to_convert = 50000

f_w = open("glove.6B.words.txt", "r")
glove_words = f_w.read().split("\n")
f_w.close()

f_i = open("IMDB Dataset.csv", "r")
lines = f_i.read().split("\n")
f_i.close()

f_o = open("imdb.filtered.txt", "a")

labels = torch.ones((to_convert))
i = 0

for line in tqdm(lines[:to_convert]):
    if not line == "":
        review = line[0:len(line)-9].lower()
        rating = line[len(line)-8:]
        if (rating == "positive"):
            rating = 1
        else:
            rating = -1
        words = nltk.word_tokenize(review)
        new_review = ""
        for word in words:
            if (word in glove_words):
                new_review = new_review + word + " "
        new_review = new_review[:-1]
        labels[i] = rating
        f_o.write(new_review+"\n")
        i += 1

f_o.close()
torch.save(labels, "IMDB_LABELS.pt")
print(i, labels.size())