from tqdm import tqdm
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cpu = torch.device("cpu")

word_vec = torch.load("GLOVE_300D_VEC.pt").to(device)

f = open("glove.6B.300d.txt", "r")
lines = f.read().split("\n")

words = []

for line in tqdm(lines):
    words.append(line.split(" ")[0])

def closest_word_to_pt(point, ret_list):
    i_word_vec = point.repeat(word_vec.size(0), 1)
    # distance = torch.sum((i_word_vec - word_vec)**2, dim=1)
    # distance = torch.norm(i_word_vec - word_vec, p=2, dim=1)
    distance = -torch.sum(i_word_vec * word_vec, dim=1)/torch.sqrt(torch.sum((i_word_vec)**2, dim=1)*torch.sum((word_vec)**2, dim=1))
    sorted_v, indices = torch.sort(distance)
    return indices[ret_list]

w1 = words.index("asia")
w2 = words.index("india")
w4 = words.index("germany")
point = word_vec[w1] - word_vec[w2] + word_vec[w4]
ret_list = range(5)
closest = closest_word_to_pt(point, ret_list)
for i in closest:
    print(words[i.item()])

# w_index = int((torch.rand((1))*word_vec.size(0)).item())
# w_index = words.index("frog")
# print(w_index)
# i_word_vec = word_vec[w_index:w_index+1].repeat(word_vec.size(0), 1)
# distance = torch.sum((i_word_vec - word_vec)**2, dim=1)
# # distance = torch.sum(i_word_vec * word_vec, dim=1)/torch.sqrt(torch.sum((i_word_vec)**2, dim=1)*torch.sum((word_vec)**2, dim=1))
# sorted_v, indices = torch.sort(distance)
# print(sorted_v[0:10], indices[0:10])
# for w in range(10):
#     print(words[indices[w].item()])

# w_index_frogs = words.index("frogs")
# print("FROG", word_vec[w_index])
# print("FROGS", word_vec[w_index_frogs])
# print("SNAKE", word_vec[9517])
# print("APE", word_vec[26266])
# print("FROGS DISTANCE", distance[w_index_frogs])