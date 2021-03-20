from torchvision import transforms
import random
import torch
import time
import math
import sys
import os

manualSeed = int(torch.rand(1).item() * 1000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

trial_name = "gentrial"

DATA_SIZE = 49000
VALID_DATA_SIZE = 1000
BATCH_SIZE = 100
NUM_EPOCHS = 10000
NUM_BATCHES = int(DATA_SIZE / BATCH_SIZE)
VALID_NUM_BATCHES = int(VALID_DATA_SIZE / BATCH_SIZE)

WORD_VEC_DIMS = 300
HIDDEN_SIZE = 64
WORDS_GEN_OUT = 100
NUM_LSTM_LAYERS = 3

lr = 0.0025
b1 = 0.5
b2 = 0.999

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cpu = torch.device("cpu") #can't print from GPU

folder = ""

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.lstm = torch.nn.LSTM(WORD_VEC_DIMS, HIDDEN_SIZE, num_layers = NUM_LSTM_LAYERS, batch_first=True)
        
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            torch.nn.Tanh(),
            torch.nn.Linear(HIDDEN_SIZE, WORD_VEC_DIMS),
        )

    def forward(self, input, h_0 = None, c_0 = None):
        
        if h_0 is None or c_0 is None:
            h_0 = torch.zeros(NUM_LSTM_LAYERS, input.size(0), HIDDEN_SIZE).to(device)
            c_0 = torch.zeros(NUM_LSTM_LAYERS, input.size(0), HIDDEN_SIZE).to(device)
        else:
            print("hello")

        output, (h_n, c_n) = self.lstm(input, (h_0, c_0))
        return self.linear(output.permute(1, 0, 2).view(-1, HIDDEN_SIZE)).view(input.size(0), -1, WORD_VEC_DIMS), (h_n, c_n)
        # return self.linear(torch.cat((h_n.permute(1, 0, 2).reshape(-1, HIDDEN_SIZE*NUM_LSTM_LAYERS), c_n.permute(1, 0, 2).reshape(-1, HIDDEN_SIZE*NUM_LSTM_LAYERS)), dim=1))

def embed_words(to_embed, glove_vec, glove_words):
    length = len(to_embed)
    ret = torch.zeros((length, WORD_VEC_DIMS))
    for (w, i) in zip(to_embed, range(len(to_embed))):
        ret[i] = glove_vec[glove_words.index(w)]
    return ret

def vec_to_word(vec, glove_vec, glove_words):
    copies = vec.view(1, WORD_VEC_DIMS).repeat((400000, 1))
    dist = torch.exp(torch.mean((copies - glove_vec)**2, dim=1))
    probs = torch.distributions.categorical.Categorical(1 - dist/torch.sum(dist))
    return glove_words[probs.sample()]

def print_vec_stats(vec, glove_vec, glove_words):
    copies = vec.view(1, WORD_VEC_DIMS).repeat((400000, 1))
    dist = torch.mean((copies - glove_vec)**2, dim=1)
    indices = []
    while len(indices) < 10:
        index = torch.argmin(dist)
        dist[index] += 1000
        indices.append(index)
    print("######################\n")
    for index in indices:
        print(glove_words[index], dist[index] - 1000)
    print("")

def train_model(train_data, train_labels, glove_vec, glove_words): #2d list of words (reviews), 1d tensor of labels, 2d tensor of glove word vectors, 1d list of glove words    
    model = Model()
    size = 0
    for parameter in model.parameters():
        size += parameter.view(-1).size(0)
    print(size)

    current_milli_time = lambda: int(round(time.time() * 1000))

    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2))

    before_time = current_milli_time()

    print("Beginning training")
    print("")

    f = open(folder + "/during_training_performance.txt", "a")

    for epoch in range(0, NUM_EPOCHS):
        os.mkdir(folder + "/epoch"+str(epoch+1))

        epoch_loss = 0
        valid_loss = 0
        epoch_before_time = current_milli_time()

        perc_correct = torch.zeros((1)).to(device)
        valid_perc_correct = torch.zeros((1)).to(device)

        for batch in range(NUM_BATCHES):
            opt.zero_grad()
            model = model.train()
            '''
            reviews_list = []
            max_len = 0
            for b in range(0, BATCH_SIZE):
                review = embed_words(train_data[b+(batch*BATCH_SIZE)], glove_vec, glove_words).to(device)
                reviews_list.append(review)
            
                review_len = review.size(0)
                if (review_len > max_len):
                    max_len = review_len

            for i in range(len(reviews_list)):
                s = reviews_list[i].size(0)
                reviews_list[i] = torch.cat((torch.zeros(max_len-s, WORD_VEC_DIMS).to(device), reviews_list[i])).to(device)
            
            reviews_stack = torch.stack(reviews_list).to(device)
            
            torch.save(reviews_stack, "IMDB_DATA/b"+str(batch)+".pt")
            '''
            reviews_stack = torch.load("IMDB_DATA/b"+str(batch)+".pt").to(device)

            batch_input = reviews_stack[:, :-1, :]
            batch_labels = reviews_stack[:, 1:, :]

            #batch_labels = train_labels[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE].to(device)

            h_0 = torch.zeros(NUM_LSTM_LAYERS, BATCH_SIZE, HIDDEN_SIZE).to(device)
            c_0 = torch.zeros(NUM_LSTM_LAYERS, BATCH_SIZE, HIDDEN_SIZE).to(device)
 
            output = model(batch_input, (h_0, c_0))
            output = output[0]
            #print_vec_stats(output[0].to(cpu), glove_vec, glove_words)

            #perc_correct += torch.mean((torch.sign(output).view(-1) == batch_labels).float())

            loss = torch.nn.functional.mse_loss(output.view(-1), batch_labels.reshape(-1))
            loss.backward()
            opt.step()
            epoch_loss += loss.to(cpu).item() / float(NUM_BATCHES)
            if (math.isnan(epoch_loss)):
                print("NaN!")

        with torch.no_grad():
            for batch in range(VALID_NUM_BATCHES):
                model = model.eval()
                '''
                reviews_list = []
                max_len = 0
                for b in range(0, BATCH_SIZE):
                    review = embed_words(train_data[b+(batch*BATCH_SIZE)+DATA_SIZE], glove_vec, glove_words).to(device)
                    reviews_list.append(review)
                    review_len = review.size(0)
                    if (review_len > max_len):
                        max_len = review_len

                for i in range(len(reviews_list)):
                    s = reviews_list[i].size(0)
                    reviews_list[i] = torch.cat((torch.zeros(max_len-s, WORD_VEC_DIMS).to(device), reviews_list[i])).to(device)

                reviews_stack = torch.stack(reviews_list).to(device)
            
                torch.save(reviews_stack, "IMDB_DATA/v"+str(batch)+".pt")
                '''
                reviews_stack = torch.load("IMDB_DATA/v"+str(batch)+".pt")

                batch_input = reviews_stack[:, :-1, :]
                batch_labels = reviews_stack[:, 1:, :]

                #batch_labels = train_labels[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE].to(device)

                h_0 = torch.zeros(NUM_LSTM_LAYERS, BATCH_SIZE, HIDDEN_SIZE).to(device)
                c_0 = torch.zeros(NUM_LSTM_LAYERS, BATCH_SIZE, HIDDEN_SIZE).to(device)

                output = model(batch_input, (h_0, c_0))
                output = output[0]
                #perc_correct += torch.mean((torch.sign(output).view(-1) == batch_labels).float())

                loss = torch.nn.functional.mse_loss(output.view(-1), batch_labels.reshape(-1))      

                valid_loss += loss.to(cpu).item() / float(VALID_NUM_BATCHES)
                if (math.isnan(epoch_loss)):
                    print("NaN!")

        f_sample = open(folder + "/epoch"+str(epoch+1) + "/sample.txt", "a")
        
        working_input = train_data[0][0:1]
        h_n = torch.zeros(NUM_LSTM_LAYERS, 1, HIDDEN_SIZE).to(device)
        c_n = torch.zeros(NUM_LSTM_LAYERS, 1, HIDDEN_SIZE).to(device)

        for word in working_input:
            f_sample.write(word+" ")

        for w in range(WORDS_GEN_OUT):
            vec, (h_n, c_n) = model(embed_words(working_input, glove_vec, glove_words).to(device).view(1, -1, WORD_VEC_DIMS), (h_n, c_n))
            vec = vec[0].view(-1)
            #print(vec)
            new_word = vec_to_word(vec.to(cpu), glove_vec, glove_words)
            #print(new_word+"\n")
            working_input = working_input[1:]
            working_input.append(new_word)
            f_sample.write(new_word+" ")
        f_sample.close()

        perc_correct = (perc_correct / NUM_BATCHES).item() * 100
        valid_perc_correct = (valid_perc_correct / VALID_NUM_BATCHES).item() * 100

        epoch_after_time = current_milli_time()
        seconds = math.floor((epoch_after_time - epoch_before_time) / 1000)
        minutes = math.floor(seconds / 60)
        seconds = seconds % 60

        print("["+str(epoch + 1)+"]   Loss : "+str(epoch_loss)+"   Validation Loss : "+str(valid_loss)+"   Percent Correct : "+str(perc_correct)+"%   Validation Percent Correct : "+str(valid_perc_correct)+"%   Took "+str(minutes)+" minute(s) and "+str(seconds)+" second(s).")

        f.write(str(epoch + 1)+" "+str(epoch_loss)+" "+str(valid_loss)+"\n")

    after_time = current_milli_time() #time of entire process

    torch.save(model.to(cpu).state_dict(), folder + "/model.pt")
    print("")
    f.close()

    seconds = math.floor((after_time - before_time) / 1000)
    minutes = math.floor(seconds / 60)
    seconds = seconds % 60

    print(str(NUM_EPOCHS) + " epochs took " + str(minutes) + " minute(s) " + str(seconds) + " second(s).")

    return model

if __name__ == "__main__":
    print("Start!")
    current_milli_time = lambda: int(round(time.time() * 1000))
    before_time = current_milli_time()

    #sets up the trial folder
    if len(sys.argv) <= 1:
        files = os.listdir(".")
        m = [int(f[len(trial_name):]) for f in files if len(f) > len(trial_name) and f[0:len(trial_name)] == trial_name]
        if len(m) > 0:
            folder = trial_name + str(max(m) + 1)
        else:
            folder = trial_name + "1"
    else: #Otherwise, if we want to set it up manually
        folder = sys.argv[1]

    os.mkdir(folder)

    print("Created session folder " + folder)

    print("Loading data...")

    f_i = open("imdb.filtered.txt", "r")
    train_data = f_i.read().split("\n")
    for i in range(len(train_data)):
        train_data[i] = train_data[i].split(" ")
    f_i.close()
    train_labels = torch.load("IMDB_LABELS.pt")

    glove_vec = torch.load("GLOVE_300D_VEC.pt")
    f_w = open("glove.6B.words.txt", "r")
    glove_words = f_w.read().split("\n")
    f_w.close()

    after_time = current_milli_time()
    seconds = math.floor((after_time - before_time) / 1000)
    minutes = math.floor(seconds / 60)
    seconds = seconds % 60
    print("Data loading took " + str(minutes) + " minute(s) " + str(seconds) + " second(s).")

    model = train_model(train_data, train_labels, glove_vec, glove_words)
