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

DATA_SIZE = 9900
VALID_DATA_SIZE = 100
BATCH_SIZE = 100
NUM_EPOCHS = 100
NUM_BATCHES = int(DATA_SIZE / BATCH_SIZE)
VALID_NUM_BATCHES = int(VALID_DATA_SIZE / BATCH_SIZE)

WORD_VEC_DIMS = 300
HIDDEN_SIZE = 16
NUM_WORDS_INPUT = 10
WORDS_GEN_OUT = 100

lr = 0.001
b1 = 0.5
b2 = 0.999

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cpu = torch.device("cpu") #can't print from GPU

folder = ""

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.lstm = torch.nn.LSTM(WORD_VEC_DIMS, HIDDEN_SIZE, batch_first=True)
        
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(HIDDEN_SIZE*2, WORD_VEC_DIMS)
        )

    def forward(self, input):
        output, (h_n, c_n) = self.lstm(input)
        return self.linear(torch.cat((h_n.permute(1, 0, 2).view(-1, HIDDEN_SIZE), c_n.permute(1, 0, 2).view(-1, HIDDEN_SIZE)), dim=1))

def embed_words(to_embed, glove_vec, glove_words):
    length = len(to_embed)
    ret = torch.zeros((length, WORD_VEC_DIMS))
    for (w, i) in zip(to_embed, range(len(to_embed))):
        ret[i] = glove_vec[glove_words.index(w)]
    return ret

def vec_to_word(vec, glove_vec, glove_words):
    copies = vec.view(1, WORD_VEC_DIMS).repeat((400000, 1))
    dist = torch.mean((copies - glove_vec)**2, dim=1)
    closest_index = torch.argmin(dist)
    return glove_words[closest_index]

def train_model(train_data, train_labels, glove_vec, glove_words): #2d list of words (reviews), 1d tensor of labels, 2d tensor of glove word vectors, 1d list of glove words    
    model = Model()

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
            reviews_list = []
            #max_len = 0
            for b in range(0, BATCH_SIZE):
                review = embed_words(train_data[b+(batch*BATCH_SIZE)], glove_vec, glove_words).to(device)
                review_len = review.size(0)
                start_index = ((review_len - NUM_WORDS_INPUT - 1)*torch.rand((1))).long()
                reviews_list.append(review[start_index:start_index + NUM_WORDS_INPUT + 1])
            #    review_len = review.size(0)
            #    if (review_len > max_len):
            #        max_len = review_len

            #for i in range(len(reviews_list)):
            #    s = reviews_list[i].size(0)
            #    reviews_list[i] = torch.cat((torch.zeros(max_len-s, WORD_VEC_DIMS).to(device), reviews_list[i])).to(device)
            
            reviews_stack = torch.stack(reviews_list).to(device)
            batch_input = reviews_stack[:, :NUM_WORDS_INPUT, :]
            batch_labels = reviews_stack[:, NUM_WORDS_INPUT:, :]

            #batch_labels = train_labels[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE].to(device)

            output = model(batch_input)

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

                reviews_list = []
                #max_len = 0
                for b in range(0, BATCH_SIZE):
                    review = embed_words(train_data[b+(batch*BATCH_SIZE)+DATA_SIZE], glove_vec, glove_words).to(device)
                    review_len = review.size(0)
                    start_index = ((review_len - NUM_WORDS_INPUT - 1)*torch.rand((1))).long()
                    reviews_list.append(review[start_index:start_index + NUM_WORDS_INPUT + 1])
                #    review_len = review.size(0)
                #    if (review_len > max_len):
                #        max_len = review_len

                #for i in range(len(reviews_list)):
                #    s = reviews_list[i].size(0)
                #    reviews_list[i] = torch.cat((torch.zeros(max_len-s, WORD_VEC_DIMS).to(device), reviews_list[i])).to(device)

                reviews_stack = torch.stack(reviews_list).to(device)
                batch_input = reviews_stack[:, :NUM_WORDS_INPUT, :]
                batch_labels = reviews_stack[:, NUM_WORDS_INPUT:, :]

                #batch_labels = train_labels[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE].to(device)

                output = model(batch_input)

                #perc_correct += torch.mean((torch.sign(output).view(-1) == batch_labels).float())

                loss = torch.nn.functional.mse_loss(output.view(-1), batch_labels.reshape(-1))      

                valid_loss += loss.to(cpu).item() / float(VALID_NUM_BATCHES)
                if (math.isnan(epoch_loss)):
                    print("NaN!")

                '''
                if batch == 0:
                    f_p = open(folder + "/epoch"+str(epoch+1) + "/positives.txt", "a")
                    f_n = open(folder + "/epoch"+str(epoch+1) + "/negatives.txt", "a")
                    pred = torch.sign(output).view(-1)
                    for b in range(0, BATCH_SIZE):
                        review = train_data[b+DATA_SIZE]
                        if (pred[b] == 1):
                            for w in review:
                                f_p.write(w+" ")
                            f_p.write("\n")
                        else:
                            for w in review:
                                f_n.write(w+" ")
                            f_n.write("\n")
                    f_p.close()
                    f_n.close()
                '''
        f_sample = open(folder + "/epoch"+str(epoch+1) + "/sample.txt", "a")
        
        working_input = train_data[DATA_SIZE][0:NUM_WORDS_INPUT]

        for word in working_input:
            f_sample.write(word+" ")

        for w in range(WORDS_GEN_OUT):
            vec = model(embed_words(working_input, glove_vec, glove_words).to(device).view(1, -1, WORD_VEC_DIMS)).view(-1)
            new_word = vec_to_word(vec.to(cpu), glove_vec, glove_words)
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