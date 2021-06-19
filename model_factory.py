################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################
import torch
from torch import LongTensor
from torch.nn import Embedding, LSTMCell, RNNCell
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# vocab stuff
from vocab import *


# Build and return the model here based on the configuration.
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    max_length = config_data['generation']['max_length']

    temperature = config_data['generation']['temperature']
    deterministic = config_data['generation']['deterministic']
    batch_size = config_data['dataset']['batch_size']

    # You may add more parameters if you want
    net = Network(2, hidden_size, embedding_size, model_type, vocab,
                  max_length, temperature, deterministic, batch_size)
    return net


class Network(nn.Module):
    def __init__(self, arch, hidden_size, embedding_size, model_type, vocab, max_length, temperature, deterministic, batch_size):
        super(Network, self).__init__()
        self.encoder = Encoder(embedding_size)
        self.arch = arch
        if arch == 1:
            self.decoder = Decoder1(model_type,
                                    embedding_size,
                                    hidden_size,
                                    vocab,
                                    max_length,
                                    temperature,
                                    deterministic,
                                    batch_size)
        elif arch == 2:
            self.decoder = Decoder2(embedding_size,
                                    hidden_size,
                                    vocab,
                                    max_length,
                                    temperature,
                                    deterministic,
                                    batch_size)

    def forward(self, images, captions):
    	features = self.encoder(images) # outputs (64 x 300) features
       	out = self.decoder(features, captions)

        return out

    def generate(self, images):
        # images = images.reshape(1, images.shape[0],
        #                         images.shape[1], images.shape[2])
        images = images.unsqueeze(0)
        features = self.encoder(images)


        out = self.decoder.generate(features)

        return out


class Encoder(nn.Module):

    """
    Encoder Architecture
    """

    def __init__(self, embedding_size):

        super(Encoder, self).__init__()
        # use resnet50 for the pretrained model
        model = models.resnet50(pretrained=True)

        # freeze model
        for param in model.parameters():
            param.requires_grad = False

        modules = list(model.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # add a trainable linear layer
        self.fc = nn.Linear(in_features=2048, out_features=embedding_size)
        torch.nn.init.xavier_uniform(self.fc.weight)

    def forward(self, images):
        """
        forward pass
        images -->

        """
        out = self.resnet(images)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out # output has shape (64x300) (images x embedding size)


class Decoder1(nn.Module):

    """
    Decoder Architecture
    """

    def __init__(self, model_type, embedding_size, hidden_size, vocab, max_length, temperature, deterministic, batch_size):
        """
        num_embeddings len(vocab) (int) --> Size of the dictionary of embeddings
        embedding_size (int) --> size of each embedding vector
        model_type (str) --> comes from config file, tells the decoder model that we are using
        hid_size (int) --> number of hidden cells that we are using in RNN
        """
        super(Decoder1, self).__init__()

        self.max_length = max_length
        self.vocab_len = len(vocab)
        self.batch_size = batch_size
        self.deterministic = deterministic
        self.temperature = temperature
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(self.vocab_len, embedding_size)
        self.soft = nn.Softmax()
        self.linear = nn.Linear(hidden_size, self.vocab_len)
        # for stochastic generating captions: store the output of the last layer
        self.past = None
        self.model_type = model_type
        if model_type == "LSTM":
            self.decoder = LSTMCell(input_size=embedding_size,
                                    hidden_size=hidden_size)
        elif model_type == "RNN":
            self.decoder = RNNCell(input_size=embedding_size,
                                hidden_size=hidden_size)

    def forward(self, features, captions):
        """
        Teacher enforced learning append the captions to the features so first chracter in every sequence
        is the images feature embedding
        """
      	embed = self.embedding(captions)  # [1, seq_len, 300]	

        embed = torch.cat(
       	    (features.unsqueeze(dim=1).float(), embed.float()), dim=1)	

        if self.model_type == 'LSTM':
            hx, cx = self.init_hidden(embed.shape[0])

        elif self.model_type == 'RNN':
            hx = self.init_hidden(embed.shape[0])

        output = torch.empty(embed.shape[0], embed.shape[1], self.hidden_size).cuda()

        for time in range(embed.shape[1]):
            time_step = embed[:, time, :]

            if self.model_type == 'LSTM':
                hx, cx = self.decoder(time_step, (hx, cx))

            elif self.model_type == 'RNN':
                hx = self.decoder(time_step, hx)


            output[:, time, :] = hx


        output = self.linear(output)

        # shave off last prediction in each seqeunce as this is the feature embedding
       	pred = output[:, :-1, :]	

        # needs to be in order og [batch, classes, seq_len]
        pred = pred.permute(0, 2, 1)

        return pred.cuda()

    def generate(self, features):
        gen = []
        count = 0

        if self.model_type == 'LSTM':
            hx, cx = self.init_hidden(1)

        elif self.model_type == 'RNN':
            hx = self.init_hidden(1)

        while True:
            if self.model_type == 'LSTM':
                hx, cx = self.decoder(features, (hx, cx))

            elif self.model_type == 'RNN':
                hx = self.decoder(features, hx)

            out = self.linear(hx)


            if self.deterministic:
                out = self.soft(out)
                word = torch.argmax(out, dim=1)
            else:
                out = out / self.temperature
                word = torch.Tensor(
                    list(WeightedRandomSampler(self.soft(out), 1))).long().cuda()


                word = word.squeeze(0)

           	gen.append(int(word))
            count += 1
            # reached max length or end token
            if count == self.max_length or int(word) == 2:
                break

            features = self.embedding(word)
            


        return torch.tensor(gen)


    def init_hidden(self, batches):
        if self.model_type == 'LSTM':

            return (torch.zeros(batches, self.hidden_size).cuda(),
                    torch.zeros(batches, self.hidden_size).cuda())

        elif self.model_type == 'RNN':
            return torch.zeros(batches, self.hidden_size).cuda()



class Decoder2(nn.Module):

    """
    Decoder Architecture
    """

    def __init__(self, embedding_size, hidden_size, vocab, max_length, temperature, deterministic, batch_size):
        """
        num_embeddings len(vocab) (int) --> Size of the dictionary of embeddings
        embedding_size (int) --> size of each embedding vector
        model_type (str) --> comes from config file, tells the decoder model that we are using
        hid_size (int) --> number of hidden cells that we are using in RNN

        vocab --> vocabulary built from vocab.py

        """
        super(Decoder2, self).__init__()


        self.max_length = max_length
        self.vocab_len = len(vocab)
        self.batch_size = batch_size
        self.deterministic = deterministic
        self.temperature = temperature
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.vocab = vocab


        self.embedding = nn.Embedding(self.vocab_len, embedding_size)
        self.soft = nn.Softmax()
        self.linear = nn.Linear(hidden_size, self.vocab_len)


        self.decoder = nn.LSTMCell(input_size= 2 * embedding_size,
                                hidden_size=hidden_size)



    def get_word(self, word):
        """
        gets the embedding of a word from the given vocabulary
        returns a [1 x embedding size] vector

        """

        word_index = self.vocab.word2idx[word]
        embed = self.embedding(torch.tensor(word_index).cuda())
        return embed.unsqueeze(0)




    def forward(self, features, captions):
        """
        Teacher enforced learning append
        	the captions to the features so that every charchater has image features
         features --> image features [64 x 300]
        (64 images, rgb, size 256x256)
         captions --> the original caption for this image
         (64 x n) caption of n words

        """

        # we have feature, as well as the word embeddings, concat these features
        ## feature shape is (64 x embedding size)
        ## captions shape is (64 x caption length)


        # get embedding and reshape features to fit embedding
       	embed = self.embedding(captions) # shape: (64 x caption length x embedding size)

        ### repeat feature vector of one photo (caption length) times for each caption, to fit embedding
        pad = self.get_word("<pad>")

        pad = pad.repeat(embed.shape[0], 1, 1) ## shape should be [pictures, 1, embed_size]

        embed = torch.cat((pad, embed), dim = 1) ##

        feat = features.unsqueeze(1).repeat(1, embed.shape[1], 1)
        concat = torch.cat((embed, feat), dim = 2) # shape: (64 x caption len x embedding size *2) (photo with word embedding)
        

        # initialize hidden states
       	hx, cx = None, None

        output = torch.empty(concat.shape[0], concat.shape[1], self.hidden_size).cuda()

        for timestep in range(concat.shape[1]):
            step = concat[:, timestep, :] # perform per caption, but 64 images at a time
            # print("step shape per caption:", step.shape)
            if (type(hx) == type(None)) and (type(cx) == type(None)):
                hx, cx = self.decoder(step)
            else:
                hx, cx = self.decoder(step,(hx, cx))


            output[:, timestep, :] = hx


        output = self.linear(output)

        # shave off last prediction in each seqeunce as this is the feature embedding
       	pred = output[:, :-1, :]

        # needs to be in order og [batch, classes, seq_len]
        pred = pred.permute(0, 2, 1)

        return pred.cuda()


    def generate(self, features):
        gen = []
        count = 0
        

        # append <pad> to features so the model will predict "<start>" as the first token, and so on...
        pad = self.get_word("<pad>")

        concat = torch.cat((pad, features), dim = 1)
        hx, cx = None, None


        while True:
            if (type(hx) == type(None)) and (type(cx) == type(None)):
                hx, cx = self.decoder(concat)
            else:
                hx, cx = self.decoder(concat,(hx, cx))

            out = self.linear(hx)


            if self.deterministic:
                out = self.soft(out)
                word = torch.argmax(out, dim=1)
            else:
                out = out / self.temperature
                word = torch.Tensor(
                    list(WeightedRandomSampler(self.soft(out), 1))).long().cuda()


                word = word.squeeze(0)

           	gen.append(int(word))
            count += 1
            # reached max length or end token
            if count == self.max_length or int(word) == 2:
                break

            concat = self.embedding(word)
            concat = torch.cat((concat, features), dim = 1)
            


        return torch.tensor(gen)

    def token2str(self, tokens):
        """
        converts tokens to string, for a sentence
        """
        return self.vocab.token_to_string(tokens)
