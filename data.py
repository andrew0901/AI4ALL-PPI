import torch
import json
from torch.utils.data import Dataset

mapping = {'A': 0,
 'C': 1,
 'D': 2,
 'E': 3,
 'F': 4,
 'G': 5,
 'H': 6,
 'I': 7,
 'K': 8,
 'L': 9,
 'M': 10,
 'N': 11,
 'P': 12,
 'Q': 13,
 'R': 14,
 'S': 15,
 'T': 16,
 'V': 17,
 'W': 18,
 'Y': 19}

class ProteinInterfaceDataset(Dataset):
    def __init__(self, path_to_json):
        with open(path_to_json) as f:
            self.data = json.load(f)
        self.idx = list(self.data.keys())
        self.mapping = mapping

    def __len__(self):
        return len(self.idx)
    
    def one_hot_seq(self, seq):
        #one-hot encode a sequence of amino acid codes
        #since no input data are longer than 300, and they have variable length, all of them are padded to length 300
        #returned data are in size (300, 20), because there are 20 amino acids
        one_hot = torch.zeros((300, 20))
        for i in range(len(seq)):
            one_hot[i][self.mapping[seq[i]]] = 1
        return one_hot

    def get_Y(self, res):
        #encode a list of list of binding partners into a tensor of labels
        #any position that has a partner get a label of 1
        #padding like before, so output a size (300,) tensor
        Y = torch.zeros(300)
        for i in range(len(res)):
            if res[i]:
                Y[i] = 1
        return Y

    def __getitem__(self, idx):
        #sequence information from both sequences in a pair of partners are concatenated
        #so the output size are (600,20) and (600,)
        dat = self.data[self.idx[idx]]
        seqA, seqB, resA, resB = dat['seqA'], dat['seqB'], dat['binding_partner_A'],  dat['binding_partner_B']
        X = torch.cat((self.one_hot_seq(seqA), self.one_hot_seq(seqB)))
        Y = torch.cat((self.get_Y(resA), self.get_Y(resB)))
        return X, Y