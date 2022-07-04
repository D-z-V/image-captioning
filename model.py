import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # pretrained model resnet50
        resnet = models.resnet50(pretrained=True)
        
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # replace the classifier with a fully connected embedding layer
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.word_embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, 
                            self.hidden_size, 
                            self.num_layers, 
                            batch_first=True)
        
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        
    
    def forward(self, features, captions):
        
        caption_embed = self.word_embedding(captions[:, :-1])
        caption_embed = torch.cat((features.unsqueeze(dim=1), caption_embed),1)
        output, self.hidden = self.lstm(caption_embed)

        output = self.fc(output)

        return output
