import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


def build_dict(tokens:list) -> tuple[dict,dict]:
    """ 
    Builds a dictionary of tokens and returns ({token: index}, {index: token})
    """
    word2idx = {}
    cur_index = 0
    for i, t in enumerate(tokens):
        if t not in word2idx.keys():
            word2idx[t] = cur_index
            cur_index += 1
    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx, idx2word


class windows_ds(Dataset):
    """ 
    Creates a dataset of consecutive windows with a context token in the middle and
    other neighbouring tokens surrounding it
    """
    def __init__(self, tokens: list, word2idx: dict, window_size: int):
        super().__init__()
        self.tokens = torch.as_tensor([word2idx[t] for t in tokens])
        self.window_size = window_size
        self.padding = window_size // 2

    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, index): 
        """
        :returns: x (token index), y (indices of neighbouring tokens)
        """
        if index >= len(self):
            raise IndexError   
        start = max(0, index - self.padding)
        end = min(len(self), index + self.padding + 1)

        if start == 0:
            end = min(len(self), start + self.window_size)
        elif end == len(self):
            start = max(0, end - self.window_size)
        
        return self.tokens[index], torch.cat([self.tokens[start:index],self.tokens[index+1:end]])


def generate_embedding(word: str, w2i: dict, model: nn.Module) -> torch.Tensor:
    """ 
    :returns: embedding tensor of shape (1xD)
    """
    model.eval()
    embed = model(torch.as_tensor(w2i[word]).long()).cpu()
    model.train()
    return embed


# model init and training
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dims):
        super().__init__()
        self.embeds = nn.Embedding(vocab_size, embed_dims)

    def forward(self, center):
        return self.embeds(center)

def training_step(model_embed, x, y, optimizer):
    optimizer.zero_grad()
    
    # onehot encoding distribution
    y_hat = torch.zeros((len(y), model_embed.embeds.weight.shape[0]))
    for i, obs in enumerate(y):
        y_hat[i, obs] = 1.0

    inp_embed = model_embed(x)
    out = nn.LogSoftmax(dim=-1)(inp_embed @ model_embed.embeds.weight.T)
    loss = -(out*y_hat).mean()
    loss.backward()
    optimizer.step()

    return loss


# training
def train(data: str):
    """
    return: w2v_dict: dict
            - key: string (word)
            - value: np.array (embedding)
    """

    # prepping data
    tokens = data.split()
    w2i, i2w = build_dict(tokens)
    config = {
        'lr': 5e-4,
        'max_epochs': 1000,
        'vocab_size': len(w2i),
        'embed_d': 256,
        'skipgram_window_size': 3,
        'batch_size': 8
    }
    train_ds = windows_ds(tokens, w2i, config['skipgram_window_size'])  # dataset
    train_dl = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)  # dataloader
    model = SkipGram(config['vocab_size'],config['embed_d'])  # model init
    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'])  # optimizer init    

    pbar = tqdm(range(0,config['max_epochs']))
    av_losses = []
    for epoch in pbar:
        losses = []
        for x, y in train_dl:
            loss = training_step(model, x, y, optimizer)
            losses.append(loss)
        av_losses.append(sum(losses) / len(losses))
        pbar.set_postfix(epoch=epoch,av_loss=av_losses[-1].item(),refresh=False)

    return {w: generate_embedding(w,w2i,model) for w in w2i.keys()}
