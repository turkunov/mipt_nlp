import numpy as np
from tqdm.auto import tqdm

# preprocessing
def build_dict(tokens: list) -> tuple[dict,dict]:
    word2idx = {}
    idx2word = {}
    for i, t in enumerate(tokens):
        idx2word[i] = t
        word2idx[t] = i
    return word2idx, idx2word

def ohe(token_idx: int, dict_size: int):
    return [0 if i != token_idx else 1 for i in range(dict_size)]

def concat(*iterables):
    for iterable in iterables:
        yield from iterable

def generate_training_data(tokens: list, word2idx: dict, window: int):
    X = []
    y = []
    n_tokens = len(tokens)
    
    for i in range(n_tokens):
        idx = concat(
            range(max(0, i - window), i), 
            range(i, min(n_tokens, i + window + 1))
        )
        for j in idx:
            if i == j:
                continue
            X.append(ohe(word2idx[tokens[i]], len(word2idx)))
            y.append(ohe(word2idx[tokens[j]], len(word2idx)))

    return np.asarray(X), np.asarray(y)


# model init
def softmax(z: np.ndarray):
    centered_z = z - np.max(z,axis=1,keepdims=True)
    return np.exp(centered_z) / (np.sum(np.exp(centered_z), axis=1, keepdims=True)+1e-10)

def relu(x: np.ndarray):
    return np.maximum(0,x)

def accuracy(y_pred: np.ndarray, y_true: np.ndarray):
        return (np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1)).sum() / len(y_true)

def ce(y_pred: np.ndarray, y_true: np.ndarray):
    return -np.sum(y_true*np.log(y_pred+1e-10))

class optimizer:
    def __init__(self):
        pass

    def step(self):
        return NotImplementedError

class SGD_optimizer(optimizer):
    def __init__(self):
        super().__init__()

    def step(self, param: np.ndarray, grad: np.ndarray, lr: float):
        return param - lr * grad

class word2vec_lite_model:
    def __init__(self, optim: optimizer, embed_dims: int, vocab_size: int, lr: float = 1e-3):
        self.W1 = np.random.randn(vocab_size, embed_dims)
        self.W2 = np.random.randn(embed_dims, vocab_size)
        self.dLdW1 = np.zeros_like(self.W1)
        self.dLdW2 = np.zeros_like(self.W2)
        self.losses = []
        self.accuracies = []
        self.lr = lr
        self.cache = {}
        self.optimizer = optim

    def forward(self, X, y=None):
        self.cache['f'] = X @ self.W1
        if y is not None:
            self.cache['z1'] = relu(self.cache['f'])
        else:
            self.cache['z1'] = self.cache['f']
        self.cache['g'] = self.cache['z1'] @ self.W2
        self.cache['z2'] = softmax(self.cache['g'])
        if y is not None:
            self.losses.append(ce(self.cache['z2'],y))
            self.accuracies.append(accuracy(self.cache['z2'],y))
        return self.cache['z2']

    def backward(self, X, y):
        dLdg = self.cache['z2'] - y
        self.dLdW2 = self.cache['z1'].T @ dLdg
        self.dz1df = np.where(self.cache['f'] <= 0, 0, 1)
        self.dLdW1 = X.T @ (self.dz1df * (dLdg @ self.W2.T))

    def update(self):
        self.W1 = self.optimizer.step(param=self.W1, grad=self.dLdW1, lr=self.lr)
        self.W2 = self.optimizer.step(param=self.W2, grad=self.dLdW2, lr=self.lr)


# utilities
def generate_embedding(word: str, w2i: dict, model: word2vec_lite_model) -> np.ndarray:
    word_input = np.array([ohe(w2i[word], len(w2i))])
    _ = model.forward(word_input)
    return model.cache['f']


# training
def train(data: str):
    """
    return: w2v_dict: dict
            - key: string (word)
            - value: np.array (embedding)
    """
    tokens = data.split()
    w2i, i2w = build_dict(tokens)
    X, y = generate_training_data(tokens,w2i,3)

    config = {
        'lr': 5e-3,
        'max_epochs': 500,
        'vocab_size': len(w2i),
        'model': 'fc_relu_fc_softmax',
        'embed_d': 32,
    }
    optimizer = SGD_optimizer()
    model = word2vec_lite_model(
        optim=optimizer,
        embed_dims=config['embed_d'],
        vocab_size=config['vocab_size'],
        lr=config['lr'],
    )
    pbar = tqdm(range(0,config['max_epochs']))
    previous_loss = np.inf

    for epoch in pbar:
        _ = model.forward(X, y)
        model.backward(X, y)
        model.update()
        pbar.set_postfix(epoch=epoch,loss=model.losses[-1],refresh=False)
        if model.losses[-1] < previous_loss:
            previous_loss = model.losses[-1]
        elif model.losses[-1] / previous_loss >= 1.05:
            break

    return {w: generate_embedding(w,w2i,model) for w in w2i.keys()}, model
