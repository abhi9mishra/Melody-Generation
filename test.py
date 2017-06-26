import numpy as np 
Wxh=np.load('wxh.npy')
Whh=np.load('whh.npy')
Why=np.load('why.npy')
bh=np.load('bh.npy')
by=np.load('by.npy')
histogram = np.load('histogram.npy')
chars = np.load('chars.npy')
# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 100 # number of steps to unroll the RNN for
learning_rate = 1e-1
hprev = np.zeros((hidden_size,1))
# data I/O
data = open('input.txt', 'r').read() # should be simple plain text file
data_size, vocab_size = len(data), len(chars)
#print ('data has ',data_sizecharacters',vocab_size, unique.' ,data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }  
ix_to_char = { i:ch for i,ch in enumerate(chars) }

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in range(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

a=input()
ix=char_to_ix[a];
sample_ix = sample(hprev, ix, 200)
txt = ''.join(ix_to_char[ix] for ix in sample_ix)
print ("----\n" ,txt ," \n----" )

p=0
hist = np.zeros((vocab_size))
while p < len(txt):
	d0=char_to_ix[txt[p]]
	hist[d0]=hist[d0]+1
	p = p+1;

hist = hist/len(txt);

#np.seterr(divide-'ignore',invalid-'ignore')
ans = np.sum(np.where(np.logical_and(hist!=0,histogram!=0),histogram*np.log(histogram/hist),0))
print(ans)
