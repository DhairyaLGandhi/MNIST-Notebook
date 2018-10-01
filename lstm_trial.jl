# # RNNs - Text Generation using LSTM
# ![title](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-SimpleRNN.png)
# An RNN works by continually remebering the previous states, by multiplying the gradients together continually as the backpropagation happens. 
# This causes the gradient exploding/ imploding problem.

# # LSTMs
# ![title](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

# # Steps of an LSTM
# LSTMs (Long Short Term Memory) are built to get around the issue of dying gradients. They allow for a lot of information to be preserved 
# by making the assumption that new predictions are largely goverened by the preceding few entries in the chain and not necessarily extending back to
# the entire history of the input sequence. In a way, they learn to forget. 

# # Step 1 - Choosing to Forget
# The first step in our LSTM is to decide what information we’re going to throw away from the cell state. 
# This decision is made by a sigmoid layer called the “forget gate layer"

# ![title](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-f.png)

# # Step 2 - Keep Incoming Information

# This has two steps:
# - First, a sigmoid layer called the “input gate layer” decides which values we’ll update. Next, a tanh layer creates a vector of new candidate values, C̃_t that could be added to the state.
# - We multiply the old state by f_t, forgetting the things we decided to forget earlier. Then we add i_t ∗ C̃_t. This is the new candidate values, scaled by how much we decided to update each state value.

# ![title](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-i.png)

# # Step 3 - Outputting Relevant Information

# This output will be based on our cell state, but will be a filtered version. First, we run a sigmoid layer which decides what parts of the cell state we’re going to output. 
# Then, we put the cell state through tanh (to push the values to be between −1 and 1) and multiply it by the output of the sigmoid gate, so that we only output the parts we decided to.

# ![title](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-C.png)

# We'll start with importing all the necessary bits from the Flux library.

# This code can also be found in the <a href = "https://github.com/FluxML/model-zoo.git"> model zoo </a>

using Flux
using Flux: onehot, argmax, chunk, batchseq, throttle, crossentropy
using StatsBase: wsample
using Base.Iterators: partition

# Style transfer is a classic task for newcomers interested in the LSTM world, we want to create a network which is able to produce text that resembles in its form, the text we input it.
# The way we do this is that we encode every possible outcome from prediction (in our case, it could be something like the alphabet, plus any punctuation that might be used in the text) in a manner
# that every such outcome has its own "label" so to speak, which is unqiue to it. It is known as a one-hot encoded vector where only one of the values is positive
# corresponding to the character we want to assign it to, and the remaining are `false`.

# ## Get the Data
# Now, we need to download a few MBs worth of a textfile, containing the life's works of a perceived genius playwright.
# The `download` utility is convenient for this task.
isfile("input.txt") ||
  download("http://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt",
           "input.txt")

# Now that we have our input data we need to worry about how to actually build our LSTM. 

text = collect(read("input.txt")) # read everything into a variable
alphabet = [unique(text)..., '_'] # get the unique characters out and add a stop symbol "_"

# Now we will one-hot encode the entire text
text = map(ch -> onehot(ch, alphabet), text)
stop = onehot('_', alphabet)

# We will define the length of sequences to train on, as well as the number of sequences per batch
N = length(alphabet)
seqlen = 50
nbatch = 50

Xs = collect(partition(batchseq(chunk(text, nbatch), stop), seqlen))
Ys = collect(partition(batchseq(chunk(text[2:end], nbatch), stop), seqlen))

# ## The Model
# ![title](https://cdn-images-1.medium.com/max/1600/1*NKhwsOYNUT5xU7Pyf6Znhg.png)

# LSTMs train similarly to traditional Recurrent neural networks, in that they unfold over the sequence they train over, as can be seen in the diagram above.

# Added to this, the internal structure of a LSTM layer makes it slightly more difficult to train on weaker hardware. 
# Let us define the model we will use. We already know that an LSTM is fairly suited for a basic language model, and extending it, we know we will need some
# way of choosing 1-of-many outputs as the predicted target value. Therefore, our model would need a softmax layer in the end. 

# We further will assume the output from the network is the correct answer and add it back to our known states, and extend our inference.

m = Chain(LSTM(N, 128),LSTM(128, 128),Dense(128, N),softmax)

# ## Getting the LSTM on the GPU

# And that's how easy it is to make a neural network model in Flux. It's pretty neat when you think about it.
# As i eluded to earlier, it is a bit of a niggle to effectively train LSTMs. To do it efficiently, it is standard practice to 
# to train them on GPUs, and even huge clusters of GPUs. And for that we have to ensure that our models are effectively transferred to the GPU.
# Now, usually, that bit in itself can amount to a significant effort being made to get these LSTMs trained on the GPU.
# Flux however, does have another trick up its sleeve.

m = gpu(m)

# And now when we train, we will do so on the GPU.

# ## The `loss` Function
# The loss function or the cost function is the function which we use to benchmark the performance of our model. These are defined in correspondence to the actual task at hand.
# Since, here our objective is to identify the next character which will be a part of our sequence, we will have to figure out which one of these 
# is best suited to our use case. Cross-entropy is a natural progession of that thought experiment, and that is exactly what we will use. 
# We will compare our result versus the expected value and train our network thus. 

function loss(xs, ys)
  l = sum(crossentropy.(m(xs), ys))
  Flux.truncate!(m)
  return l
end

# Notice the call to `m(xs)` in our little loss function. It represents a forward pass through the nwtwork, which is essentially an inference pass over the network.

opt = ADAM(params(m), 0.01)
tx, ty = (Xs[5], Ys[5])
evalcb = () -> @show loss(tx, ty)


# We will use the <a href = "">ADAM </a> optimiser and define a callback function which will display the loss. Later, we will set it so it shows us the loss every 30 secs.
# This will take a fair bit of time. 


# Flux.train!(loss, zip(Xs, Ys), opt,
#             cb = throttle(evalcb, 30))

# Now that we have a trained LSTM, its time to actually see what we have done. 
# For that, we will make a function that allows us to sample from the model using a seed that we choose at random. This seed is just a character or a sequence of characters that the
# model assumes is its start state. It will build from there.

function sample(m, alphabet, len; temp = 1)
  Flux.reset!(m)
  buf = IOBuffer()
  c = rand(alphabet)
  for i = 1:len
    write(buf, c)
    c = wsample(alphabet, m(onehot(c, alphabet)).data)
  end
  return String(take!(buf))
end

sample(m, alphabet, 1000) |> println

# It may not make much logical sense, but it does sound like what Shakespeare would've written all those years ago.

# # But that's not where it ends
# ![title](https://cdn-images-1.medium.com/max/1600/1*6YwqrScyczEaG0l05G4-_A.jpeg)
# An important thing to note is that, LSTMs or CNNs or anything of that sort can be used to create networks that can perform any task we explicitly train it for. 
# Certain ways of representing our problem better suit the algorithms we train them for, but in recent times, that hard limit has blurred significantly.

# The image you see above is an LSTM trained with handwriting data to generatively create the text we ask it
# to and the result is indistinguishable fom hand written.
# # Sampling from a Trained Dataset

using Flux
using Flux: onehot, argmax, chunk, batchseq, throttle, crossentropy
using StatsBase: wsample
using Base.Iterators: partition
using BSON: @load, @save

@load "/Users/dhairyagandhi/Downloads/shakespeare_weights.bson" weights
@load "/Users/dhairyagandhi/Downloads/shakespeare_alphabet.bson" alphabet

N = 68; 
m = Chain(
  LSTM(N, 128),
  LSTM(128, 128),
  Dense(128, N),
  softmax);

# `loadparams!` can be used to load the parameters of a model from external weights granted the shapes of the two matrices match.
Flux.loadparams!(m, weights)

# The same `sample` function as before.
function sample(m, alphabet, len; temp = 1)
     Flux.reset!(m)
     buf = IOBuffer()
     c = rand(alphabet)
     for i = 1:len
       write(buf, c)
       c = wsample(alphabet, m(onehot(c, alphabet)).data)
     end
     return String(take!(buf))
end

sample(m, alphabet, 1000) |> println
