import torch
import torch.nn.functional as F

# Hyper-Parameters
batchSize = 64 # Number of independent sequences of characters we want to process in parallel
blockSize = 256 # Maximum context length of predictions
learningRate = 3e-4
epochs = 50000
evaluationIntervals = 500
evaluationIterations = 200
numberOfEmbeddingDimensions = 384
numberOfHeads = 6
numberOfLayers = 6
headSize = numberOfEmbeddingDimensions
dropoutProbability = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Manual Seed
torch.manual_seed(69420)

# Dataset
with open('../Datasets/Harry_Potter_Books.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Vocabulary
characters = sorted(list(set(text))) # Gives us all the characters in the english alphabet, hopefully our dataset has all of them
vocabularySize = len(characters) # We define a common vocabulary size
stoi = {character:index for index, character in enumerate(characters)}
itos = {index:character for index, character in enumerate(characters)}
encode = lambda string: [stoi[character] for character in string] # Token Encoder that takes in a string as an input, and outputs a list of integers
decode = lambda list: ''.join([itos[index] for index in list]) # Token Decoder that takes in the encoded list of integers and outputs the decoded string

# Train-Validation Split
data = torch.tensor(encode(text), dtype=torch.long)
nintyPercentOfDatasetLength = int((0.9 * len(data)))
trainingData = data[:nintyPercentOfDatasetLength] # Data up till 90% of the length
validationData = data[nintyPercentOfDatasetLength:] # Data from 90% of the length

# Loading dataset into batches
def getBatch(split):
    # Take the trainingData if the split is 'train' otherwise take the validationData
    data = trainingData if split=='train' else validationData
    # Generates random integers of batchSize between 0 and len(data) - blockSize
    indexes = torch.randint(high=len(data) - blockSize, size=(batchSize,))
    # Takes the inputs and outputs after stacking them up in a single tensor
    inputs = torch.stack([data[i:i+blockSize] for i in indexes])
    outputs = torch.stack([data[i+1:i+blockSize+1] for i in indexes])
    inputs, outputs = inputs.to(device=device), outputs.to(device=device)
    return inputs, outputs

# Estimating Losses
@torch.no_grad()
def estimateLoss():
    output = {}
    # Set the model to evalutaion mode
    model.eval()

    for split in ['train','validation']:
        # Define a losses tensor for the `evaluationIterations` size
        losses = torch.zeros(evaluationIterations)
        for evaluationIteration in range(evaluationIterations):
            inputBatch, outputBatch = getBatch(split)
            logits, loss = model(inputBatch, outputBatch)
            losses[evaluationIteration] = loss.item()
        output[split] = losses.mean()
        
    # Set the model to training mode
    model.train()
    return output

# Head Module Definiton
class Head(torch.nn.Module):
    """ Single Head of Self Attention """
    # Constructor for the Head
    def __init__(self, headSize):
        super().__init__()
        self.key = torch.nn.Linear(numberOfEmbeddingDimensions, headSize, bias=False)
        self.query = torch.nn.Linear(numberOfEmbeddingDimensions, headSize, bias=False)
        self.value = torch.nn.Linear(numberOfEmbeddingDimensions, headSize, bias=False)
        self.register_buffer(name='lowerTriangularMatrix', tensor=torch.tril(torch.ones(blockSize, blockSize)))
        self.dropout = torch.nn.Dropout(p=dropoutProbability)

    # Forward Pass
    def forward(self, inputs):
        # Unpacking the shape of inputs
        batch, time, channel = inputs.shape
        # Forwarding the inputs to keys and queries
        k = self.key(inputs) # (B, T, C)
        q = self.query(inputs) # (B, T, C)
        # Initializing weights with scaled dot product
        weights = q @ k.transpose(-2, -1) * headSize ** -0.5 # (B, T, T)
        # Masking the weights
        weights = weights.masked_fill(self.lowerTriangularMatrix[:time, :time] == 0, float('-inf')) # (B, T, T)
        # Softmax the weights
        weights = F.softmax(weights, dim=-1) # (B, T, T)
        weights = self.dropout(weights)
        # Forwarding the inputs to values
        v = self.value(inputs) # (B, T, C)
        # Aggregating the weights and the values
        output = weights @ v # (B, T, C)
        return output

# Multi-Head Attention Module Definiton
class MultiHeadAttention(torch.nn.Module):
    """ Multiple Heads of Self Attention in Parallel """
    # Constructor for the Multi-Head Attention
    def __init__(self, numberOfHeads, headSize):
        super().__init__()
        self.heads = torch.nn.ModuleList([Head(headSize=headSize) for _ in range(numberOfHeads)])
        self.projection = torch.nn.Linear(numberOfEmbeddingDimensions, numberOfEmbeddingDimensions)
        self.dropout = torch.nn.Dropout(p=dropoutProbability)

    # Forward Pass
    def forward(self, inputs):
        # Returns the concatenated heads over the channel dimension
        output = torch.cat([head(inputs) for head in self.heads], dim=-1)
        output = self.dropout(self.projection(inputs))
        return output

# Feed Forward Module Definition
class FeedForward(torch.nn.Module):
    """ Simple Feed Forward Network """
    # Constructor for the Feed Forward Network
    def __init__(self, numberOfEmbeddingDimensions):
        # Initializing the layers
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(numberOfEmbeddingDimensions, 4 * numberOfEmbeddingDimensions),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * numberOfEmbeddingDimensions, numberOfEmbeddingDimensions),
            torch.nn.Dropout(p=dropoutProbability)
        )

    # Forward Pass
    def forward(self, inputs):
        return self.network(inputs)

# Transformer Block Module Definition
class TransformerBlock(torch.nn.Module):
    """ Communication Followed By Computation """
    # Constructor for the Transformer Block 
    def __init__(self, numberOfEmbeddingDimensions, numberOfHeads):
        # Initializing the Transformer Block
        super().__init__()
        self.selfAttention = MultiHeadAttention(numberOfHeads=numberOfHeads, headSize=numberOfEmbeddingDimensions//numberOfHeads)
        self.feedforwardnetwork = FeedForward(numberOfEmbeddingDimensions=numberOfEmbeddingDimensions)
        self.selfAttentionLayerNorm = torch.nn.LayerNorm(numberOfEmbeddingDimensions)
        self.feedforwardnetworkLayerNorm = torch.nn.LayerNorm(numberOfEmbeddingDimensions)

    # Forward Pass
    def forward(self, embeddings):
        embeddings = embeddings + self.selfAttention(self.selfAttentionLayerNorm(embeddings)) # (B, T, C)
        embeddings = embeddings + self.feedforwardnetwork(self.feedforwardnetworkLayerNorm(embeddings)) # (B, T, C)
        return embeddings

# Model Module Definition
class GPTModel(torch.nn.Module):
    # Constructor for the model
    def __init__(self):
        # Initializing the model parameters
        super().__init__()
        self.tokenEmbeddingTable = torch.nn.Embedding(vocabularySize, numberOfEmbeddingDimensions)
        self.positionalEmbeddingTable = torch.nn.Embedding(blockSize, numberOfEmbeddingDimensions)
        self.blocks = torch.nn.Sequential(*[TransformerBlock(numberOfEmbeddingDimensions=numberOfEmbeddingDimensions, numberOfHeads=numberOfHeads) for _ in range(numberOfLayers)])
        self.layerNorm = torch.nn.LayerNorm(numberOfEmbeddingDimensions)
        self.languageModelingHead = torch.nn.Linear(numberOfEmbeddingDimensions, vocabularySize)

    # Forward Pass
    def forward(self, indeces, labels=None):
        # Unpacking the shape of indeces
        batch, time = indeces.shape

        # Index into embeddings to get the token embeddings
        tokenEmbeddings = self.tokenEmbeddingTable(indeces) # (B, T, C)
        # Index into embeddings to get the positional embeddings
        positionalEmbeddings = self.positionalEmbeddingTable(torch.arange(time, device=device)) # (T, C)
        # Fuse the token embeddings and positional embeddings together to pack the information in a single tensor
        embeddings = tokenEmbeddings + positionalEmbeddings # (B, T, C)
        # Pass the concatenated embeddings into our blocks
        embeddings = self.blocks(embeddings) # (B, T, C)
        # Pass the embeddings to layer normalization
        embeddings = self.layerNorm(embeddings) # (B, T, C)
        # Pass the embeddings through a linear layer
        logits = self.languageModelingHead(embeddings) # (B, T, C)

        if labels is None:
            loss = None
        else:
            # Pop out the shape dimensions
            batch, time, channel = logits.shape
            # Stretch out the logits and labels
            logits = logits.view(batch*time, channel)
            labels = labels.view(batch*time)
            # Calculate loss
            loss = F.cross_entropy(logits, labels)
        return logits, loss

    # Generation
    def generate(self, indeces, maximumNewTokens):
        for _ in range(maximumNewTokens):
            # Crop the indeces upto most recent block size context
            croppedIndeces = indeces[:, -blockSize:]
            # Forward Through Model
            logits, loss = self(croppedIndeces)
            # Focus on the last time step
            logits = logits[:, -1, :]
            # Applying softmax for the last dimension
            probabilities = F.softmax(logits, dim=-1)
            # Sample from distribution
            nextIndex = torch.multinomial(probabilities, num_samples=1)
            # Concatenate currentIndex with nextIndex
            indeces = torch.cat((indeces, nextIndex), dim=1)
        return indeces

# Initializing the model
model = GPTModel().to(device=device)

# Initializing the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate)

for iteration in range(epochs):
    # Check if iteration reaches interval
    if iteration % evaluationIntervals == 0:
        # Save the losses in a variable
        losses = estimateLoss()
        # Print the losses (Training and Validation)
        print(f"Step {iteration}: Training Loss {losses['train']:.4f}, Validation Loss {losses['validation']:.4f}")

    
    # Get the inputBatch and outputBatch
    inputBatch, outputBatch = getBatch('train')
    # Forward the model
    logits, loss = model(inputBatch, outputBatch)
    # Setting the gradients to None
    optimizer.zero_grad(set_to_none=True)
    # Backward to calculate gradients
    loss.backward()
    # Update the gradients
    optimizer.step()

# Generating from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(indeces=context, maximumNewTokens=500)[0].tolist()))