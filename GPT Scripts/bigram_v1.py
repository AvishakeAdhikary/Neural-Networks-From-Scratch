import torch
import torch.nn.functional as F

# Hyper-Parameters
batchSize = 32 # Number of independent sequences of characters we want to process in parallel
blockSize = 8 # Maximum context length of predictions
learningRate = 1e-2
epochs = 30000
evaluationIntervals = 500
evaluationIterations = 200
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

# Model Module Definition
class BigramLanguageModel(torch.nn.Module):
    # Constructor for the model
    def __init__(self, vocabularySize):
        # Initializing the embedding table
        super().__init__()
        self.tokenEmbeddingTable = torch.nn.Embedding(vocabularySize, vocabularySize)

    # Forward Pass
    def forward(self, indeces, labels=None):
        # Index into embeddings to get the logits
        logits = self.tokenEmbeddingTable(indeces) # (B, T, C)
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
            # Forward Through Model
            logits, loss = self(indeces)
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
model = BigramLanguageModel(vocabularySize).to(device=device)

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