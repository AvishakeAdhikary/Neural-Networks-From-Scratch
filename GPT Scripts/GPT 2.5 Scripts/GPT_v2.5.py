from dataclasses import dataclass
import torch
import torch.nn.functional as F
import math
import tiktoken
import time
import inspect

# Causal Self Attention (Scaled Dot-Product Attention + Multi-Head Attention)
class CausalSelfAttention(torch.nn.Module):
    def __init__(self, configuration):
        super().__init__()
        assert configuration.numberOfEmbeddingDimensions % configuration.numberOfHeads == 0
        self.causalAttention = torch.nn.Linear(configuration.numberOfEmbeddingDimensions, 3 * configuration.numberOfEmbeddingDimensions)
        self.causalProjection = torch.nn.Linear(configuration.numberOfEmbeddingDimensions, configuration.numberOfEmbeddingDimensions)
        self.numberOfHeads = configuration.numberOfHeads
        self.numberOfEmbeddingDimensions = configuration.numberOfEmbeddingDimensions
    
    def forward(self, inputs):
        B, T, C = inputs.size()
        query_key_value = self.causalAttention(inputs)
        query, key, value = query_key_value.split(self.numberOfEmbeddingDimensions, dim=2)
        query = query.view(B, T, self.numberOfHeads, C // self.numberOfHeads).transpose(1, 2)
        key = key.view(B, T, self.numberOfHeads, C // self.numberOfHeads).transpose(1, 2)
        value = value.view(B, T, self.numberOfHeads, C // self.numberOfHeads).transpose(1, 2)
        outputs = F.scaled_dot_product_attention(query=query, key=key, value=value, is_causal=True)
        outputs = outputs.transpose(1, 2).contiguous().view(B, T, C)
        outputs = self.causalProjection(outputs)
        return outputs

# Multi Layer Perceptron (MLP)
class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, configuration):
        super().__init__()
        self.currentFullyConnected = torch.nn.Linear(configuration.numberOfEmbeddingDimensions, 4 * configuration.numberOfEmbeddingDimensions)
        self.gelu = torch.nn.GELU(approximate="tanh")
        self.currentProjection = torch.nn.Linear(4 * configuration.numberOfEmbeddingDimensions, configuration.numberOfEmbeddingDimensions)

    def forward(self, inputs):
        inputs = self.currentFullyConnected(inputs)
        inputs = self.gelu(inputs)
        inputs = self.currentProjection(inputs)
        return inputs

# Transformer Block
class Block(torch.nn.Module):
    def __init__(self, configuration):
        super().__init__()
        self.layerNormalization1 = torch.nn.LayerNorm(configuration.numberOfEmbeddingDimensions)
        self.attention = CausalSelfAttention(configuration)
        self.layerNormalization2 = torch.nn.LayerNorm(configuration.numberOfEmbeddingDimensions)
        self.multiLayerPerceptron = MultiLayerPerceptron(configuration)
    
    def forward(self, inputs):
        inputs = inputs + self.attention(self.layerNormalization1(inputs))
        inputs = inputs + self.multiLayerPerceptron(self.layerNormalization2(inputs))
        return inputs

# GPT configuration hyper-parameters
@dataclass
class GPTConfiguration:
    blockSize: int = 1024
    vocabularySize: int = 50257
    numberOfLayers: int = 12
    numberOfHeads: int = 12
    numberOfEmbeddingDimensions: int = 768
    NANOGPT_SCALE_INIT: bool = True

# GPT model architecture
class GPTModel(torch.nn.Module):
    def __init__(self, configuration):
        super().__init__()
        self.configuration = configuration

        self.transformer = torch.nn.ModuleDict(dict(
            wordTokenEmbeddings = torch.nn.Embedding(configuration.vocabularySize, configuration.numberOfEmbeddingDimensions),
            wordPositionalEmbeddings = torch.nn.Embedding(configuration.blockSize, configuration.numberOfEmbeddingDimensions),
            hidden = torch.nn.ModuleList(Block(configuration) for _ in range(configuration.numberOfLayers)),
            finalLayerNorm = torch.nn.LayerNorm(configuration.numberOfEmbeddingDimensions)
        ))

        self.languageModelingHead = torch.nn.Linear(configuration.numberOfEmbeddingDimensions, configuration.vocabularySize, bias=False)
    
        # Weight-Sharing-Scheme (Parameter Weight Sharing)
        self.transformer.wordTokenEmbeddings.weight = self.languageModelingHead.weight

        # Initialize Correct Parameters
        self.apply(self._initializeParameters)

    def forward(self, indeces, labels=None):
        Batch, Time = indeces.size()
        assert Time <= self.configuration.blockSize, f"Cannot forward sequence of length {Time}, Block Size is only {self.configuration.blockSize}"

        tokenPositions = torch.arange(0, Time, dtype=torch.long, device=indeces.device)
        positionalEmbeddings = self.transformer.wordPositionalEmbeddings(tokenPositions)
        tokenEmbeddings = self.transformer.wordTokenEmbeddings(indeces)
        inputs = tokenEmbeddings + positionalEmbeddings

        for block in self.transformer.hidden:
            inputs = block(inputs)
        inputs = self.transformer.finalLayerNorm(inputs)
        logits = self.languageModelingHead(inputs)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return logits, loss

    def _initializeParameters(self, module):
        if isinstance(module, torch.nn.Linear):
            standardDeviation = 0.02
            if self.configuration.NANOGPT_SCALE_INIT:
                standardDeviation *= (2 * self.configuration.numberOfLayers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=standardDeviation)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # Method to transfer weights from Hugging Face GPT-2
    @classmethod
    def from_pretrained(cls, modelType):
        assert modelType in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("Loading weights from pretrained GPT: %s" % modelType)
        
        # Creating separate configurations for separate GPT-2 models
        blockSize = 1024
        vocabularySize = 50257
        configurationArguements = {
            'gpt2':         dict(numberOfLayers=12, numberOfHeads=12, numberOfEmbeddingDimensions=768),  # 124M parameters
            'gpt2-medium':  dict(numberOfLayers=24, numberOfHeads=16, numberOfEmbeddingDimensions=1024), # 350M parameters
            'gpt2-large':   dict(numberOfLayers=36, numberOfHeads=20, numberOfEmbeddingDimensions=1280), # 774M parameters
            'gpt2-xl':      dict(numberOfLayers=48, numberOfHeads=25, numberOfEmbeddingDimensions=1600), # 1558M parameters
        }[modelType]
        configurationArguements['vocabularySize'] = 50257
        configurationArguements['blockSize'] = 1024

        configuration = GPTConfiguration(**configurationArguements)
        model = GPTModel(configuration)
        stateDictionary = model.state_dict()
        stateDictionaryKeys = stateDictionary.keys()
        stateDictionaryKeys = [key for key in stateDictionaryKeys if not key.endswith('.attention.bias')]

        huggingfaceModel = GPT2LMHeadModel.from_pretrained(modelType)
        huggingfaceStateDictionary = huggingfaceModel.state_dict()
        huggingfaceStateDictionaryKeys = huggingfaceStateDictionary.keys()
        huggingfaceStateDictionaryKeys = [key for key in huggingfaceStateDictionaryKeys if not key.endswith('.attn.masked_bias')]
        huggingfaceStateDictionaryKeys = [key for key in huggingfaceStateDictionaryKeys if not key.endswith('.attn.bias')]
        transposedParameters = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(huggingfaceStateDictionaryKeys) == len(stateDictionaryKeys), f"Mismatched Keys: {len(huggingfaceStateDictionaryKeys)} != {len(stateDictionaryKeys)}"

        parameterKeyMapping = {
            customKey: huggingfaceKey
            for customKey, huggingfaceKey in zip(stateDictionaryKeys, huggingfaceStateDictionaryKeys)
            }

        for customKey, huggingfaceKey in parameterKeyMapping.items():
            if (huggingfaceStateDictionary[huggingfaceKey].shape != stateDictionary[customKey].shape):
                # Special treatment for the Conv1D weights (Transposed Weights)
                if (huggingfaceKey.endswith(word) for word in transposedParameters):
                    assert huggingfaceStateDictionary[huggingfaceKey].shape[::-1] == stateDictionary[customKey].shape
                    with torch.no_grad():
                        stateDictionary[customKey].copy_(huggingfaceStateDictionary[huggingfaceKey].t())
            # Vanilla copy for other parameters
            else:
                assert huggingfaceStateDictionary[huggingfaceKey].shape == stateDictionary[customKey].shape
                with torch.no_grad():
                    stateDictionary[customKey].copy_(huggingfaceStateDictionary[huggingfaceKey])
        return model
    
    def configureOptimizers(self, weightDecayRate, learningRate, device):
        parameterDictionary = {parameterNumber: parameter for parameterNumber, parameter in self.named_parameters()}
        parameterDictionary = {parameterNumber: parameter for parameterNumber, parameter in parameterDictionary.items() if parameter.requires_grad}

        decayParameters = [parameter for parameterNumber, parameter in parameterDictionary.items() if parameter.dim() >= 2]
        nonDecayParameters = [parameter for parameterNumber, parameter in parameterDictionary.items() if parameter.dim() < 2]

        optimizerGroups = [
            {
                'params': decayParameters, 'weight_decay': weightDecayRate
            },
            {
                'params': nonDecayParameters, 'weight_decay': 0.0
            }
        ]
        numberOfDecayParameters = sum(parameter.numel() for parameter in decayParameters)
        numberOfNonDecayParameters = sum(parameter.numel() for parameter in nonDecayParameters)

        print(f"Number of decayed parameter tensors: {len(decayParameters)}, with {numberOfDecayParameters} parameters")
        print(f"Number of non-decayed parameter tensors: {len(nonDecayParameters)}, with {numberOfNonDecayParameters} parameters")

        isFusedAvailable = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        useFused = isFusedAvailable and device == "cuda"
        print(f"Using Fused AdamW: {useFused}")
        optimizer = torch.optim.AdamW(params=optimizerGroups, lr=learningRate, betas=(0.9, 0.95), eps=1e-8, fused=useFused)
        return optimizer


# Data-Loader
class DataLoaderLite:
    def __init__(self, Batch, Time):
        self.Batch = Batch
        self.Time = Time
        with open("Datasets/Harry_Potter_Books.txt", "r", encoding="UTF-8") as file:
            text = file.read()
        encoder = tiktoken.get_encoding('gpt2')
        encodedDataTokens = encoder.encode(text)
        self.encodedDataTokens = torch.tensor(encodedDataTokens)
        print(f"Loaded {len(self.encodedDataTokens)} Tokens")
        print(f"1 Epoch = {len(self.encodedDataTokens) // (Batch * Time)} Batches")

        # State
        self.currentPosition = 0
        
    def nextBatch(self):
        Batch, Time = self.Batch, self.Time
        buffer = self.encodedDataTokens[self.currentPosition : self.currentPosition + Batch*Time + 1]
        inputs = buffer[:-1].view(Batch, Time)
        labels = buffer[1:].view(Batch, Time)
        self.currentPosition += Batch * Time
        if self.currentPosition + (Batch * Time + 1) > len(self.encodedDataTokens):
            self.currentPosition = 0
        return inputs, labels

# Device Auto-Detection
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"Using Device: {device}")

# Data-Loader
totalBatchSize = 524288
Batch, Time = 4, 32
assert totalBatchSize % (Batch * Time) == 0, f"Make sure totalBatchSize is divisible by (Batch * Time)"
gradientAccumulationSteps = totalBatchSize // (Batch * Time)
print(f"Total Desired Batch Size: {totalBatchSize}")
print(f"Calculated Gradient Accumulation Steps: {gradientAccumulationSteps}")
trainingLoader = DataLoaderLite(Batch=Batch, Time=Time)

# Constructing Model
model = GPTModel(GPTConfiguration(vocabularySize=50304))

model.eval()
model.to(device=device)
model = torch.compile(model)

# Optimization
maximumLearningRate = 6e-4
minimumLearningRate = maximumLearningRate * 0.1
warmupEpochs = 10
epochs = 50
def getLearningRate(epoch):
    if epoch < warmupEpochs:
        return maximumLearningRate * (epoch + 1) / warmupEpochs
    if epoch > epochs:
        return minimumLearningRate
    decayRatio = (epoch - warmupEpochs) / (epochs - warmupEpochs)
    assert 0 <= decayRatio <= 1
    coefficient = 0.5 * (1.0 + math.cos(math.pi * decayRatio))
    return minimumLearningRate + coefficient * (maximumLearningRate - minimumLearningRate)

optimizer = model.configureOptimizers(weightDecayRate=0.1, learningRate=maximumLearningRate, device=device)
for epoch in range(epochs):
    startTime = time.time()
    optimizer.zero_grad()
    lossAccumulator = 0.0
    for microStep in range(gradientAccumulationSteps):
        inputs, labels = trainingLoader.nextBatch()
        inputs, labels = inputs.to(device=device), labels.to(device=device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(inputs, labels)
        loss = loss / gradientAccumulationSteps
        lossAccumulator += loss.detach()
        loss.backward()
    normalization = torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
    learningRate = getLearningRate(epoch=epoch)
    for parameterGroup in optimizer.param_groups:
        parameterGroup['lr'] = learningRate
    optimizer.step()
    torch.cuda.synchronize()
    endTime = time.time()
    timeDifference = (endTime - startTime) * 1000
    tokensPerSecond = (trainingLoader.Batch * trainingLoader.Time) * gradientAccumulationSteps / (endTime - startTime)
    print(f"Step: {epoch}, Loss: {lossAccumulator.item():.6f}, Learning Rate: {learningRate:.4e},  Normalization: {normalization:.4f}, Time Difference: {timeDifference:.2f}ms, Tokens/Second: {tokensPerSecond:.2f}tokens/sec")

# Halting Generation...(Will Remove Later)
import sys; sys.exit(0)

# Generation
maximumGenerationLength = 30
numberOfSequences = 5

encoder = tiktoken.get_encoding('gpt2')
encodedTokens = encoder.encode("Hello, I'm a language model,")
encodedTokens = torch.tensor(encodedTokens, dtype=torch.long)
encodedTokens = encodedTokens.unsqueeze(0).repeat(numberOfSequences, 1)
inputs = encodedTokens.to(device=device)

torch.manual_seed(69)
torch.cuda.manual_seed(69)

while inputs.size(1) < maximumGenerationLength:
    with torch.no_grad():
        logits, loss = model(inputs)
        logits = logits[:, -1, :]
        probabilites = F.softmax(logits, dim=-1)

        topKProbabilites, tokKIndeces = torch.topk(input=probabilites, k=50, dim=-1)

        tokenIndeces = torch.multinomial(input=topKProbabilites, num_samples=1)
        columnOfTokenIndeces = torch.gather(input=tokKIndeces, dim=-1, index=tokenIndeces)

        inputs = torch.cat((inputs, columnOfTokenIndeces), dim=1)

for i in range(numberOfSequences):
    tokensToDecode = inputs[i, :maximumGenerationLength].tolist()
    decodedTokens = encoder.decode(tokensToDecode)
    print(">", decodedTokens)

# # If you want to remove the HuggingFace Model manually
# from transformers import file_utils
# print(file_utils.default_cache_path)