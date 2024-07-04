from dataclasses import dataclass
import torch
import torch.nn.functional as F
import math

# Causal Self Attention (Scaled Dot-Product Attention + Multi-Head Attention)
class CausalSelfAttention(torch.nn.Module):
    def __init__(self, configuration):
        super().__init__()
        assert configuration.numberOfEmbeddingDimensions % configuration.numberOfHeads == 0
        self.causalAttention = torch.nn.Linear(configuration.numberOfEmbeddingDimensions, 3 * configuration.numberOfEmbeddingDimensions)
        self.causalProjection = torch.nn.Linear(3 * configuration.numberOfEmbeddingDimensions, configuration.numberOfEmbeddingDimensions)
        self.numberOfHeads = configuration.numberOfHeads
        self.numberOfEmbeddingDimensions = configuration.numberOfEmbeddingDimensions
        self.register_buffer("bias", torch.tril(torch.ones(configuration.blockSize, configuration.blockSize)).view(1, 1, configuration.blockSize, configuration.blockSize))
    
    def forward(self, inputs):
        B, T, C = inputs.size()
        query_key_value = self.causalAttention(inputs)
        query, key, value = query_key_value.split(self.numberOfEmbeddingDimensions, dim=2)
        query = query.view(B, T, self.numberOfHeads, C // self.numberOfHeads).transpose(1, 2)
        key = key.view(B, T, self.numberOfHeads, C // self.numberOfHeads).transpose(1, 2)
        value = value.view(B, T, self.numberOfHeads, C // self.numberOfHeads).transpose(1, 2)
        attention = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
        attention = attention.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        attention = F.softmax(attention, dim=-1)
        outputs = attention @ value
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
        inputs = inputs + self.multiLayerPerceptron(self.layerNormalization1(inputs))

# GPT configuration hyper-parameters
@dataclass
class GPTConfiguration:
    blockSize = 1024
    vocabularySize = 50257
    numberOfLayers = 12
    numberOfHeads = 12
    numberOfEmbeddingDimensions = 768

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

model = GPTModel(GPTConfiguration())
print(model)