from dataclasses import dataclass
import torch
import torch.nn.functional as F

# Transformer Block
class Block:
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