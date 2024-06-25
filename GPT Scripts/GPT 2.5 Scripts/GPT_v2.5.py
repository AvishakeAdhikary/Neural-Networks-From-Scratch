from dataclasses import dataclass
import torch
import torch.nn.functional as F


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
        ))

model = GPTModel(GPTConfiguration())