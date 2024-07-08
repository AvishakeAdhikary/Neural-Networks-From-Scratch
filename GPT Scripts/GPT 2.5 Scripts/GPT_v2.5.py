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
        self.causalProjection = torch.nn.Linear(configuration.numberOfEmbeddingDimensions, configuration.numberOfEmbeddingDimensions)
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
    blockSize: int = 1024
    vocabularySize: int = 50257
    numberOfLayers: int = 12
    numberOfHeads: int = 12
    numberOfEmbeddingDimensions: int = 768

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
    
    # Method to transfer weights from Hugging Face GPT-2
    @classmethod
    def from_pretrained(cls, modelType):
        assert modelType in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("Loading weights from pretrained gpt: %s" % modelType)
        
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

model = GPTModel.from_pretrained('gpt2')
print(model)


# # If you want to remove the HuggingFace Model manually
# from transformers import file_utils
# print(file_utils.default_cache_path)