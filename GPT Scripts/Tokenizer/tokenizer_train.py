unicodetext = open("../../Datasets/Tokenizer/tokenizer_train.txt", encoding="UTF-8").read()
rawByteTokens = unicodetext.encode("UTF-8")
decimalTokens = list(map(int, rawByteTokens))

def getPairFrequency(tokens):
    frequencyCounts = {}
    for pair in zip(tokens, tokens[1:]):
        frequencyCounts[pair] = frequencyCounts.get(pair, 0) + 1
    return frequencyCounts

def mergePair(tokens, pair, newVocabularyToken):
    encodedTokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
            encodedTokens.append(newVocabularyToken)
            i+=2
        else:
            encodedTokens.append(tokens[i])
            i+=1
    return encodedTokens

# Hyper-Parameter
vocabularySize = 276
# Calculating Number of Merges from the Hyper-parameter
numberOfMerges = vocabularySize - 256
# Creating the copy of the original token sequence
tokens = list(decimalTokens)

# Initializing the replacement table
replacementTable = {}

for i in range(numberOfMerges):
    pairFrequency = getPairFrequency(tokens=tokens)
    topPair = max(pairFrequency, key=pairFrequency.get)
    newVocabularyToken = 256 + i
    # print(f"Merging pair {topPair} by generating a new vocabulary token {newVocabularyToken}")
    tokens = mergePair(tokens=tokens, pair=topPair, newVocabularyToken=newVocabularyToken)
    replacementTable[topPair] = newVocabularyToken

# print("Length of original tokens:", len(decimalTokens))
# print("Length of encoded tokens:", len(tokens))
# print("Compression Ratio:", f"{len(decimalTokens) / len(tokens):.2f}X")

vocabulary = {index: bytes([index]) for index in range(256)}

for (position0, position1), index in replacementTable.items():
    vocabulary[index] = vocabulary[position0] + vocabulary[position1]


def decode(tokenSequence):
    tokens = b"".join(vocabulary[index] for index in tokenSequence)
    rawText = tokens.decode("UTF-8", errors="replace")
    return rawText

def encode(text):
    # Given a string, return a list of integers
    tokens = list(text.encode("UTF-8"))
    while len(tokens) >= 2:
        pairFrequency = getPairFrequency(tokens=tokens)
        minPair = min(pairFrequency, key=lambda pair: replacementTable.get(pair, float("inf")))
        # Nothing to merge
        if minPair not in replacementTable:
            break
        newVocabularyToken = replacementTable[minPair]
        tokens = mergePair(tokens=tokens, pair=minPair, newVocabularyToken=newVocabularyToken)
    return tokens

# Testing tokenizer
print(decode(encode("ये हिंदी है")))