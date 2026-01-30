"""
Uses a python data dump via
https://github.com/acmeism/RosettaCodeData/tree/main
which I combined

git clone https://github.com/acmeism/RosettaCodeData/tree/main
circa 2400 files
cd RosettaCodeData/Lang/Python
mkdir all_files
for file in */*; do
  [[ -f "$file" ]] && cp "$file" all_files/
done
cd all_files
cat *.py > input.txt
input is 91k lines, I then zipped it
"""
import os
import pickle
import requests
import zipfile
import numpy as np

# take the rosetta python code
input_file_path = os.path.join(os.path.dirname(__file__), 'input.zip')
with zipfile.ZipFile(input_file_path, 'r') as zf:
    with zf.open('input.txt', 'r') as f:
        data = f.read().decode('utf-8')
# if I had the unzipped file I could do:
#input_file_path="input.txt"
#with open(input_file_path, 'r') as f:
#    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

#length of dataset in characters: 2,557,167
#all the unique characters: 	
# !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~£«°±²³·¹»¼½ßàáçèéìíóöøùúăēīĭŉŏūŭſǎǐǒǔɛʒΓΔΘΚΛΞΠΡΣΦΧΨΩαβγδεζηθικλμνξοπρςστυφχψωЖאתᵢ​–—‘’“”‾⁰⁴⁵⁶⁷⁸⁹€ℤⅫ→∈−∩∪≈≠≡␡␢─━│┃┌┍┎┏┐┑┒┓└┕┖┗┘┙┚┛├┝┞┟┠┡┢┣┤┥┦┧┨┩┪┫┬┭┮┯┰┱┲┳┴┵┶┷┸┹┺┻┼┽┾┿╀╁╂╃╄╅╆╇╈╉╊╋╱╲╴╵╶╷╸╹╺╻╼╽╾╿▁▂▃▄▅▆▇█▓△►●◸◹◺◻◿♘♛♞♠♣♥♦は一丁下丑丙乙二五亥今代兔初化十午卯图土地壬天始子寅己巳干庚戊戌成支日木未水火牛狗猴生甲申画癸空羊肖虎蛇行表豬辛辰酉金阳阴馬鸡鼠龍�𝄞𝒪𝔘𝔠𝔡𝔢𝔦𝔫𝔬🌱🍂💃😀😍🙌
#vocab size: 393
#train has 2,301,450 tokens
#val has 255,717 tokens

