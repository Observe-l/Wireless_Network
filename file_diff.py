from difflib import SequenceMatcher

text1 = open('CMAPSSData/train_FD003.txt').read()
text2 = open('receive.txt').read()
m = SequenceMatcher(None, text1, text2)
print(m.quick_ratio())