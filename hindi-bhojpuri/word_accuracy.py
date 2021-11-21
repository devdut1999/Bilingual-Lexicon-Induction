from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import nltk

#print(fuzz.ratio("this is a test", "this is a pre-test!"))

def get_string_similarity(i, j):
	return fuzz.ratio(i, j)


def load_data(data_file):
	text = open(data_file, 'r', encoding="utf-8").readlines()[1:]
	word_list = []

	for line in text:
		# line = line.strip()
		# stripped_line = line.replace('\u200b','')
		# stripped_line = line.replace('\u200d','').split('\t\t\t')
		# word_list.append(stripped_line)
		# print(word_list)
    src, trg = line.strip().split(",")
		l = []
    l.append(src)
    l.append(trg)
    word_list.append(l)

  X = [c[0] for c in word_list]
  y = [c[1] for c in word_list]
	for i,j in zip(X,y):
		print(i + '\t' + j)

	#X = [list(x) for x, w in zip(X, y) if len(x) > 0 and len(w) > 0] # list of lists
	#y = [list(w) for x, w in zip(X,y) if len(x) > 0 and len(w) > 0]

	
	return (X, y)
	

data_file = "/content/drive/MyDrive/Data_explo/valid_final.txt"
#data_file = "AttentionDecoder_with_dropout8batches0.2dropout.txt"
#data_file = "Transformer_output.txt"
# X and y being list of lists, each list contains characters of words
X, y = load_data(data_file)

cnt = 0
total = len(X)
for i,j in zip(X,y):
	if i == j:
		cnt += 1

res = cnt/total
print("Word accuracy of whole doc: ", res*100)
