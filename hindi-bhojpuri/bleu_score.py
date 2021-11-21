import nltk

def load_data(data_file):
	text = open(data_file, 'r', encoding="utf-8").readlines()

	word_list = []

	for line in text:
		line = line.strip()
		stripped_line = line.replace('\u200b','')
		stripped_line = line.replace('\u200d','').split(',')
		word_list.append(stripped_line)
		#print(word_list)
		
	X = [c[0] for c in word_list]
	y = [c[1] for c in word_list]

	X = [list(x) for x, w in zip(X, y) if len(x) > 0 and len(w) > 0] # list of lists
	y = [list(w) for x, w in zip(X,y) if len(x) > 0 and len(w) > 0]

	print(X[:20])
	print(y[:20])

	return (X, y)
	

data_file = "/content/drive/MyDrive/Data_explo/valid_final_2.txt"
# X and y being list of lists, each list contains characters of words
X, y = load_data(data_file)

X_max = max([len(word) for word in X])
y_max = max([len(word) for word in y])
# to find the total no of n-grams that can be encountered
max_len = max(X_max,y_max)

# BLEU score calculation 
bleu = 0

for i,j in zip(X,y):
	references = [i]
	list_of_references = [references]
	list_of_hypotheses = [j]
	# 11 equal weights for 11-grams 
	bleu += nltk.translate.bleu_score.sentence_bleu(references, j, 
		weights= (0.0909, 0.0909, 0.0909, 0.0909, 0.0909, 0.0909, 0.0909, 0.0909, 0.0909, 0.0909))

print(bleu/len(X))