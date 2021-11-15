
import re
import codecs
import numpy as np
import pandas as pd
import string
from nltk import ngrams #, word_tokenize,
from collections import Counter
import math


from sklearn.model_selection import train_test_split, cross_val_score
#from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

##########################################
'''
This first section focuses on pre-processing
The data in read in and punctuation is removed
/altered accordingly and all text is converted
to lower case.
'''
##########################################



txt_eng = ""
#Regex to find sentences within HTML brackets
TAG_RE = re.compile(r'<[^>]+>')

#Define punctuation to be removed
punc_replace = ["<<", ">>", ";", ":", "  ", ",,", "\"", "-", "--", "¡", "¿"]
#Define punctuation to be replaced with full stop
punc_replace_two = ["?", "!", "...", "\n", "…"]

#Reading in text files
def preprop(file, encoder):
	with open(file,"r",encoding=encoder) as f:
		text_eng = f.read()
	#Removing punctuation
	for punc in punc_replace:
		text_eng = text_eng.replace(punc, "")
	#Converting ounctuation to fullstop
	for punc in punc_replace_two:
		text_eng = text_eng.replace(punc, ".")

	#Replace HTML tags with nothing(remove them)
	text_eng = TAG_RE.sub('', text_eng)
	#Split text file based off full stop to indicate senetences
	sentences = text_eng.split(".")

	for i in range(len(sentences)):
		sentences[i] = sentences[i].strip()
	#Strip textfile into sentences
	sentences = [x for x in sentences if x != ""]
	return sentences

#Reading in text files in correct encoding
english = preprop("english.txt", "utf-8")
czech = preprop("czech.txt", "iso8859-2")
igbo = preprop("bbc-igbo.txt", "utf-8")
french = preprop("french.txt", "ISO-8859-1")
spanish = preprop("spanish.txt", "ISO-8859-1")

#Convert all letters to lower case
english = [x.lower() for x in english]
czech = [x.lower() for x in czech]
igbo = [x.lower() for x in igbo]
french = [x.lower() for x in french]
spanish = [x.lower() for x in spanish]


##########################################
'''
This first section converts the combined 
language data and calculates all the distinct 
possible bigrams and produces a normalsied 
count vector of all the bigrams
'''
##########################################


#Combine lists into one
calculate_biagrams_text = (english + czech + igbo + french + spanish)
file_nl_removed = ""
file_p = "".join(calculate_biagrams_text)
#Create a list of all the character bigrams
calculate_biagrams = list(ngrams(file_p, 2))

#Create a list of all distinct character biagrams
create_biagrams_header = list(set(calculate_biagrams))
calculate_biagrams_text_matrix = calculate_biagrams_text


#Define a function to create feature matrix
def make_matrix(headlines, vocab):
	#Initialise empty array
	matrix = []
	for headline in headlines:
		#Create a list off all the character biagrams
		headline = list(ngrams(headline, 2))
		# Count each word in the headline, and make a dictionary.
		counter = Counter(headline)
		# Turn the dictionary into a matrix row using the vocab.
		row = [counter.get(w, 0) for w in vocab]
		matrix.append(row)
	#Define array as a pandas datafrane
	df = pd.DataFrame(matrix)
	#Append column names
	df.columns = vocab
	return df
df = make_matrix(calculate_biagrams_text, create_biagrams_header)

X = np.array(df)
min_max_scaler = preprocessing.MinMaxScaler()
#Normalise to feature vector
X = min_max_scaler.fit_transform(X)

#Create a 1D array of the labels
y = np.array(['english']*len(english) + ['czech']*len(czech) + ['igbo']*len(igbo) + ['french']*len(french) + ['spanish']*len(spanish))


##########################################
'''
This section perfroms the classification 
algorithms using a 5 fold cross validation 
method to find the optimal hyperparemeters
'''
##########################################



#Define SKLearn classifier Multinominal Naive bayes
def MultinomND(X, y):
	#Define a list of seeds
	seed_list = [123, 145, 178]

	#Itterate through the seed list
	for j in seed_list:
		#Define empty accuracy list
		accur = []
		#Test train split with the predefined seed and set train size
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=j)

		#Itterate over new range for test size
		for i in range(1, 6):
			#Define enw test size
			i = i*0.1
			X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=i, random_state = j)

			#Define tuned parameter for CV
			tuned_parameters = [{'alpha': [0, 0.2, 0.4, 0.6, 0.8, 1]}]
			scores = ['precision', 'recall']
			#Perform cross validation default number is 5 cross validations
			clf = GridSearchCV(MultinomialNB(), tuned_parameters, scoring='f1_macro')
			#Fit the classifier
			clf.fit(X_train,y_train)
			print("Best parameters set found on development set:")	
			print(clf.best_params_)
			#Predict labels
			y_pred = clf.predict(X_test_1)
			#Produce classification report
			print(classification_report(y_test_1, y_pred))
			#Caldulate accuracy score
			acc = accuracy_score(y_test_1, y_pred)
			accur.append(acc)
		#Define x axis
		x = [0.1, 0.2, 0.3, 0.4, 0.5]
		plt.plot(x, accur, label = "Test train split with seed {}".format(j))
	plt.xlabel('Test train split size')
	plt.ylabel('Accuracy of predicted lables')
	plt.legend()
	plt.savefig('Multi.png')
	plt.close()

def GB(data, labels):
	labels = labels
	#Define crass val parameter
	tuned_parameters = [{'var_smoothing': [1e-9, 5e-9, 10e-9, 15e-9, 20e-9]}]
	#Define seed list
	seed_list = [123, 145, 178]
	for j in seed_list:
		accur = []
		X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.5,random_state=j)
		for i in range(1, 6):
			seed_split = i*3
			i = i*0.1
			X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(data, labels, test_size=i,random_state=j)
			print("rest train")
			#Perform 5 fold cross validation
			clf = GridSearchCV(GaussianNB(), tuned_parameters, scoring='f1_macro')
			#Fitting the model
			clf.fit(X_train, y_train)
			print("Best parameters set found on development set:")
			#Print best parameter value
			print(clf.best_params_)
			clf = clf.fit(X_train,y_train)
			#Compute predicted labels
			y_pred = clf.predict(X_test_1)
			#Produce classification report
			print(classification_report(y_test_1, y_pred))
			acc = accuracy_score(y_test_1, y_pred)
			accur.append(acc)
		#Define x axis
		x = [0.1, 0.2, 0.3, 0.4, 0.5]
		plt.plot(x, accur, label = "Test train split with seed {}".format(j))
	plt.xlabel('Test train split size')
	plt.ylabel('Accuracy of predicted lables')
	plt.savefig('GB.png')
	plt.close()


print("multinom class report")
MultinomND(X, y)

print("Gaussian bayes class report")
GB(X, y)



