###################################################################
#				Importing the libraries					  		  #
###################################################################
"""
Numpy: It is the core library for scientific computing in Python. It provides a high-performance multidimensional array object and tools for working with these arrays.

Matplotlib: Here the import line merely imports the module matplotlib. pyplot and binds that to the name plt.It is used for plotting graphs in python.

Pandas: Pandas is an open-source library that allows us to perform data manipulation in Python. It is built on NumPy meaning that pandas cannot operate until NumPy is imported.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


###################################################################
#				Importing the dataset				  	  		  #
###################################################################
"""
The data we have is in the form of tsv file which is imported with the help of ‘read_csv’ function of pandas library but since we need to add tsv file we set delimiter parameter to /t 
"""
dataset = pd.read_csv('Reviews.tsv', delimiter = '\t', quoting = 3,encoding= 'unicode_escape')


###################################################################
#				Cleaning the texts					  			  #
###################################################################
"""
LIBRARIES:
1)Re: Built-in re module provides excellent support for regular expressions, with a modern and complete regular expression flavour.
2)NLTK: NLTK is one of the leading platforms for working with human language data and Python, the module NLTK is used for natural language processing. NLTK is literally an acronym for Natural Language Toolkit.
	Stopwords: It is a  list of words which are not much importance to categorize review such as(are, is, that, those, etc)
3)PorterStemmer: It is yet another library imported for the stemming process
"""
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

"""
FUNCTION:
In this part of code, the data containing review is being clean only the parts that are necessary for the computation of helpful review are used: 
1)TOKENIZATION: Here all the reviews are tokenized meaning that all the characters and punctuation are removed and the review only has characters from a-z and A-Z.
2)LowerCase conversion: All the alphabets in the review are converted in lowercase so as to ease the process learning for the machine.
3)STEMMING: Here all the words are converted into its root form such as(kindest, kindness, etc converts to kind). 
4)Stopword removal: All the stopwords are removed from the reviews.

INPUT:
The review column of the tsv file is taken as the input.

OUTPUT:
The cleaned or filtered review is the output.

"""

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


###################################################################
#			Creating the Bag of Words model				          #
###################################################################
"""
LIBRARIES
CountVectorizer from Feature_Extraction of ‘scikit-learn’: this library is used to extract some features of a language that are used very much in a particular review to predict the result and convert the data into a sparse matrix.

FUNCTION:
To extract the features from the cleaned dataset and then convert the data into a sparse matrix to tell whether a feature is present in a review or not 

INPUT:
The input given here is the text that has filtered before.

OUTPUT:
The sparse matrix is formed of the dataset after feature extraction.
"""
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,4 ].values


###################################################################
#				Splitting the dataset					  		  #
###################################################################
"""
FUNCTION:
To split the dataset into training and test sets for machines to learn and predict results.

INPUT:  
Sparse Matrix created in the previous section.

OUTPUT:
The training and test set of the sparse matrix.
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

###################################################################
#				Fitting Naive Bayes					 			  #
###################################################################
"""
Here class “GaussianNB” is imported from sklearn.naive_bayes. Furthermore, we created an object of class GaussianNB and then fitted the Naive Bayes model to the data.
In Gaussian Naive Bayes, continuous values associated with each feature are assumed to be distributed according to a Gaussian distribution or normal distribution.

"""
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

###################################################################
#	    Similarly other classification algorithm fitting		  #
###################################################################

"""# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0,probability= True)
classifier.fit(X_train, y_train)"""


###################################################################
#			Predicting the Test set result		              	  #
###################################################################
"""
FUNCTION: Here classifier.predict tries to classify the data and returns the predicted value into y_pred
INPUT:   X_test
OUTPUT: Y_pred.
"""
y_pred = classifier.predict(X_test)


###################################################################
#				Making the Confusion Matrix				  		  #
###################################################################
"""
LIBRARY:
Here class ’confusion_matrix’ is imported from sklearn.metrics and ‘cm’ object does the creation of a performance matrix which compares test values and predicted values in y matrix.

FUNCTION:
To create a confusion matrix to predict accuracy.
i)A confusion matrix is a summary of prediction results on a classification problem. The confusion matrix shows the ways in which your classification model is confused when it makes predictions
ii)Accuracy=(True Positive+True Negative)/(True Positive+True Negative+False Positive+False Negative )
INPUT:
y_test and y_pred.

OUTPUT:
Confusion Matrix.
"""
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
