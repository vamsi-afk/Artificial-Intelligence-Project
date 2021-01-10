
# importing the required libraries
from flask import Flask
from flask_restful import Api, Resource
import pandas as pd
# creating an app within flask API
app = Flask(__name__)
# wrapping app in an API
api = Api(app)
# creating a resource with an API
class HelloWorld(Resource):
    # getting a GET request
    def post(self,name):
        # Importing the required SKlearnModels
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        import re
        # Importing the nltk library and downloading the stopwords.
        import nltk
        nltk.download('wordnet')
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        # Initializing an empty list
        corpus = []
        # Importing the training data
        data = pd.read_csv("airline_sentiment_analysis1.csv")
        for i in range(0, 11541):
            review = re.sub('[^a-zA-z]', ' ', data['text'][i])
            review = review.lower()
            review = review.split()
            ps = WordNetLemmatizer()
            all_stopwords = stopwords.words('english')
            review = [ps.lemmatize(word) for word in review if word not in set(all_stopwords)]
            review = ' '.join(review)
            corpus.append(review)
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.preprocessing import LabelEncoder
        # Encoding the sentiment column into 1 and 0 (1 for positive and 0 for negative)
        encoder = LabelEncoder()
        # Creating an object of CountVectorizer class
        cv = CountVectorizer()
        # Initializing the sentiment (labelled column)
        x1 = data.iloc[:, 1].values
        # Converting the reviews into an array of BagOfWords
        x2 = cv.fit_transform(corpus).toarray()
        x1 = encoder.fit_transform(x1)
        # Splitting the data into training set and the test set
        x_train, x_test, y_train, y_test = train_test_split(x2, x1, test_size=0.2)
        from sklearn.ensemble import RandomForestClassifier
        # Importing and initializing the RandomForestClassifier Class
        classifier = RandomForestClassifier()
        # Training the model on the dataset
        classifier.fit(x_train, y_train)
        # Predicting the test set results
        y_pred = classifier.predict(x_test)
        # Calculating the test set accuracy
        score = accuracy_score(y_test, y_pred)
        # Same process as above for the inputted text
        corpus1 = []
        name = name.lower()
        name = name.split()
        ps1 = WordNetLemmatizer()
        all_stopwords = stopwords.words('english')
        name = [ps1.lemmatize(word) for word in name if word not in set(all_stopwords)]
        name = ' '.join(name)
        corpus1.append(name)
        name1 = cv.transform(corpus1).toarray()
        y_pred1 = classifier.predict(name1)
        # Classifiying the prediction as positive and negative
        if y_pred1 == 0:
            return {"Sentiment": "Negative"}
        else:
            return {"Sentiment": "Postive"}

# Registering the resource and making it accessible through a URL


api.add_resource(HelloWorld, "/helloworld/<string:name>")
# starting the application server
if __name__ == "__main__":
    app.run(debug=True)
