import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

# Cleaning Data:
# 1 URLs,
# 2 hashtags,
# 3 mentions,
# 4 special letters,
# 5 punctuations:
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

def main():
    df = pd.read_csv('UpdatedResumeDataSet.csv')
    print(df['Category'].value_counts())

    plt.figure(figsize=(15, 5))
    sns.countplot(x='Category', data=df)
    plt.xticks(rotation=90)
    plt.show()

    counts = df['Category'].value_counts()
    labels = df['Category'].unique()
    plt.figure(figsize=(15, 10))

    plt.pie(counts, labels=labels, autopct='%1.1f%%', shadow=True, colors=plt.cm.plasma(np.linspace(0, 1, len(labels))))
    plt.show()

    df['Resume'] = df['Resume'].apply(lambda x: cleanResume(x))

    print(df['Resume'][0])
    # words into categorical values
    le = LabelEncoder()
    le.fit(df['Category'])
    df['Category'] = le.transform(df['Category'])

    # Vectorization
    tfidf = TfidfVectorizer(stop_words='english')

    tfidf.fit(df['Resume'])
    requiredText = tfidf.transform(df['Resume'])

    # Save the trained TfidfVectorizer
    with open('tfidf.pkl', 'wb') as f:
        pickle.dump(tfidf, f)

    X_train, X_test, y_train, y_test = train_test_split(requiredText, df['Category'], test_size=0.2, random_state=42)
    print(X_train.shape)
    print(X_test.shape)

    # Train the model and print the classification report
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    clf.fit(X_train, y_train)
    ypred = clf.predict(X_test)
    print(accuracy_score(y_test, ypred))
    print(ypred)

    # Save the trained classifier
    with open('clf.pkl', 'wb') as f:
        pickle.dump(clf, f)

    myresume = """I am a data scientist specializing in machine
    learning, deep learning, and computer vision. With
    a strong background in mathematics, statistics,
    and programming, I am passionate about
    uncovering hidden patterns and insights in data.
    I have extensive experience in developing
    predictive models, implementing deep learning
    algorithms, and designing computer vision
    systems. My technical skills include proficiency in
    Python, Sklearn, TensorFlow, and PyTorch.
    What sets me apart is my ability to effectively
    communicate complex concepts to diverse
    audiences. I excel in translating technical insights
    into actionable recommendations that drive
    informed decision-making.
    If you're looking for a dedicated and versatile data
    scientist to collaborate on impactful projects, I am
    eager to contribute my expertise. Let's harness the
    power of data together to unlock new possibilities
    and shape a better future.
    Contact & Sources
    Email: 611noorsaeed@gmail.com
    Phone: 03442826192
    Github: https://github.com/611noorsaeed
    Linkdin: https://www.linkedin.com/in/noor-saeed654a23263/
    Blogs: https://medium.com/@611noorsaeed
    Youtube: Artificial Intelligence
    ABOUT ME
    WORK EXPERIENCE
    SKILLES
    NOOR SAEED
    LANGUAGES
    English
    Urdu
    Hindi
    I am a versatile data scientist with expertise in a wide
    range of projects, including machine learning,
    recommendation systems, deep learning, and computer
    vision. Throughout my career, I have successfully
    developed and deployed various machine learning models
    to solve complex problems and drive data-driven
    decision-making
    Machine Learnine
    Deep Learning
    Computer Vision
    Recommendation Systems
    Data Visualization
    Programming Languages (Python, SQL)
    Data Preprocessing and Feature Engineering
    Model Evaluation and Deployment
    Statistical Analysis
    Communication and Collaboration
    """

    with open('tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('clf.pkl', 'rb') as f:
        clf = pickle.load(f)

    cleaned_resume = cleanResume(myresume)

    input_features = tfidf.transform([cleaned_resume])

    prediction_id = clf.predict(input_features)[0]
    print(f'Predicted Category ID: {prediction_id}')

if __name__ == "__main__":
    main()
