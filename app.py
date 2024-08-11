from flask import Flask, render_template, request
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from afinn import Afinn
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import os

app = Flask(__name__)

#Load the pre-trained sentiment analysis model
afinn = Afinn()

# TF-IDF vectorizer
vectorizer = TfidfVectorizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Read the uploaded CSV file into a DataFrame
            df = pd.read_csv(file)
            
            # Preprocess the text data
            df = preprocess_data(df)
            
            # Perform sentiment analysis
            df['sentiment'] = df['content'].apply(lambda x: 'Positive' if afinn.score(x) > 0 else ('Negative' if afinn.score(x) < 0 else 'Neutral'))
            
            # Return the result
            return render_template('result.html', data=df.to_dict())
        else:
            return 'No file uploaded'

def preprocess_data(df):
    # Tokenization
    df = df.dropna(subset=['content']).drop_duplicates()
    stop_words = stopwords.words('english')
    df['content'] = df['content'].apply(lambda x: [word.lower() for word in word_tokenize(x) if (word.isalpha() and word.lower() not in stop_words)])
    df['content'] = df['content'].apply(lambda x: ' '.join(x))
    
    # Stemming
    stemmer = PorterStemmer()
    df['content'] = df['content'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
    
    return df

@app.route('/sentiment_analysis')
def sentiment_analysis():
    # Load data
    df = pd.read_csv('/content/PUBG_V2.8.0_Cleaned.csv')
    df = df.loc[:, ['userName', 'content', 'score']]

    # Perform preprocessing
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = stopwords.words('english')
    df = df.dropna(subset=['content']).drop_duplicates()
    df['content'] = df['content'].apply(lambda x: [word.lower() for word in word_tokenize(x) if (word.isalpha() and word.lower() not in stop_words)])
    df['content'] = df['content'].apply(lambda x: ' '.join(x))
    stemmer = PorterStemmer()
    df['content'] = df['content'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

    # Perform sentiment analysis
    afinn = Afinn()
    df['sentiment'] = df['content'].apply(lambda x: 'Positive' if afinn.score(x) > 0 else ('Negative' if afinn.score(x) < 0 else 'Neutral'))

    # Generate word clouds
    wordcloud_neutral = generate_wordcloud(df[df['sentiment'] == 'Neutral']['content'])
    wordcloud_positive = generate_wordcloud(df[df['sentiment'] == 'Positive']['content'])
    wordcloud_negative = generate_wordcloud(df[df['sentiment'] == 'Negative']['content'])

    # Generate funnel chart
    funnel_chart = generate_funnel_chart(df)

    # Generate donut plots
    neutral_donut = generate_donut_plot(df[df['sentiment'] == 'Neutral']['content'])
    positive_donut = generate_donut_plot(df[df['sentiment'] == 'Positive']['content'])
    negative_donut = generate_donut_plot(df[df['sentiment'] == 'Negative']['content'])

    # Generate target distribution plot
    target_distribution = generate_target_distribution(df)

    # Render template with image URLs
    return render_template('result.html', 
                           neutral_wordcloud=wordcloud_neutral,
                           positive_wordcloud=wordcloud_positive,
                           negative_wordcloud=wordcloud_negative,
                           funnel_chart=funnel_chart,
                           neutral_donut=neutral_donut,
                           positive_donut=positive_donut,
                           negative_donut=negative_donut,
                           target_distribution=target_distribution)

def generate_wordcloud(text):
    wordcloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(' '.join(text))
    filename = 'wordcloud.png'
    wordcloud.to_file(filename)
    return filename

def generate_funnel_chart(df):
    # Generate funnel chart code here
    filename = 'funnel_chart.png'
    # Save funnel chart to file
    plt.savefig(filename)
    return filename

def generate_donut_plot(text):
    # Generate donut plot code here
    filename = 'donut_plot.png'
    # Save donut plot to file
    plt.savefig(filename)
    return filename

def generate_target_distribution(df):
    # Generate target distribution code here
    filename = 'target_distribution.png'
    # Save target distribution plot to file
    plt.savefig(filename)
    return filename

if __name__ == '__main__':
    app.run(debug=True)
