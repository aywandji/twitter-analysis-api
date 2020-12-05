import json

from flask import Flask
from flask import request
import pandas as pd
from searchtweets import load_credentials

from utils import get_tweets, compute_sentiment, get_sentiment_model, get_topics_from_tweets

app = Flask(__name__)
# Load Twitter developer CREDENTIALS
search_tweets_args = load_credentials(filename="twitter_keys.yaml",
                                      yaml_key="search_tweets_v2",
                                      env_overwrite=False)
CACHE_DIR = "transformers_models/"
NLTK_DATA_PATH = "nltk_data"
# Get tensorflow model for sentiment classification (from HuggingFace)
model, tokenizer = get_sentiment_model(cache_dir=CACHE_DIR)


@app.route('/')
def hello_world():
    return 'Welcome to your Flask API'


@app.route('/sentiment')
def get_sentiment():
    """Function triggered when Flask API is requested

    Returns:
        str: JSON containing analysis results

    """
    # USER REQUEST PARAMETERS
    hashtag = request.args.get('hashtag', '')
    if hashtag == "":
        return "Please specify a non null hashtag"
    nb_days = request.args.get('nb_days', 7)
    nb_days = int(min(max(nb_days, 1), 7))
    nb_tweets = max(request.args.get('nb_tweets', nb_days * 10), nb_days)
    get_topic_words = request.args.get('get_topic_words', False)
    n_topics = request.args.get('n_topics', 1)
    n_words_per_topic = request.args.get('n_words_per_topic', 10)
    lda_passes = request.args.get('lda_passes', 4)
    return_tweets = request.args.get('return_tweets', False)
    language = request.args.get('language', "en")

    # TWITTER REQUEST PARAMETERS
    days_offsets = range(-nb_days + 1, 1)
    query_key_value = " -is:retweet -is:quote lang:" + language
    tweet_fields = "created_at,public_metrics,author_id"
    max_nb_tweets_per_day = nb_tweets // len(days_offsets)
    query_string = "#" + hashtag.strip() + query_key_value

    # COMPUTE RESULTS
    tweets = get_tweets(query_string, days_offsets, tweet_fields,
                        max_nb_tweets_per_day, nb_tweets, search_tweets_args)
    sentiments_df, cleaned_tweets_texts, filtered_tweets_df = compute_sentiment(
        tweets, model, tokenizer)

    if get_topic_words:
        top_topics = get_topics_from_tweets(NLTK_DATA_PATH, cleaned_tweets_texts, n_topics=n_topics,
                                            n_words_per_topic=n_words_per_topic, n_passes=lda_passes,
                                            force_download=False)

    if return_tweets:
        sentiments_tweets_df = pd.concat(
            (sentiments_df, filtered_tweets_df.reset_index(drop=True)), axis=1)

        results = {"sentiments_json": sentiments_tweets_df.to_json()}
    else:
        results = {"sentiments_json": sentiments_df.to_json()}

    if get_topic_words:
        results["top_topics"] = top_topics.to_json()

    return json.dumps(results)
