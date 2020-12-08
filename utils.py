import os
import re

from searchtweets import gen_request_parameters, collect_results
import pandas as pd
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
import tensorflow as tf
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_tags, strip_punctuation, strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_numeric, remove_stopwords, strip_short
from gensim.models import LdaModel
from gensim import corpora
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import nltk


class SentimentModelInputTypeError(Exception):
    def __init__(self):
        super(SentimentModelInputTypeError, self)


def get_sentiment_model(model_name="distilbert-base-uncased-finetuned-sst-2-english",
                        cache_dir="transformers_models/"):
    if not (isinstance(model_name, str) and isinstance(cache_dir, str)):
        raise SentimentModelInputTypeError()

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir,
                                              local_files_only=False)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir,
                                                                 local_files_only=False)
    return model, tokenizer


def get_query(query_string, jjmin, jjmax, tweet_fields, nb_tweets=10):
    """Generate formatted query for Twitter v2 API

    Args:
        query_string (string): string used to build query
        jjmin (int): min day offset from current day 
        jjmax (int): max day offset from current day
        tweet_fields (string): fields required to query Tiwtter API
        nb_tweets (int, optional): Max number of tweets to return. Defaults to 10.

    Returns:
        dict: formatted query

    """

    current_date = pd.to_datetime('today')
    start_time = (current_date + pd.Timedelta(jjmin, "D")).strftime("%Y-%m-%d")
    end_time = (current_date + pd.Timedelta(jjmax, "D")).strftime("%Y-%m-%d")

    if jjmax == 1:  # Return end time as today - 1min
        end_time = (current_date - pd.Timedelta(1, "m")
                    ).strftime("%Y-%m-%dT%H:%M")

    query = gen_request_parameters(query_string,
                                   tweet_fields=tweet_fields,
                                   start_time=start_time,
                                   end_time=end_time,
                                   results_per_call=nb_tweets)
    return query


def get_tweets(query_string, days_offsets, tweet_fields,
               max_nb_tweets_per_day, total_nb_tweets, search_tweets_args):
    tweets = []
    remaining_number_of_tweets = 0

    # generate query and request tweets for each day offset.
    for i, day_offset in enumerate(days_offsets):
        max_tweets = max_nb_tweets_per_day + remaining_number_of_tweets
        if i == len(days_offsets) - 1:
            max_tweets = total_nb_tweets - len(tweets)
        query = get_query(query_string, day_offset, day_offset + 1, tweet_fields, nb_tweets=10)
        collected_tweets = collect_results(
            query, max_tweets=max_tweets, result_stream_args=search_tweets_args)[:-1]
        tweets.extend(collected_tweets)
        remaining_number_of_tweets = max_tweets - len(collected_tweets)

    return tweets


def get_sentiment_df(sentiment_model_output_logits, labels_dict):
    """generate sentiment probabilities from sentiment_model_output_logits using softmax function.

    Args:
        sentiment_model_output_logits (output_class): Output logits generated from model inference
        labels_dict (dict): dictionaary of sentiment index to label)

    Returns:
        pandas.DataFrame: Dataframe containing sentiment and corresponding probabilities

    """
    sentiment_dict = {}
    outputs_proba = tf.keras.activations.softmax(
        sentiment_model_output_logits)
    for idx, s_name in labels_dict.items():
        sentiment_dict[s_name] = outputs_proba[:, idx].numpy()

    return pd.DataFrame(sentiment_dict)


def tokenize_and_predict_sentiment(input_texts, tokenizer, model):
    encodings = tokenizer(input_texts, padding=True, return_tensors="tf")
    sentiment_model_output_tensors = model(encodings)

    return get_sentiment_df(sentiment_model_output_tensors.logits, model.config.id2label)


def de_emojify(text):
    """
    Remove Emojies from text using regex patterns.
    """
    # regrex_pattern = re.compile(pattern = "["
    #     u"\U0001F600-\U0001F64F"  # emoticons
    #     u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    #     u"\U0001F680-\U0001F6FF"  # transport & map symbols
    #     u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    #                        "]+", flags = re.UNICODE)
    # return regrex_pattern.sub(r'',text)
    return re.sub(r'[^\x00-\x7F]+', ' ', text)


def clean_tweet(tweet_text):
    # Remove urls
    tweet_text = re.sub(r"http\S+|www\S+|https\S+", '',
                        tweet_text, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    tweet_text = re.sub(r'\@|\#', '', tweet_text)
    # Remove \n from tweet
    tweet_text = re.sub(r'\n\n', '.', tweet_text)
    tweet_text = re.sub(r'\n', '.', tweet_text)
    # Remove emoji from tweet
    tweet_text = de_emojify(tweet_text)

    return tweet_text


def get_cleaned_texts(tweets):
    tweets_df = pd.DataFrame(tweets)
    tweets_df = tweets_df.loc[tweets_df.created_at.notna()]
    cleaned_texts = tweets_df.text.map(clean_tweet)

    return list(cleaned_texts), tweets_df


def compute_sentiment(tweets, model, tokenizer):
    tweets_texts, filtered_tweets_df = get_cleaned_texts(tweets)
    sentiment_df = tokenize_and_predict_sentiment(
        tweets_texts, tokenizer, model)

    return sentiment_df, tweets_texts, filtered_tweets_df


def get_wordnet_pos(word):
    """
    Map POS tag to first character NLTK WordNetLemmatizer.lemmatize() accepts
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_sentences(list_of_sentences):
    results = []
    lemmatizer = WordNetLemmatizer()
    for list_of_tokens in list_of_sentences:
        results.append([lemmatizer.lemmatize(w, get_wordnet_pos(w))
                        for w in list_of_tokens])

    return results


def get_topics_from_tweets(nltk_data_path, cleaned_tweets_texts,
                           n_topics=1, n_words_per_topic=10, n_passes=2, force_download=False):
    """Retrieves topics from cleaned_tweets_texts using LDA algorithm.

    Args:
        nltk_data_path (str): path to NLTK data
        cleaned_tweets_texts (list of str): List of tweets texts from which topics are learned
        n_topics (int, optional): Number of topics to learn. Defaults to 1.
        n_words_per_topic (int, optional): Number of words per topic. Defaults to 10.
        n_passes (int, optional): Passes to train LDA. See LDA doc for more details. Defaults to 2.
        force_download (bool, optional): If True, NLTK data will be downloaded. Defaults to False.

    Returns:
        pandas.DataFrame: DataFrame containing topic words and their probabilities (weights)

    """
    # Check NLTK data
    nltk_data_available = os.path.isdir(
        nltk_data_path) and len(os.listdir(nltk_data_path)) != 0

    if force_download or not nltk_data_available:
        nltk.download('wordnet', download_dir=nltk_data_path)
        nltk.download('averaged_perceptron_tagger',
                      download_dir=nltk_data_path)
    else:
        nltk.data.path.append(nltk_data_path)

    # Get dictionary from downloaded tweets
    custom_filters = [lambda x: x.lower(), strip_tags, strip_punctuation,
                      strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short]
    preprocessed_tweets = [preprocess_string(
        d, custom_filters) for d in cleaned_tweets_texts]
    preprocessed_tweets = lemmatize_sentences(preprocessed_tweets)
    dictionary = corpora.Dictionary(preprocessed_tweets)

    # Train LDA to reveal topics
    vectorized_tweets = [dictionary.doc2bow(
        p_tweet) for p_tweet in preprocessed_tweets]
    lda = LdaModel(vectorized_tweets, num_topics=n_topics,
                   passes=n_passes, id2word=dictionary,
                   alpha="auto", eta="auto")
    top_topics = lda.top_topics(vectorized_tweets, topn=n_words_per_topic)

    # Convert top_topics to dataframe
    top_topics_words = []
    top_topics_proba = []
    top_topics_indexes = []
    for i, topic in enumerate(top_topics):
        topic_words = [w_tuple[1] for w_tuple in topic[0]]
        topic_proba = [w_tuple[0] for w_tuple in topic[0]]
        top_topics_indexes.extend([i] * len(topic[0]))
        top_topics_words.extend(topic_words)
        top_topics_proba.extend(topic_proba)

    top_topics_df = pd.DataFrame(
        {"words": top_topics_words, "topics": top_topics_indexes, "proba": top_topics_proba})

    return top_topics_df
