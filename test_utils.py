from unittest.mock import patch
import json

import pytest
from searchtweets import collect_results
import numpy as np
import tensorflow as tf


from utils import get_sentiment_model, SentimentModelInputTypeError
from utils import get_query, get_tweets, get_sentiment_df
from utils import clean_tweet


@pytest.mark.parametrize(
    ("model_name", "cache_dir"), [(22, "cache_dir"), ("name", 3)]
)
def test_get_sentiment_model_bad_input_type(model_name, cache_dir):
    with pytest.raises(SentimentModelInputTypeError):
        model, tokenizer = get_sentiment_model(model_name, cache_dir)


def test_get_query():
    query_string = "#december -is:retweet -is:quote lang:en"
    nb_tweets = 10
    jjmin = -2
    jjmax = 1
    tweet_fields = "created_at,public_metrics,author_id"
    twitter_expected_query = {"query": query_string,
                              "max_results": nb_tweets,
                              # "start_time": "2020-12-01T00:00:00Z",
                              # "end_time": "2020-12-07T09:24:00Z",
                              "tweet.fields": tweet_fields}
    generated_query = json.loads(get_query(query_string, jjmin, jjmax,
                                           tweet_fields, nb_tweets=nb_tweets))
    generated_fields = list(generated_query.keys())
    for query_field in twitter_expected_query:
        assert twitter_expected_query[query_field] == generated_query[query_field]
    # test start_time and end_time fields (we should also test their values)
    # assertIn("start_time", generated_fields)
    # assertIn("end_time", generated_fields)
    assert "start_time" in generated_fields
    assert "end_time" in generated_fields


@patch('utils.collect_results')
def test_get_tweets_total_nb_tweets(collect_results_mock):
    # collect_results_mock mocks collect_results from twitter api
    expected_tweets = [[{"text": "hello"},{}], [{"text": "hi"},{}]]
    collect_results_mock.side_effect = expected_tweets
    expected_tweets = [et[0] for et in expected_tweets]
    total_nb_tweets = 10
    query_string = "#december -is:retweet -is:quote lang:en"
    days_offsets = range(-1, 1)
    tweet_fields = " "
    max_nb_tweets_per_day = 2
    search_tweets_args = {}
    returned_tweets = get_tweets(query_string, days_offsets, tweet_fields,
                                 max_nb_tweets_per_day, total_nb_tweets,
                                 search_tweets_args)
    assert len(returned_tweets) <= total_nb_tweets
    assert returned_tweets == expected_tweets
    # collect_results_mock.assert_called_with(search_tweets_args)


def test_get_sentiment_df():
    # we can also check that dataframe values are within 0 and 1
    sentiment_model_output_logits = tf.ones((5,2))
    labels_dict = {0:"negative",1:"positive"}
    returned_df = get_sentiment_df(sentiment_model_output_logits, 
                                    labels_dict)
    
    assert returned_df.shape == (sentiment_model_output_logits.shape[0],len(labels_dict))
    assert returned_df.values.max() <= 1
    assert returned_df.values.min() >= 0

def test_clean_tweet():
    # test that : urls, @, #, \n and emojies are removed
    tweet = "a@b #c \n\n d \n e  https://d.fr f http://di.fr g"
    expected_cleaned_tweet = "ab c . d . e   f  g"

    cleaned_tweet = clean_tweet(tweet)
    assert expected_cleaned_tweet == cleaned_tweet
