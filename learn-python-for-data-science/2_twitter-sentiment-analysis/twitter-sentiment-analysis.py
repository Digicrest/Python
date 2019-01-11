# Take input text, split it into several words or sentences (tokenization)
# count the number of times each woird shows up once it is tokenzied, (bag of words model)
# look up sentiment value of each word from a sentiment lexicon (which has it all saved)
# then work out the sentiment value of sentence

import tweepy
from textblob import TextBlob

# ------------- AUTHORIZATION ------------------
# Not sure if any of these values can be shared safely, so I'm removing them just in case
consumer_key = "---"
consumer_secret = "---"

access_token = "---"
access_token_secret = "---"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
# ------------- END AUTHORIZATION ---------------

api = tweepy.API(auth)

public_tweets = api.search('Nintendo Switch')

for tweet in public_tweets:
    print("----------------")
    print(tweet.text)
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)
