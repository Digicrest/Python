# numpy - math
# scipy - math
# lightm - allows us to perform reccomendation algorithms

import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# fetch data and format it
data = fetch_movielens(min_rating = 4.0)

# print training and testing data
print(repr(data['train']))
print(repr(data['test']))

# create model
model = LightFM(loss='warp') 

# measures difference between prediction and desired output,.
# minimize over training to make more accurate


# Waited Approximate-Rank Pairwise (WARP)
# Create recommendations for each user by looking at existing user-rating pairs and predicting rankings for each, 
# Uses gradient descent algorithm to iteratively find the weights that improve prediction over time


# train model, fit takes data set, number of epochs to run for, and number of threads
model.fit(data['train'], epochs=30, num_threads=2)

# generate recommendation from our model
def sample_recommendation(model, data, user_ids):

    #number of users and items (movies) in training data
    n_users, n_items = data['train'].shape

    # generate recommendations for each user we input
    for user_id in user_ids:
        # movies they already like
        # we consider ratings 5 positive or 4 and below negative to make problem binary
        # store these positive ratings in 'compressed sparse row format' <SEE: CSRF.PNG>
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        # movies our model predicts they will like
        # arange() returns every number from 0 up to number of items
        scores = model.predict(user_id, np.arange(n_items))

        # rank them in order of most liked to least liked
        # argsort() returns the score indices in desc order due to negative sign
        top_items = data['item_labels'][np.arange(-scores)]

        # print out results
        print("User %s" % user_id)
        print(" Known Positives:")

        for x in known_positives[:3]:
            print("     %s" % x)
        
        print(" Recommended:")

        for x in top_items[:3]:
            print(" %s" % x)
        
    sample_recommendation(model, data, [3, 25, 450])
