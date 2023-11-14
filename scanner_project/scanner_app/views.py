from django.shortcuts import render

from apify_client import ApifyClient

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax


def api(word):
    # Initialize the ApifyClient with your API token
    client = ApifyClient("apify_api_akBxmyWY9jgxw3MCJIkVFTyvmRw61w10LSca")

    # Prepare the Actor input
    run_input = {
        "absolute_max_tweets": 2,
        "filter:blue_verified": False,
        "filter:has_engagement": False,
        "filter:images": False,
        "filter:media": False,
        "filter:nativeretweets": False,
        "filter:quote": False,
        "filter:replies": False,
        "filter:retweets": False,
        "filter:safe": False,
        "filter:twimg": False,
        "filter:verified": False,
        "filter:videos": False,
        "language": "en",
        "max_attempts": 1,
        "max_tweets": 2,
        "only_tweets": True,
        "queries": [
            word
        ],
        "use_experimental_scraper": False
    }

    # Run the Actor and wait for it to finish
    run = client.actor("wHMoznVs94gOcxcZl").call(run_input=run_input)

    # Fetch and print Actor results from the run's dataset (if there are any)
    tweet = client.dataset(run["defaultDatasetId"])
    # tweet={'tweet_avatar': 'https://pbs.twimg.com/profile_images/1681069131557220354/nmqnizNC_bigger_400x400.jpg', 'tweet_id': '1713959878312566798', 'url': 'https://twitter.com/Cryptooisseur/status/1713959878312566798', 'query': 'dumb lang:en -filter:retweets -filter:replies -filter:quote', 'text': "Crypto is dumb but being dumb isn't illegal", 'username': '@Cryptooisseur', 'fullname': 'Pleb #1 🛡', 'timestamp': '2023-10-16 16:47:00+00:00', 'language': None, 'in_reply_to': [], 'replies': 0, 'retweets': 0, 'quotes':  
    # 0, 'images': [], 'likes': 0, 'banner_image': None, 'total_tweets': None, 'num_following': None, 'num_followers': None, 'total_likes': None, 'tweet_links': [], 'tweet_hashtags': [], 'tweet_mentions': []}
    # Crypto is dumb but being dumb isn't illegal
    # {'tweet_avatar': 'https://pbs.twimg.com/profile_images/1697292081171931136/3j9_NusG_bigger_400x400.jpg', 'tweet_id': '1713959837602660654', 'url': 'https://twitter.com/sugarcoralx/status/1713959837602660654', 'query': 'dumb lang:en -filter:retweets -filter:replies -filter:quote', 'text': 'Eva… you dumb bitch go to wishtender now', 'username': '@sugarcoralx', 'fullname': 'Goddess Sugar', 'timestamp': '2023-10-16 16:47:00+00:00', 'language': None, 'in_reply_to': [], 'replies': 0, 'retweets': 0, 'quotes': 0, 'images': [], 'likes': 0, 'banner_image': None, 'total_tweets': None, 'num_following': None, 'num_followers': None, 'total_likes': None, 'tweet_links': [], 'tweet_hashtags': [], 'tweet_mentions': []}
    # print("This is tweet---->    ",tweet)
        
    tweet_list=[]       #List of Tweets
    for item in tweet.iterate_items():
        print(item['text'])
        tweet_list.append(item['text'])
    return (tweet_list)

def anal(tweet):
    # precprcess tweet
    tweet_words = []

    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)

    tweet_proc = " ".join(tweet_words)

    # load model and tokenizer
    roberta = "cardiffnlp/twitter-roberta-base-sentiment"

    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)

    
    # sentiment analysis
    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
    # output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
    output = model(**encoded_tweet)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    

    # print(scores.tolist())
    score_l = scores.tolist()
    return(score_l)

def score_giver(tweet_list):
    final_score = [0, 0, 0]
    for i in tweet_list:
        anal_score=anal(i)
        final_score=[final_score[0]+anal_score[0], final_score[1]+anal_score[1], final_score[2]+anal_score[2]]
    final1_score = [final_score[0]/len(tweet_list), final_score[1]/len(tweet_list),final_score[2]/len(tweet_list)]
    # print(final1_score)
    labels = ['Negative', 'Neutral', 'Positive']

    for i in range(len(final1_score)):
            
            l = labels[i]
            s = final1_score[i]
            print(l,s)
    return {'output':final1_score}
    


def Negative(request):
    return render(request, 'Negative.html')
def Neutral(request):
    return render(request, 'Neutral.html')
def Positive(request):
    return render(request, 'Positive.html')

def home(request):
    if request.method == "POST":
        word1 = request.POST
        print(word1)
        word = word1.get('word')
        # print(type(word))  # To print the inputed word
        call_api = api(word)
        data = {'a': score_giver(call_api)}
        #print(data,type(data))
        Score_list=(data['a']['output'])
        Score_list = [score * 10 for score in Score_list]

        keys = ["Negative", "Neutral", "Positive"]
        score_dict = dict(zip(keys, Score_list))        #this will return dictionary of sentiment score
        print(score_dict)
        max_key = max(score_dict, key=score_dict.get)   #this will return Sentiment score
        print(max_key)

        if max_key == 'Negative':
            return render(request, 'Negative.html', data)
        if max_key == 'Neutral':
            return render(request, 'Neutral.html', data)
        if max_key == 'Positive':
            return render(request, 'Positive.html', data)
              
        return render(request, 'home.html', data)
    else:
        # Handle other cases if needed
        return render(request, 'home.html')




