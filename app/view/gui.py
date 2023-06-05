from InquirerPy import inquirer
import pandas as pd
from faq.Mental_Health_FAQ import *
from clean_dataset import *
from transformation import filter_irrelevant_posts

# Disable log messages
import logging
logging.disable(logging.ERROR)

def ask_sentence(reddit_posts):
    # Ask user to select between a predefined post or a custom question 
    mode = inquirer.select(
        message="Select mode:",
        choices=["Introduce a question", "Select a Reddit post"],
    ).execute()

    # Get question
    if(mode == "Introduce a question"):
        # Keyboard input
        question = inquirer.text(message="Introduce sentence:").execute()
    elif(mode == "Select a Reddit post"):
        posts = pd.read_csv(reddit_posts)

        # filter posts
        posts = filter_irrelevant_posts(posts)

        # select 3 random posts
        question = inquirer.select(
            message="Recommended posts:",
            choices=posts['title'].sample(n=3),
        ).execute()
        
    # Clean input question
    question = preprocess(question)
    
    # Define model parameters
    top_k = 1
    dataset = 'MentalFAQ'
    model = 'publichealthsurveillance/PHS-BERT'
    fields = ['question_answer']
    model_path = dataset+"/models/"+model

    index_name = "mentalfaq"
    
    loss_type = 'softmax'; neg_type = 'simple'; query_type = 'faq'

    answer = ranker(top_k, model_path, fields, index_name, loss_type, neg_type, query_type, question)

    if(answer):
        print("Answer: "+confidence_estimator(answer[0]["answer"],answer[0]["score"]))
    else:
        print("Error: Answer was not retrieved")

    print("Total score: "+str(answer[0]["score"]))