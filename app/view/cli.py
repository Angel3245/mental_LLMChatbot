from clean_data import clean

def ask_chatbot(text_generator):
    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break

        # Clean input sentence
        user_input = clean(user_input)

        # Analyze input and create predefined responses if needed
        # Check if user_input words are higher than 3
        if(len(user_input.split()) < 3):
            response = "Sorry, I did not understand you. Could you explain it better?"
        else:
            response = text_generator.generate_response(user_input)
        
        print(f"Chatbot: {response}")