from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer

chatbot = ChatBot(
    'ChatBoy', 
    logic_adapters=[
        {
            'import_path': 'chatterbot.logic.MathematicalEvaluation',
            'default_response': "Sorry, I don't understand you. I'll try to be better!",
        },
        {
            'import_path': 'chatterbot.logic.BestMatch',
            'default_response': "Sorry, I don't understand you. I'll try to be better!",
            'maximum_similarity_threshold': 0.90
        },
        {
            'import_path': 'chatterbot.logic.TimeLogicAdapter',
            'input_text': "What's the time?",
            'default_response': "Sorry, I don't understand you. I'll try to be better!",
            'maximum_similarity_treshold': 1
        }
    ],
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    database_uri='sqlite:///database.sqlite3'

)

# Training cu propriile intrebari
trainer = ListTrainer(chatbot)
training_data_quesans = open('training_data/ques_ans.txt').read().splitlines()
training_data_personal = open('training_data/personal_ques.txt').read().splitlines()
training_data = training_data_quesans + training_data_personal
trainer.train(training_data)

# Training cu corpus

trainer_corpus = ChatterBotCorpusTrainer(chatbot)
# Antenam robotul sa salute
trainer_corpus.train("chatterbot.corpus.english.greetings")
# Antrenam robotul sa aiba conversatii simple
trainer_corpus.train("chatterbot.corpus.english.conversations")
# Antrenam robotul dupa english corpus + greetings in alte limbi
trainer_corpus.train("chatterbot.corpus.english")
trainer_corpus.train("chatterbot.corpus.french.greetings")
trainer_corpus.train("chatterbot.corpus.italian.greetings")
trainer_corpus.train("chatterbot.corpus.spanish.greetings")
trainer_corpus.train("chatterbot.corpus.swedish.greetings")
