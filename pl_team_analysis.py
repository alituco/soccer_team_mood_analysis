from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import os
import re
import json
import random
import requests
import numpy as np
import spacy
import tensorflow as tf

leagues = {
    "1": {
        "name": "Premier League",
        "teams": [
            "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
            "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham",
            "Liverpool", "Luton Town", "Manchester City", "Manchester United",
            "Newcastle United", "Nottingham Forest", "Sheffield United",
            "Tottenham Hotspur", "West Ham United", "Wolverhampton Wanderers"
        ]
    },
    "2": {
        "name": "La Liga",
        "teams": [
            "Real Madrid", "Barcelona", "Atletico Madrid", "Sevilla", "Valencia",
            "Villarreal", "Real Sociedad", "Real Betis", "Athletic Bilbao", "Celta Vigo"
        ]
    },
    "3": {
        "name": "Serie A",
        "teams": [
            "Juventus", "Inter Milan", "AC Milan", "Napoli", "AS Roma",
            "Lazio", "Atalanta", "Sassuolo", "Fiorentina", "Torino"
        ]
    },
    "4": {
        "name": "Bundesliga",
        "teams": [
            "Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen", 
            "Schalke 04", "Wolfsburg", "Eintracht Frankfurt", "Borussia Monchengladbach", 
            "Hertha Berlin", "Union Berlin"
        ]
    },
    "5": {
        "name": "Ligue 1",
        "teams": [
            "Paris Saint-Germain", "Lyon", "Marseille", "Monaco", "Lille",
            "Rennes", "Nice", "Saint-Etienne", "Montpellier", "Strasbourg"
        ]
    }
}

# Get Twitter Bearer Token from .env file
BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")

def fetch_tweets_for_team(team_name, num_tweets=10):
    if not BEARER_TOKEN:
        print("\nNo real bearer token found. Returning mock data instead.\n")
        return fetch_mock_data(num_samples=num_tweets, team_name=team_name)
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    search_url = "https://api.twitter.com/2/tweets/search/recent"
    query_params = {"query": f"{team_name} -is:retweet lang:en", "max_results": str(min(num_tweets, 10))}
    response = requests.get(search_url, headers=headers, params=query_params)
    if response.status_code == 200:
        data = response.json()
        if "data" in data:
            return [tweet_obj["text"] for tweet_obj in data["data"]]
        else:
            print(f"No tweets found for {team_name}.")
            return []
    else:
        print(f"Error fetching tweets for {team_name}: {response.text}")
        return []

def fetch_mock_data(num_samples=10, team_name="SomeTeam"):
    sample_texts = [
        f"Wow, {team_name} looks unstoppable these days!",
        f"{team_name} are so inconsistent; they win one day and lose badly the next.",
        f"{team_name} is so fun to watch with their young players.",
        f"{team_name} keep missing chances, absolutely horrible finishing.",
        f"{team_name} games are too slow, boring sometimes.",
        f"{team_name} unstoppable in the final third!",
        f"{team_name} can’t hold a lead, super inconsistent performance.",
        f"Love the pressing style of {team_name}, they are exciting!",
        f"{team_name} look dreadful; their defense is so bad.",
        f"Watching {team_name} sometimes puts me to sleep—boring soccer!"
    ]
    return [random.choice(sample_texts) for _ in range(num_samples)]

LABELS = ["dominant", "inconsistent", "exciting", "really bad", "boring"]

def mock_label_data(tweets):
    labeled_data = []
    for t in tweets:
        label = random.choice(LABELS)
        labeled_data.append((t, label))
    return labeled_data

nlp = spacy.load("en_core_web_sm")

def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop and len(token) > 1]

def build_vocabulary(dataset, vocab_size=2000):
    from collections import Counter
    word_counter = Counter()
    for text, _ in dataset:
        tokens = clean_and_tokenize(text)
        word_counter.update(tokens)
    most_common = word_counter.most_common(vocab_size - 2)
    vocab2int = {w: i+2 for i, (w, _) in enumerate(most_common)}
    vocab2int["<PAD>"] = 0
    vocab2int["<UNK>"] = 1
    return vocab2int

def encode_data(dataset, vocab2int, max_len=40):
    X_encoded, y_encoded = [], []
    label2int = {"dominant": 0, "inconsistent": 1, "exciting": 2, "really bad": 3, "boring": 4}
    for text, label in dataset:
        tokens = clean_and_tokenize(text)
        seq = [vocab2int.get(tok, 1) for tok in tokens]
        seq = seq[:max_len]
        seq += [0] * (max_len - len(seq))
        X_encoded.append(seq)
        y_encoded.append(label2int[label])
    return np.array(X_encoded), np.array(y_encoded)

def build_model(vocab_size, embedding_dim=64, max_len=40, num_classes=5):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=['accuracy'])
    return model

def main():
    print("Welcome! Please select a league from the top 5:")
    for key in sorted(leagues.keys()):
        print(f"{key}: {leagues[key]['name']}")
    league_choice = input("Enter a number (1-5): ").strip()
    while league_choice not in leagues:
        league_choice = input("Invalid choice. Please enter a number from 1 to 5: ").strip()
    selected_league = leagues[league_choice]
    print(f"\nGreat, you selected {selected_league['name']}.")
    teams = selected_league["teams"]
    print("\nNow, please choose a team:")
    for idx, team in enumerate(teams, 1):
        print(f"{idx}: {team}")
    team_choice = input(f"Enter a number (1-{len(teams)}): ").strip()
    valid_indices = [str(i) for i in range(1, len(teams)+1)]
    while team_choice not in valid_indices:
        team_choice = input(f"Invalid choice. Enter a number (1-{len(teams)}): ").strip()
    team_name = teams[int(team_choice)-1]
    print(f"\nAwesome! You selected {team_name}.")
    print(f"\nFetching tweets about {team_name}...\n")
    tweets = fetch_tweets_for_team(team_name, num_tweets=10)
    print(f"Fetched {len(tweets)} tweets.\n")
    if not tweets:
        print("No tweets found or token missing. Exiting now.")
        return
    dataset = mock_label_data(tweets)
    print("Building vocabulary...")
    vocab2int = build_vocabulary(dataset, vocab_size=2000)
    vocab_size = len(vocab2int)
    print(f"Vocabulary size: {vocab_size}\n")
    X, y = encode_data(dataset, vocab2int, max_len=40)
    if len(X) < 5:
        print("Not enough data to split. Training on all available data.\n")
        X_train, y_train = X, y
        X_test, y_test = X, y
    else:
        split = int(0.8 * len(X))
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]
    print("Building our model...\n")
    model = build_model(vocab_size=vocab_size, embedding_dim=64, max_len=40, num_classes=5)
    model.summary()
    print("\nTraining the model... This might take a moment.")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=4)
    if len(X_test) > 0:
        loss, acc = model.evaluate(X_test, y_test)
        print(f"\nTest Loss: {loss:.4f}, Test Accuracy: {acc:.4f}\n")
    label_map = {0: "dominant", 1: "inconsistent", 2: "exciting", 3: "really bad", 4: "boring"}
    print("Here are the predictions for each tweet:")
    for text, _ in dataset:
        tokens = clean_and_tokenize(text)
        seq = [vocab2int.get(t, 1) for t in tokens][:40]
        seq += [0] * (40 - len(seq))
        seq = np.array([seq])
        preds = model.predict(seq)[0]
        predicted_label = label_map[np.argmax(preds)]
        print(f"Tweet: {text}")
        print(f"Predicted category: {predicted_label} (softmax probabilities: {preds})\n")

if __name__ == "__main__":
    main()
