import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

def load_raiting_data(data_path, n_users, n_movies):
    
    data = np.zeros((n_users, n_movies), dtype=np.float32)
    movie_id_mapping = {}
    movie_n_rating = defaultdict(int)
    with open(data_path, 'r') as f:
        for line in f.readlines()[1:]:
            user_id, movie_id, rating, _ = line.split('::')
            user_id = int(user_id) - 1
            if movie_id not in movie_id_mapping:
                movie_id_mapping[movie_id] = len(movie_id_mapping)
            rating = int(rating)
            data[user_id, movie_id_mapping[movie_id]] = rating
            if rating > 0:
                movie_n_rating[movie_id] += 1
    return data, movie_n_rating, movie_id_mapping

def display_distribution(data):
    values, counts = np.unique(data, return_counts=True)
    for value, count in zip(values, counts):
        print(f"Rating: {value}, Count: {count}")
    

def main():
    data_path = 'data/ratings.dat'
    n_users = 6040
    n_movies = 3900
    
    data, movie_n_rating, movie_id_mapping = load_raiting_data(data_path, n_users, n_movies)
    #display_distribution(data)
    movie_id_most, n_rating_most = sorted(movie_n_rating.items(), key=lambda x: x[1], reverse=True)[0]
    #print(f"Most rated movie: {movie_id_most}, Number of ratings: {n_rating_most}")
    
    X_raw = np.delete(data, movie_id_mapping[movie_id_most], axis=1)
    Y_raw = data[:, movie_id_mapping[movie_id_most]]
    X = X_raw[Y_raw > 0]
    Y = Y_raw[Y_raw > 0]
    #print(f"Shape of X: {X.shape}, Shape of Y: {Y.shape}")
    #display_distribution(Y)
    
    recommend = 3
    Y[Y <= recommend] = 0
    Y[Y > recommend] = 1
    n_pos = (Y == 1).sum()
    n_neg = (Y == 0).sum()
    #print(f"Number of positive samples: {n_pos}, Number of negative samples: {n_neg}")
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    #print(len(Y_train), len(Y_test))
    clf = MultinomialNB(alpha=1, fit_prior=True)
    clf.fit(X_train, Y_train)
    prediction_proba = clf.predict_proba(X_test)
    #print("Prediction probabilities: ", prediction_proba[0:10])
    prediction = clf.predict(X_test)
    print("Predictions: ", prediction[0:10])
    accuracy = clf.score(X_test, Y_test)
    print(f"Accuracy: {accuracy*100:.1f}%")
    
if __name__ == "__main__":
    main()