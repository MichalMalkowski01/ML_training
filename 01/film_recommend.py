import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)


def load_rating_data(data_path: str, n_users: int, n_movies: int):
    """Loads MovieLens ratings into a user-movie matrix."""
    data = np.zeros((n_users, n_movies), dtype=np.float32)
    movie_id_mapping = {}
    movie_rating_count = defaultdict(int)

    with open(data_path, "r", encoding="latin-1") as file:
        next(file)

        for line in file:
            user_id, movie_id, rating, _ = line.strip().split("::")

            user_index = int(user_id) - 1

            if movie_id not in movie_id_mapping:
                movie_id_mapping[movie_id] = len(movie_id_mapping)

            movie_index = movie_id_mapping[movie_id]
            rating = int(rating)

            data[user_index, movie_index] = rating
            movie_rating_count[movie_id] += 1

    return data, movie_rating_count, movie_id_mapping


def display_distribution(data: np.ndarray) -> None:
    values, counts = np.unique(data, return_counts=True)

    for value, count in zip(values, counts):
        print(f"Rating: {value}, Count: {count}")


def plot_roc_curve(y_test, prediction_proba) -> float:
    positive_probability = prediction_proba[:, 1]

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test,
        positive_probability,
    )

    auc_score = roc_auc_score(y_test, positive_probability)

    plt.figure(figsize=(8, 6))
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=2,
        label=f"ROC curve, AUC = {auc_score:.4f}",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=2)

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=300)
    plt.show()

    return auc_score


def evaluate_model(y_test, prediction) -> None:
    print("Confusion matrix:")
    print(confusion_matrix(y_test, prediction, labels=[0, 1]))

    precision = precision_score(y_test, prediction, pos_label=1)
    recall = recall_score(y_test, prediction, pos_label=1)
    f1 = f1_score(y_test, prediction, pos_label=1)

    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:  {f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            prediction,
            target_names=["Not Recommend", "Recommend"],
        )
    )


def prepare_dataset(data, movie_rating_count, movie_id_mapping, recommend_threshold=3):
    most_rated_movie_id, rating_count = max(
        movie_rating_count.items(),
        key=lambda item: item[1],
    )

    target_movie_index = movie_id_mapping[most_rated_movie_id]

    x_raw = np.delete(data, target_movie_index, axis=1)
    y_raw = data[:, target_movie_index]

    rated_mask = y_raw > 0

    x = x_raw[rated_mask]
    y = y_raw[rated_mask].copy()

    y = np.where(y <= recommend_threshold, 0, 1)

    print(f"Most rated movie ID: {most_rated_movie_id}")
    print(f"Number of ratings: {rating_count}")
    print(f"X shape: {x.shape}")
    print(f"Y shape: {y.shape}")
    print(f"Positive samples: {(y == 1).sum()}")
    print(f"Negative samples: {(y == 0).sum()}")

    return x, y

def cross(X, y, model, n_splits=5):
    k_fold = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
    smoothing_factor_option = [1, 2, 3, 4, 5, 6]
    fit_prior_option = [True, False]
    auc_record = {}
    for train_indices, test_indices in k_fold.split(X, y):
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        for alpha in smoothing_factor_option:
            if alpha not in auc_record:
                auc_record[alpha] = {}
            for fit_prior in fit_prior_option:
                clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
                clf.fit(X_train, y_train)
                prediction_prob = clf.predict_proba(X_test)
                pos_prob = prediction_prob[:, 1]
                auc = roc_auc_score(y_test, pos_prob)
                auc_record[alpha][fit_prior] = auc + auc_record[alpha].get(fit_prior, 0)
    return auc_record
def main():
    data_path = "data/ratings.dat"
    n_users = 6040
    n_movies = 3900

    data, movie_rating_count, movie_id_mapping = load_rating_data(
        data_path,
        n_users,
        n_movies,
    )

    x, y = prepare_dataset(
        data,
        movie_rating_count,
        movie_id_mapping,
        recommend_threshold=3,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = MultinomialNB(alpha=1.0, fit_prior=True)
    model.fit(x_train, y_train)

    prediction = model.predict(x_test)
    prediction_proba = model.predict_proba(x_test)

    accuracy = model.score(x_test, y_test)

    print(f"\nAccuracy: {accuracy * 100:.2f}%\n")

    evaluate_model(y_test, prediction)

    auc_score = plot_roc_curve(y_test, prediction_proba)
    print(f"ROC AUC Score: {auc_score:.4f}")

    auc_record = cross(x, y, model)
    for smoothing, smoothing_record in auc_record.items():
        for fit_prior, auc in smoothing_record.items():
            print(f"Alpha: {smoothing}, Fit Prior: {fit_prior!s:<6}, Average AUC: {auc/5:.5f}")
            
    clf = MultinomialNB(alpha=5, fit_prior=True)
    clf.fit(x_train, y_train)
    pos_prob = clf.predict_proba(x_test)[:, 1]
    print(f"Best model AUC: {roc_auc_score(y_test, pos_prob):.4f}")
if __name__ == "__main__":
    main()