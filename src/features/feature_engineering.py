from sklearn.feature_extraction.text import TfidfVectorizer
from utils import *
from ast import literal_eval


def feature_eng():

    current_dir_path = os.path.dirname(os.path.abspath(__file__))
    df_train = load_data("data/processed/train_data.csv").dropna()
    df_test = load_data("data/processed/test_data.csv").dropna()
    params = load_params(os.path.join(current_dir_path, "../../params.yaml"))
    ngram_range = params['feature_engineering']['ngram_range']
    max_features = params['feature_engineering']['max_features']
    
    max_features = max_features
    ngram_range=ngram_range
    vectorizer = TfidfVectorizer(ngram_range=literal_eval(ngram_range), max_features=max_features)

    train_vec = vectorizer.fit_transform(df_train['text']).toarray()
    test_vec = vectorizer.transform(df_test['text']).toarray()
    
    save_model(vectorizer, 'model/vectorizer.joblib')
    
    dir_name = 'data/vectorized'
    save_data(train_vec, os.path.join(current_dir_path, f"../../{dir_name}/train_vec.joblib"))
    save_data(test_vec, os.path.join(current_dir_path, f"../../{dir_name}/test_vec.joblib"))


if __name__ == "__main__":
    feature_eng()

