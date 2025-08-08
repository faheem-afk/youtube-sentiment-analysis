import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from utils import *
from sklearn.model_selection import train_test_split


def lemmas(word):
    tagged_pos = pos_tag([word])[0][1]
    if tagged_pos.startswith('N'):
        return 'n'
    if tagged_pos.startswith('J'):
        return 'a'
    if tagged_pos.startswith('V'):
        return 'v'
    if tagged_pos.startswith('R'):
        return 'r'
    else:
        return 'n'
    

def process_comment(comment):
    
    comment = comment.lower()
    comment = ' '.join(comment.split())
    comment = re.sub(r"\n", ' ', comment)
    comment = re.sub(r"[^A-za-z0-9\s!?]", '', comment)
    stops = set(stopwords.words('english')) - {'however', 'but', 'not', 'no', 'yet'}
    comment = ' '.join([word for word in comment.split() if word not in stops])
    pos_tags = list(map(lemmas, comment.split()))
    zipped_com_pos = zip(comment.split(), pos_tags)
    lemmatizer = WordNetLemmatizer()
    comment = ""
    for word, pos_ in list(zipped_com_pos):
        comment += lemmatizer.lemmatize(word, pos=pos_) + " "
    comment = comment.rstrip()
    return comment


def data_pre():
    
    current_dir_path = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(current_dir_path, '../../params.yaml')
    params = load_params(params_path=params_path)
    test_size = params['data_preprocessing']['test_size']
  
    df = load_data("data/raw/data.csv")
    # df['text'] = df['text'].apply(process_comment)
    
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
    
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    
    save_data(train_data, os.path.join(current_dir_path, "../../data/processed/train_data.csv"))
    save_data(test_data, os.path.join(current_dir_path, "../../data/processed/test_data.csv"))
    

if __name__ == "__main__":
    data_pre()

