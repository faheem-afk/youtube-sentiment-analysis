from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')
from utils import *

current_dir_path = os.path.dirname(os.path.abspath(__file__))
    
params = load_params(os.path.join(current_dir_path, "../../params.yaml"))

objective = params['model_training']['objective']
num_class = params['model_training']['num_class']
metric = params['model_training']['metric']
is_unbalance = params['model_training']['is_unbalance']
class_weight = params['model_training']['class_weight']
reg_alpha = params['model_training']['reg_alpha']
reg_lambda = params['model_training']['reg_lambda']
learning_rate = params['model_training']['learning_rate']
max_depth = params['model_training']['max_depth']
n_estimators = params['model_training']['n_estimators']


def model_training():
    best_model=LGBMClassifier(
        objective=objective,
        num_class=num_class,
        metric=metric,
        is_unbalance=is_unbalance,
        class_weight=class_weight,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators
        
    )

    train_vec = joblib.load(os.path.join(current_dir_path, '../../data/vectorized/train_vec.joblib'))
    train_x_y = load_data(os.path.join(current_dir_path, '../../data/processed/train_data.csv')).dropna()

    y_train = train_x_y['label']

    best_model.fit(train_vec, y_train)

    joblib.dump(best_model, os.path.join(current_dir_path, '../../model/model.joblib'))


if __name__ == "__main__":
    model_training()