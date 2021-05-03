import pandas as pd
import utils
from tensorflow.keras.utils import to_categorical

data_path = '../../digit-recognizer/'
train_path = data_path + 'train.csv'
test_path = data_path + 'test.csv'
valid_path = data_path + 'sample_submission.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
valid_df = pd.read_csv(valid_path)

train_df['img_array'] = train_df.iloc[:, 1:].apply(lambda x: utils.convert_2d(x), axis=1)
test_df['img_array'] = test_df.iloc[:, :].apply(lambda x: utils.convert_2d(x), axis=1)

img = train_df['img_array'].iloc[0]
label = train_df['label'].iloc[0]
# img, label = train_df[['img_array','label']].iloc[0]
img = test_df['img_array'].iloc[1]
label = y_test[1]
# utils.print_img(img, label)
utils.print_img(img, label)
# train_df.head()
# test_df.head()
# valid_df.head()

X = train_df.iloc[:,1:-1]
y = to_categorical(train_df['label'])

X_test = test_df.iloc[:,:-1]

idx = 8
img = test_df['img_array'].iloc[idx]
base_result = test_df['base'].iloc[idx]
cnn_result = test_df['cnn'].iloc[idx]

utils.model_compare(img, base_result,cnn_result )