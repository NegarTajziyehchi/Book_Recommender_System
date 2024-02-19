import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_recommenders as tfrs

df = pd.read_csv('/home/ubuntu/recommender_system/data/processed/clean_data.csv')
for col in df.columns:
    if col not in ['rating','Age']:
        df[col] = df[col].astype(str)
    else:
        df[col] = df[col].astype(int)

# converting df to dictionary
df_dict = {name: np.array(val) for name, val in df.items()}

# converting dictionary to tensor slices
data = tf.data.Dataset.from_tensor_slices(df_dict)

# getting a dictionary of unique values in our features

vocabularies = {}

for feature in df_dict:
    if feature != 'rating':
        vocab = np.unique(df_dict[feature])
        vocabularies[feature] = vocab


# converting book-title to a tensorflow dataset
book_titles = tf.data.Dataset.from_tensor_slices(vocabularies['Book-Title'])

def return_book_titles():
    return book_titles

book_authors = df['Book-Author'].unique()
user_age = df['Age'].values

# shuffling and splitting our dataset into train, validation and test
tf.random.set_seed(42)

shuffled = data.shuffle(100_000, seed=42, reshuffle_each_iteration=False)


train = shuffled.take(46_797)
validation = shuffled.skip(46_797).take(9_359)
test = shuffled.skip(56_156).take(6_240)

class UserModel(tf.keras.Model):
  
    def __init__(self):
        super().__init__()
        
        max_tokens = 10_000
        
        # 1. User ID
        self.user_id_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=vocabularies['user'],
                mask_token=None),
            tf.keras.layers.Embedding(len(vocabularies['user'])+1, 32)
        ])
             
        
        #2. Book Authors
        self.author_vectorizer = keras.layers.TextVectorization(max_tokens=max_tokens)
        self.author_vectorizer.adapt(book_authors)
        self.author_text_embedding = keras.Sequential([
            self.author_vectorizer,
            keras.layers.Embedding(max_tokens, 32, mask_zero=True),
            keras.layers.GlobalAveragePooling1D()
        ])
        
        self.author_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=vocabularies['Book-Author'],
                mask_token=None),
            tf.keras.layers.Embedding(len(vocabularies['Book-Author'])+1, 32)
        ])
         
        
        # 3. User age
        self.normalized_age = keras.layers.Normalization()
        self.normalized_age.adapt(vocabularies['Age'].reshape(-1,1))
        
    # call method passes out input features to the embeddings above, excutes them and returns the output
    def call(self, inputs):
        
        return tf.concat([
            self.user_id_embedding(inputs['user']),
            self.author_embedding(inputs['Book-Author']),
            self.author_text_embedding(inputs['Book-Author']),
            tf.reshape(self.normalized_age(inputs['Age']), (-1,1))
        ], axis=1) 


class TitleModel(tf.keras.Model):
    
    def __init__(self,):
        super().__init__()
        
        max_tokens = 10_000
        
        #1. Book-Titles
        self.book_vectorizer = keras.layers.TextVectorization(max_tokens=max_tokens)
        self.book_vectorizer.adapt(book_titles)
        self.book_text_embedding = keras.Sequential([
            self.book_vectorizer,
            keras.layers.Embedding(max_tokens, 32, mask_zero=True),
            keras.layers.GlobalAveragePooling1D()
        ])
        
        self.book_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=vocabularies['Book-Title'],
                mask_token=None),
            tf.keras.layers.Embedding(len(vocabularies['Book-Title'])+1, 32)
        ])
        
        
    # call method passes category to the embedding layer above, executes it and returns the output embeddings
    def call(self, inputs):
        
        return tf.concat([
            self.book_embedding(inputs),
            self.book_text_embedding(inputs),
        ], axis=1)
    

tf.random.set_seed(7)
np.random.seed(7)


class FullModel(tfrs.models.Model):
    
    def __init__(self,):
        super().__init__()
        
        # handles how much weight we want to assign to the rating and retrieval task when computing loss
        self.rating_weight = 0.5
        self.retrieval_weight = 0.5
        
        #User model
        self.user_model = tf.keras.Sequential([
            UserModel(),
            tf.keras.layers.Dense(32),
        ])
        
        # Category model
        self.title_model = tf.keras.Sequential([
            TitleModel(),
            tf.keras.layers.Dense(32)
        ])
        
        
        # Deep & Cross layer
        self._cross_layer = tfrs.layers.dcn.Cross(projection_dim=None, kernel_initializer='he_normal')
        
        # Dense layers with l2 regularization to prevent overfitting
        self._deep_layers = [
            keras.layers.Dense(512, activation='relu', kernel_regularizer='l2'),
            keras.layers.Dense(256, activation='relu', kernel_regularizer='l2'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation='relu', kernel_regularizer='l2'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu', kernel_regularizer='l2'),
            keras.layers.Dense(32, activation='relu', kernel_regularizer='l2'),
        ]
        
        # output layer
        self._logit_layer = keras.layers.Dense(1)
    
        # Multi-task Retrieval & Ranking
        self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
        self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=book_titles.batch(128).map(self.title_model)
            )
        )
       
            
    def call(self, features) -> tf.Tensor:
        user_embeddings = self.user_model({
            'user': features['user'],
            'Book-Author': features['Book-Author'],
            'Age': features['Age'],
        })
        
        
        title_embeddings = self.title_model(
            features['Book-Title']
        )
        
        x = self._cross_layer(tf.concat([
                user_embeddings,
                title_embeddings], axis=1))
        
        for layer in self._deep_layers.layers:
            x = layer(x)
            
        
        return (
            user_embeddings, 
            title_embeddings,
            self._logit_layer(x)
        )
        
        
        

    def compute_loss(self, features, training=False) -> tf.Tensor:
        user_embeddings, title_embeddings, rating_predictions = self.call(features)
        # Retrieval loss
        retrieval_loss = self.retrieval_task(user_embeddings, title_embeddings)
        # Rating loss
        rating_loss = self.rating_task(
            labels=features['rating'],
            predictions=rating_predictions
        )
        
        # Combine two losses with hyper-parameters (to be tuned)
        return (self.rating_weight * rating_loss + self.retrieval_weight * retrieval_loss)
    

def build_model():
    """Instantiates a model and compiles it."""
    keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    # instantiating the model
    model = FullModel()
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

    return model
