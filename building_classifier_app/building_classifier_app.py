import flask
import keras
import glob
import pickle
from skimage import io
from flask import request

from keras.models import Model, Input
from keras.layers import Dense, Lambda, Layer
from keras import backend as K
from keras import metrics
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

#---------- MODEL IN MEMORY ----------------#

# load image function


def loadImage(img_size, path):
  '''
  This function loads a jpg and returns a numpy array.
  '''
  imgSize = (img_size, img_size)
  img = cv2.imread(path)
  img = cv2.resize(img, imgSize)

  return np.array([img])


model = keras.models.load_model('/Users/kennyleung/_ds/metis/metisgh/proj5-kojak/building_classifier_app/model')
vae_model = Model(inputs=model.input, outputs=model.layers[312].output)

batch_size = 100
original_dim = 1024
latent_dim = 2
intermediate_dim = 256
epochs = 15
epsilon_std = 1.0

x = Input(shape=(original_dim,), name='input')
h = Dense(intermediate_dim, activation='relu', name='intermediate')(x)
z_mean = Dense(latent_dim, name='z-mean')(h)
z_log_var = Dense(latent_dim, name='z-variance')(h)


def sampling(args):
  z_mean, z_log_var = args
  epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                            stddev=epsilon_std)
  return z_mean + K.exp(z_log_var / 2) * epsilon


z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu', name='decoder_h')
decoder_mean = Dense(original_dim, activation='sigmoid', name='decoder_mean')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)


# Custom loss layer
class CustomVariationalLayer(Layer):
  def __init__(self, **kwargs):
    self.is_placeholder = True
    super(CustomVariationalLayer, self).__init__(**kwargs)

  def vae_loss(self, x, x_decoded_mean):
    xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(xent_loss + kl_loss)

  def call(self, inputs):
    x = inputs[0]
    x_decoded_mean = inputs[1]
    loss = self.vae_loss(x, x_decoded_mean)
    self.add_loss(loss, inputs=inputs)
    # We won't actually use the output.
    return x


y = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model(x, y)
vae.compile(optimizer='rmsprop', loss=None)

filename = 'weights-improvement-11-85.2492.hdf5'
vae.load_weights(filename)

encoder = Model(inputs=vae.input, outputs=vae.get_layer(vae.layers[2].name).output)

#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = flask.Flask(__name__)
# app._static_folder = '/Users/kennyleung/_ds/metis/metisgh/proj5-kojak/'
# Homepage


@app.route("/")
def viz_page():
  """
  Homepage: serve our visualization page, awesome.html
  """
  with open("index.html", 'r') as viz_file:
    return viz_file.read()

# Get an example and return it's score from the predictor model


@app.route("/score", methods=["POST"])
def score():
  """
  When A POST request with json data is made to this uri,
  Read the example from the json, predict probability and
  send it with a response
  """
  # Get decision score for our example that came with the request
  r = request.json
  img_size = 256
  image = io.imread(r)
  imgSize = img_size, img_size
  img = cv2.resize(image, imgSize).reshape(1, img_size, img_size, 3)

  styles = ['art_deco', 'art_nouveau', 'chinese', 'gothic', 'modernist',
            'neoclassicism', 'renaissance', 'romanesque', 'russian']

  prediction = model.predict(img)
  features = vae_model.predict(img)

  min_max_scaler = pickle.load(open("scaler.sav", "rb"))
  x_train = min_max_scaler.transform(features)

  features_2d = encoder.predict(x_train)

  out1 = pd.Series(features_2d[0][0]).to_json(orient='values')
  out2 = pd.Series(features_2d[0][1]).to_json(orient='values')

  df = pd.read_csv('static/d3_df.csv')
  df.loc[len(df)] = ['new_building', r, styles[np.argmax(prediction)],
                     features_2d[0][0], features_2d[0][1], styles[np.argmax(prediction)],
                     0, 0, 0, 0, 0, 0, 0, 0]

  df.to_csv('static/d3_df.csv', index=False)

  results = (styles[np.argmax(prediction)].title(), out1, out2, r)

  return flask.jsonify(results)

#--------- RUN WEB APP SERVER ------------#


# Start the app server on port 80
# (The default website port)
app.run(host='0.0.0.0')
app.run(debug=True)
