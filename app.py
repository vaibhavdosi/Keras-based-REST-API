from flask import Flask,request,jsonify

from sklearn.externals import joblib #for model save ,not required here.
import pandas as pd

from keras.models import model_from_json
import tensorflow as tf

app=Flask(__name__)
#CORS(app)

headers=['Open','High','Low','Volume']

#Model loading
json_model=open('abc.json','r')
load_model=json_model.read()
json_model.close()
loaded_model=model_from_json(load_model)

#load saved weights 
loaded_model.load_weights('wt.h5')

graph = tf.get_default_graph()

#Compile before making prediction
loaded_model.compile(optimizer = 'adam', loss = 'MSE', metrics = ['accuracy'])

#make url
@app.route('/predict',methods=['POST'])

#associate funtion with URL
def predict():
	payload=request.json['Level']
	inp=[float(i) for i in payload.split(',')]
	inpu=pd.DataFrame([inp],columns=headers,dtype=float,index=['input'])
	with graph.as_default():
		pre=loaded_model.predict(inpu)
	ret='{" Stock Prediction":'+str(pre) +'}'
	return ret

if __name__ == '__main__':
    app.run()
   
