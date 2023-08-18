from flask import Flask, request, jsonify,render_template, Markup
from PIL import Image
import numpy as np
import tensorflow as tf
# from tensorflow.keras.utils import img_to_array,load_img
from tensorflow.keras.applications.resnet50 import preprocess_input
from utils.disease import disease_dic

from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
# import cv2

from tensorflow.keras.models import load_model
# import io
import os


app = Flask(__name__)
app.secret_key='flash message'

app.config['IMAGE_UPLOADS'] = 'C:/Users/user/Desktop/Flask/static/images'


@ app.route('/')
def home():
    title = 'Plantleafnet - Home'
    return render_template('index.html', title=title)
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    d={0:'aeroplane',
        1:'beach',
        2:'bench',
        3:'bike',
        4:'book',
        5:'bottle',
        6:'car',
        7:'cat',
        8:'chair',
        9:'dog',
        10:'flags',
        11:'fruits',
        12:'garbage bags',
        13:'hand bag',
        14:'leaf',
        15:'paper',
        16:'pens',
        17:'plastic bags',
        18:'sandals',
        19:'shoes',
        20:'sky',
        21:'stars',
        21:'table',
        22:'windows',
        23:'women'}
    class_mapping = {
        0: 'Apple___Apple_scab',
        1: 'Apple___Black_rot',
        2: 'Apple___Cedar_apple_rust',
        3: 'Apple___healthy',
        4: 'Blueberry___healthy',
        5: 'Cherry_(including_sour)___Powdery_mildew',
        6: 'Cherry_(including_sour)___healthy',
        7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        8: 'Corn_(maize)___Common_rust_',
        9: 'Corn_(maize)___Northern_Leaf_Blight',
        10: 'Corn_(maize)___healthy',
        11: 'Grape___Black_rot',
        12: 'Grape___Esca_(Black_Measles)',
        13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        14: 'Grape___healthy',
        15: 'Orange___Haunglongbing_(Citrus_greening)',
        16: 'Peach___Bacterial_spot',
        17: 'Peach___healthy',
        18: 'Pepper,_bell___Bacterial_spot',
        19: 'Pepper,_bell___healthy',
        20: 'Potato___Early_blight',
        21: 'Potato___Late_blight',
        22: 'Potato___healthy',
        23: 'Raspberry___healthy',
        24: 'Soybean___healthy',
        25: 'Squash___Powdery_mildew',
        26: 'Strawberry___Leaf_scorch',
        27: 'Strawberry___healthy',
        28: 'Tomato___Bacterial_spot',
        29: 'Tomato___Early_blight',
        30: 'Tomato___Late_blight',
        31: 'Tomato___Leaf_Mold',
        32: 'Tomato___Septoria_leaf_spot',
        33: 'Tomato___Spider_mites Two-spotted_spider_mite',
        34: 'Tomato___Target_Spot',
        35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        36: 'Tomato___Tomato_mosaic_virus',
        37: 'Tomato___healthy',
        38:'non_leaf'
    }

    if request.method== 'POST':
        image= (request.files['image'])
    
        filename= secure_filename(image.filename)

        basedir= os.path.abspath(os.path.dirname(__file__))
        image.save(os.path.join(basedir, app.config['IMAGE_UPLOADS'], filename))

        test_image= plt.imread("C:/Users/user/Desktop/Flask/static/images/" + filename)  

        test_image="C:/Users/user/Desktop/Flask/static/images/" + filename

        img = Image.open(image)  
      

    
        
        img_size=(100,100)
        predicted_class = predict_image(test_image,img_size)
        print("Predicted Class Index:", d[predicted_class])
       


      
        N=d[predicted_class]
        if N=="leaf":

            img = preprocess_image(img)

            model1 = tf.keras.models.load_model('C:/Users/user/Desktop/Flask/RESNET50_PLANT_DISEASE.h5')

            predictions = model1.predict(np.expand_dims(img, axis=0))

            class_label = np.argmax(predictions)
            

            class_label_str = class_mapping[class_label]
            class_label_str = Markup(str(disease_dic[class_label_str]))

            return render_template('result.html', class_label=class_label_str)

        else:
            return jsonify({'error': 'given image is not a leaf'})

def predict_image(image_file, img_size):
    img = tf.keras.preprocessing.image.load_img(image_file, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Predict the image class
    model = tf.keras.models.load_model('E:\Flask\leaf_nonLeaf.hdf5')
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction[0])
    return class_index
def preprocess_image(image):
    resized_image = image.resize((224, 224))
    i = np.array(resized_image)
    im = preprocess_input(i)
    return im

if __name__ == '__main__':
    app.run(debug=True)




  



    