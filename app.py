from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import cv2
import os
import base64
from io import BytesIO
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Sequential, model_from_json
from tensorflow.keras.utils import to_categorical
from PIL import Image
import matplotlib.pyplot as plt
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
index_by_directory = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    '+': 10, '-': 11, 'x': 12, 'a': 13, 'b': 14
}

def solveEquation(equation, variable):
    n = len(equation)
    sign = 1
    coeff = 0
    total = 0
    i = 0

    for j in range(0, n):
        if equation[j] in ['+', '-']:
            if j > i:
                total = total + sign * float(equation[i:j])
            i = j
        elif equation[j] == variable:
            if i == j or equation[j - 1] == '+':
                coeff += sign
            elif equation[j - 1] == '-':
                coeff = coeff - sign
            else:
                coeff = coeff + sign * float(equation[i:j])
            i = j + 1
        elif equation[j] == '=':
            if j > i:
                total = total + sign * float(equation[i:j])
            sign = -1
            i = j + 1

    if i < n:
        total = total + sign * float(equation[i:])

    if coeff == 0 and total == 0:
        return "Infinite solutions"
    if coeff == 0 and total:
        return "No solution"

    return float(-total / coeff)

class ConvolutionalNeuralNetwork:
    def __init__(self):
        if os.path.exists('model/model_weights.h5') and os.path.exists('model/model.json'):
            self.load_model()
        else:
            raise FileNotFoundError("Model files not found!")

    def load_model(self):
        print('Loading Model...')
        #
        model_json = open('model/model.json', 'r')
        loaded_model_json = model_json.read()
        model_json.close()
        loaded_model = model_from_json(loaded_model_json)
        #
        print('Loading weights...')
        #
        loaded_model.load_weights("model/model_weights.h5")
        self.model = loaded_model
        #
        print("Model loaded successfully!")

    def predict(self, operationBytes):
        Image.open(operationBytes).save('_aux_.png')
        #
        img = cv2.imread('_aux_.png', 0)
        #
        os.remove('_aux_.png')
        if img is not None:
            img_data = extract_imgs(img)
            operation = ''
            for i in range(len(img_data)):
                img_data[i] = np.array(img_data[i])
                img_data[i] = img_data[i].reshape(-1, 28, 28, 1)

                pred = self.model.predict(img_data[i])
                result = np.argmax(pred, axis=1)
                if result[0] == 10:
                    operation += '+'
                elif result[0] == 11:
                    operation += '-'
                elif result[0] == 12:
                    operation += 'x'
                elif result[0] == 13:
                    operation += 'a'
                elif result[0] == 14:
                    operation += 'b'
                else:
                    operation += str(result[0])
            return operation

def preprocess_image(img):
    im = cv2.resize(img, (600, 200))
    ret, thresh = cv2.threshold(im, 127, 255, 0)
    kernel = np.zeros((3,3), np.uint8)
    #
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = thresh[10:, 10:]
    #
    retval, buffer_img = cv2.imencode('.png', thresh)
    data = base64.b64encode(buffer_img)
    data = str(data)
    data = str(data)[2:len(data)-1]
    return BytesIO(base64.urlsafe_b64decode(data))

def extract_imgs(img):
    img = ~img
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    #
    ctrs, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #
    cnt = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    #
    img_data = []
    rects = []
    for c in cnt:
        x, y, w, h = cv2.boundingRect(c)
        rect = [x, y, w, h]
        rects.append(rect)
    #
    bool_rect = []
    for r in rects:
        l = []
        for rec in rects:
            flag = 0
            if rec != r:
                if r[0] < (rec[0] + rec[2] + 10) and rec[0] < (r[0] + r[2] + 10) and r[1] < (rec[1] + rec[3] + 10) and rec[1] < (r[1] + r[3] + 10):
                    flag = 1
                l.append(flag)
            else:
                l.append(0)
        bool_rect.append(l)
    #
    dump_rect = []
    for i in range(0, len(cnt)):
        for j in range(0, len(cnt)):
            if bool_rect[i][j] == 1:
                area1 = rects[i][2] * rects[i][3]
                area2 = rects[j][2] * rects[j][3]
                if(area1 == min(area1, area2)):
                    dump_rect.append(rects[i])
    #

    final_rect = [i for i in rects if i not in dump_rect]
    for r in final_rect:
        x = r[0]
        y = r[1]
        w = r[2]
        h = r[3]

        im_crop = thresh[y:y+h+10, x:x+w+10]
        im_resize = cv2.resize(im_crop, (28, 28))
        im_resize = np.reshape(im_resize, (1, 28, 28))
        img_data.append(im_resize)

    return img_data

try:
    cnn = ConvolutionalNeuralNetwork()
    model_status = "Model loaded successfully!"
except Exception as e:
    model_status = f"Error loading model: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html', model_status=model_status)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        mode = request.form.get('mode', 'basic')
        
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        #
        processed_img = preprocess_image(img)
        #
        operation = cnn.predict(processed_img)
        #
        
        if mode == 'basic':
            try:
                operation = operation.replace('x', '*')
                result = eval(operation)
            except:
                result = "Could not evaluate expression"
        else:  # linear equation mode
            result = {}
            if 'a' in operation:
                result['a'] = solveEquation(operation, 'a')
            if 'b' in operation:
                result['b'] = solveEquation(operation, 'b')
            if not result:
                result = "No variables found to solve"

        retval, buffer = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'operation': operation,
            'result': result,
            'processed_image': f'data:image/png;base64,{img_base64}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)