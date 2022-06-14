import os
from base64 import b64decode, b64encode

import cv2 as cv
import numpy as np
import paddlehub as hub
from flask import Flask, request, flash, redirect, send_file
from flask import jsonify
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.secret_key = os.urandom(12).hex()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def to_img(string: str):
    img = b64decode(string)
    return img


def to_base64(src):
    string = b64encode(src).decode()
    return string


@app.route('/', methods=['POST'])
def hello_world():  # put application's code here
    return 'Hello World!'


seg_dir='seg_output'
slr_dir='slr_output'


'''
输入为图片，使用form-data格式
返回图像
'''


@app.route('/human_segmentation', methods=['POST'])
def human_seg():
    if 'file' not in request.files:
        flash('No file part')
        return 'no files'
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filename = filename[:-4]+'_humanseg' + filename[-4:]
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        src = cv.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        model = hub.Module(name="FCN_HRNet_W18_Face_Seg")
        result = model.Segmentation(images=[src], visualization=False)
        output = result[0].get('face')
        out_filename = os.path.join(seg_dir, filename[:-4]+'_res.png')
        cv.imwrite(out_filename, output)
        
        return send_file(out_filename, attachment_filename=filename)
        
        
    else:
        return redirect(request.url)


'''
超分需要限制图片大小，要不然计算量太大
'''
@app.route('/super_res', methods=['POST'])
def super_res():
    if 'file' not in request.files:
        flash('No file part')
        return 'no files'
    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filename = filename[:-4]+'_superres' + filename[-4:]
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        src = cv.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print(src.shape)
        sr_model = hub.Module(name='falsr_c')
        im = src.astype('float32')
        res = sr_model.reconstruct(images=[im], visualization=False, output_dir=None)
        output = res[0].get('data')
        out_filename = os.path.join(slr_dir, filename[:-4]+'_res.png')
        cv.imwrite(out_filename, output)
        
        return send_file(out_filename, attachment_filename=filename)
        
        
    else:
        return redirect(request.url)


if __name__ == '__main__':
    
    if not  os.path.exists(slr_dir):
        os.mkdir(slr_dir)
    if not os.path.exists(seg_dir):
        os.mkdir(seg_dir)
    
    app.debug = True
    app.run(debug=True)
