from flask import Flask, render_template, request, redirect, url_for, send_file
import os
from PIL import Image
import torch
import io
import time
import shutil
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'

# 创建上传和结果文件夹
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# 加载YOLOv5模型（首次会自动下载权重）
model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='yolov5n.pt')

if os.path.exists("./results") and os.path.isdir("./results"):
    shutil.rmtree("./results")

def detect_and_save(image_path, result_path):
    results = model(image_path)

    if os.path.exists("./results") and os.path.isdir("./results"):
        shutil.rmtree("./results")

    unique_dir = "results"
    
    results.save(save_dir=unique_dir)
    # YOLOv5会自动保存图片到save_dir，文件名和原图一致
    filename = os.path.basename(image_path)
    return os.path.join(unique_dir, filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return '没有文件部分'
        file = request.files['file']
        if file.filename == '':
            return '没有选择文件'
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            # 物体识别
            result_img_path = detect_and_save(filepath, app.config['RESULT_FOLDER'])
            print(filepath)
            print(result_img_path)
            return render_template('index.html', result_img=result_img_path, origin_img=filepath)
    return render_template('index.html', result_img=None, origin_img=None)

@app.route('/results/<filename>')
def result_file(filename):
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True) 
