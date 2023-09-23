
#pip install scikit-image pillow pytorch transformers tqdm opencv-python flask
#pip install git+https://github.com/openai/CLIP.git


print("Starting..")
from flask import Flask,render_template,request,json,jsonify,session,redirect,send_file,url_for,flash
import os
from werkzeug.utils import secure_filename
from CLIPImageCaptioning import get_caption

import cv2
import numpy as np
from PIL import Image 

# import the necessary packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import cv2



app=Flask(__name__)
app.secret_key="secure"
app.config['UPLOAD_FOLDER'] = './static/uploads'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/',methods=["post","get"])
def first_page():
    if request.method=="POST":
        global image_name,image_data

        file = request.files['file']
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print("=============")
            print(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("=============")
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename).replace("\\", "/"))

            op = get_caption('static/uploads/'+filename)
            solution = op
            return render_template("data_page.html",
                           filename=filename, result = op, solution = solution.split("\n"))
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif')
            return redirect(request.url)

    else:
        return render_template("form_page.html")


@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

app.run(debug=True, host="0.0.0.0")
