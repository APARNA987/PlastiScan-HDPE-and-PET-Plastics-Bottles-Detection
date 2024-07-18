import os
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img
from ultralytics import YOLO
import matplotlib.pyplot as plt

app = Flask(__name__)
model = YOLO('best.pt')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def innerpage():
    return render_template("inner-page.html")

@app.route("/submit", methods=["POST"])
def submit():
    if request.method =="POST":
        file = request.files["file"]
        
        if file:
            #base_path = os.path.dirname(__file__)
            file_path = os.path.join("uploads", file.filename)
            file.save(file_path)
            
            img = load_img(file_path)
            
            # Assuming model.predict() takes image data as input
            results = model.predict(img)
            
          
            # Assuming results[0] is the predicted image
            img = results[0].plot()
            plt.imshow(img)
            save_path = os.path.join("static", "predicted_image.jpg")
            plt.savefig(save_path)  # Save the predicted image
            plt.close()
            
            
        
            
            return render_template("portfolio.html", predict="predicted_image.jpg")
        else:
            return "No file uploaded"

if __name__ == '__main__':
    app.run(port=5678)



  # If model.predict() returns a path to the predicted image
  # uncomment below and comment the previous line
  # results = model.predict(file_path)
  
