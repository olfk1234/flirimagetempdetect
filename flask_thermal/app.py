
from flask import Flask, request, send_file
from flask_cors import CORS
import ultralytics
import cv2
from PIL import Image

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = 'static'

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        image_file = request.files['imageFile']
        min_temperature = request.form['minTemperature']
        max_temperature = request.form['maxTemperature']
        temperature_unit = request.form['temperatureUnit']
        temperature_unit_output = request.form['temperatureUnitOutput']
        
        if image_file:
            image_file.save('uploads/' + image_file.filename)

        image = cv2.imread('uploads/' + image_file.filename)
        image = cv2.resize(image, (640, 640))
        
        model_2 = ultralytics.YOLO("best.pt")

        results = model_2(image)

        grayscale_image = cv2.cvtColor(image, cv2.IMREAD_ANYDEPTH)

        grayscale_image_max = grayscale_image.max()
        grayscale_image_min = grayscale_image.min()
        normalized_image = (grayscale_image - grayscale_image_min) * (1/(grayscale_image_max - grayscale_image_min)) # Normalize temperatures

        normalized_image = float(min_temperature) + (float(max_temperature) - float(min_temperature)) * normalized_image

        
        for result in results:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0]  
                label = result.names[int(box.cls)]
                conf = box.conf
                x = int((x_min+x_max)/2)
                y = int((y_min+y_max)/2)
                temp = normalized_image[y,x][2]
                if(temperature_unit[0] == "C" and temperature_unit_output[0] == "F"):

                    temp = (temp * 9/5) + 32 

                elif(temperature_unit[0] == "F" and temperature_unit_output[0] == "C"):
                    temp = (temp-32) * (5/9)
                else:
                    temp = 1*temp
                print(temp)
                cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                cv2.rectangle(image, (x, y), (x, y), (0, 255, 0), 2)
                cv2.putText(image, label + ' ' + str(conf) + ", " + str(round(temp,2))+ " degrees " + temperature_unit_output[0], (int(x_min), int(y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        image = Image.fromarray(image)
        image.save("static//img.jpg")
        return "Image received", 200
    return 'No image received', 400

@app.route('/', methods=['GET'])
def send_image():
    return send_file("img.jpg", mimetype='image/jpg')

if __name__ == '__main__':
    app.run(debug=True)
