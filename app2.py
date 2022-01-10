import flask
from PIL import Image
app = flask.Flask(__name__)


print("Loding model")
from model import *
print("Modle Loaded")


@app.route('/', methods = ['GET'])
def welcome():
    return "Hello World"
@app.route('/predict', methods = ['POST'])
def handle_request():
    print(len(flask.request.files.keys()))
    imagefile = flask.request.files['data']

    print("\nReceived image File name : " + imagefile.filename)


    image = Image.open(imagefile)
    width, height = image.size
    resized = image.resize(size=(width, height))
    if height > 384:
        width = int(width / height * 384)
        height = 384
        resized = resized.resize(size=(width, height))
    width, height = resized.size
    if width > 512:
        width = 512
        height = int(height / width * 512)
        resized = resized.resize(size=(width, height))

    print("Predicting")
    print (image)
    caption = predict(image)
    caption_en = caption
    print(caption)
    image.close()
    return caption

if __name__ == '__main__':
   app.run(host='0.0.0.0')