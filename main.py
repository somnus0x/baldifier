import os
import io
import sys
import imghdr
from flask import Flask, Response, request, send_file
from werkzeug.utils import secure_filename
from baldify import Baldify
from baldify_err import *

baldifier = Baldify()
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    
    image_input = request.data
    try:
        image_output = baldifier.baldify(image_input)
    except BaldifyException as e:
        return Response(e.code, status=400, mimetype='text/plain')
    except Exception as e:
        return Response(err_baldify_error, status=400, mimetype='text/plain')
    else:
        return send_file(
                        io.BytesIO(image_output),
                        attachment_filename='head.png',
                        mimetype='image/png'
                )

print(sys.argv[1])
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(sys.argv[1]))
