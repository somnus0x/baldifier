import os
import io
from baldify import Baldify

baldifier = Baldify()

f = open('./face/1.jpg', 'r+')
jpgdata = f.read()
f.close()

out = baldifier.baldify(jpgdata)