err_face_invalid = '3'
err_file_invalid = '2'
err_baldify_error = '0'

class BaldifyException(Exception):
    def __init__(self, value):
        self.code = value
    def __str__(self):
        return repr(self.code)