import flask
import os
import sys
# Create the application.
if getattr(sys, 'frozen', False):
    template_folder = os.path.join(sys._MEIPASS, 'templates')
    APP = flask.Flask(__name__, template_folder=template_folder)
else:
    APP = flask.Flask(__name__)


BASE_PATH = os.path.dirname(__file__)
@APP.route('/')
def index():
    """ Displays the index page accessible at '/'
    """
    return flask.render_template('index.html')


if __name__ == '__main__':
    APP.debug=True
    APP.run()