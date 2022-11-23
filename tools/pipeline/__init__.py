import os

from flask import Flask
from flask_login import LoginManager

from database import db
from models import User

def create_app():
    """
    Init app with config and create database
    Return app <object>
    """
    app = Flask(__name__, instance_relative_config=False)
    basedir = os.path.abspath(os.path.dirname(__file__))
    app.config.from_pyfile("config.py", silent=True)
    app.config['SQLALCHEMY_DATABASE_URI'] =\
        'sqlite:///' + os.path.join(basedir, 'static', 'uploads', 'AD.sqlite3')
    db.init_app(app)
    # Login manager
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)
    @login_manager.user_loader
    def load_user(user_id):
        """
        Return current user login
        """
        return User.query.get(int(user_id))
    return app
    