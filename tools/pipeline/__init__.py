from flask import Flask, Blueprint
from database import db
from models import User
import os
from flask_login import LoginManager

def create_app():
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
        return User.query.get(int(user_id))
    
    # from auth import auth as auth_blueprint
    # app.register_blueprint(auth_blueprint)
    # from views import views as views_blueprint
    # app.register_blueprint(views_blueprint)
    return app