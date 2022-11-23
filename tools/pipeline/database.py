from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def init_app(app):
    """Init app
    """
    db.init_app(app)

def drop_table():
    """Reflect and drop all
    """
    db.reflect()
    db.drop_all()
