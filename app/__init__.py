from flask import Flask
from app.config import Config
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)

    # Import models here so that create_all knows about them
    from app.models import data_models # Add this line

    # Register blueprints here
    from app.routes import bp as main_bp
    app.register_blueprint(main_bp)

    with app.app_context():
        db.create_all() # Creates tables based on models defined (we'll define models later)

    return app