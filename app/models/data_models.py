from app import db
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

class Customer(db.Model):
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4) # Internal UUID
    customer_id_external = db.Column(db.String(50), unique=True, nullable=True) # From your utils
    name = db.Column(db.String(100), nullable=False) # You'll need to decide how to get this
    email = db.Column(db.String(100), unique=True, nullable=False) # You'll need to decide how to get this

    # Add fields from app/utils.py that you want to store:
    signup_date = db.Column(db.DateTime, nullable=True)
    last_activity_date = db.Column(db.DateTime, nullable=True)
    total_spend = db.Column(db.Float, nullable=True)
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(50), nullable=True)
    contract_type = db.Column(db.String(50), nullable=True)
    monthly_charges = db.Column(db.Float, nullable=True)
    total_charges = db.Column(db.Float, nullable=True)
    has_support_ticket = db.Column(db.Boolean, nullable=True)
    churned = db.Column(db.Boolean, nullable=True)

    # Note: generate_sample_customer_data doesn't produce 'name' and 'email'.
    # You'll need to add them to the generator or make them nullable/remove them
    # if customer_id_external is the primary way to identify customers from that data

class SalesData(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    date = db.Column(db.Date, nullable=False)
    store = db.Column(db.Integer, nullable=False)
    item = db.Column(db.Integer, nullable=False)
    sales = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f'<SalesData {self.date} Store {self.store} Item {self.item} Sales {self.sales}>'
