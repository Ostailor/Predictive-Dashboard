from app import db # Import the db instance from app/__init__.py
from datetime import datetime

class SalesData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    amount = db.Column(db.Float, nullable=False)
    product_category = db.Column(db.String(100))
    region = db.Column(db.String(100))

    def __repr__(self):
        return f'<SalesData {self.id} - {self.timestamp} - ${self.amount}>'

class Customer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    customer_id_external = db.Column(db.String(100), unique=True, nullable=False, index=True) # External ID
    signup_date = db.Column(db.DateTime, default=datetime.utcnow)
    last_activity_date = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    total_spend = db.Column(db.Float, default=0.0)
    # Features for churn prediction
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10)) # e.g., 'Male', 'Female', 'Other'
    contract_type = db.Column(db.String(50)) # e.g., 'Monthly', 'Annual'
    monthly_charges = db.Column(db.Float)
    total_charges = db.Column(db.Float)
    has_support_ticket = db.Column(db.Boolean, default=False)
    churned = db.Column(db.Boolean, default=False, index=True) # Target variable

    def __repr__(self):
        return f'<Customer {self.customer_id_external} - Churned: {self.churned}>'

# Add other models here as needed, e.g., WebsiteTraffic
# class WebsiteTraffic(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
#     page_views = db.Column(db.Integer, nullable=False)
#     unique_visitors = db.Column(db.Integer)
#     source = db.Column(db.String(100)) # e.g., 'organic', 'referral', 'direct'