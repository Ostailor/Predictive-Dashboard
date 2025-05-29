from app import create_app, db
from app.models.data_models import SalesData, Customer
from app.utils import generate_sample_sales_data, generate_sample_customer_data
from datetime import datetime

app = create_app()

def seed():
    with app.app_context():
        # Optional: Clear existing data
        SalesData.query.delete()
        Customer.query.delete()
        db.session.commit()
        print("Cleared existing sales and customer data.")

        # Add Sample Sales Data
        print("Generating sample sales data...")
        sample_sales = generate_sample_sales_data() # Removed num_records argument here
        for sale_data in sample_sales:
            sale = SalesData(**sale_data)
            db.session.add(sale)
        db.session.commit()
        print(f"Added {len(sample_sales)} sample sales records.")

        # Add Sample Customer Data
        print("Generating sample customer data...")
        sample_customers = generate_sample_customer_data(num_records=150) # num_records is still valid for customers
        for cust_data in sample_customers:
            customer = Customer(**cust_data)
            db.session.add(customer)
        db.session.commit()
        print(f"Added {len(sample_customers)} sample customer records.")

        print("Database seeded successfully.")

if __name__ == '__main__':
    seed()