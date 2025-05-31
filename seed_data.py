from app import create_app, db
from app.models.data_models import SalesData, Customer
from app.utils import generate_sample_customer_data
from app.ml_models import import_sales_csv_to_db
import os
import traceback
import random # For generating random names and emails
from sqlalchemy.exc import IntegrityError # To catch unique constraint violations

app = create_app()

def generate_random_name_email(index):
    """Generates a somewhat unique name and email based on an index."""
    first_names = ["Alex", "Jamie", "Chris", "Jordan", "Taylor", "Morgan"]
    last_names = ["Smith", "Jones", "Williams", "Brown", "Davis", "Miller"]
    name = f"{random.choice(first_names)} {random.choice(last_names)} {index}"
    email_name = name.lower().replace(" ", ".")
    email = f"{email_name}@example.com"
    return name, email

def seed():
    with app.app_context():
        print("Dropping all existing database tables...")
        db.drop_all()
        print("All tables dropped.")

        print("Creating database tables based on current models...")
        db.create_all()
        print("Database tables created.")

        # Add Sales Data from CSV
        print("Importing sales data from train.csv...")
        project_root = os.path.dirname(os.path.abspath(__file__))
        # Assuming train.csv is directly in the project root /Users/omtailor/predictive_dashboard/
        csv_path = os.path.join(project_root, 'data/train.csv')

        if not os.path.exists(csv_path):
            print(f"ERROR: CSV file not found at {csv_path}. Please check the path.")
        else:
            try:
                import_sales_csv_to_db(csv_path) # This function is in app/ml_models.py
            except Exception as e:
                print(f"An error occurred during CSV import for sales data: {e}")
                traceback.print_exc()

        # Add Sample Customer Data using all fields from generate_sample_customer_data
        # and adding randomized name and email.
        print("Generating and adding detailed sample customer data...")
        sample_customer_dictionaries = generate_sample_customer_data(num_records=150) # Default from utils.py
        
        customers_added_count = 0
        for i, cust_dict in enumerate(sample_customer_dictionaries):
            try:
                # Generate and add name and email, as they are not in cust_dict
                # but are required by the Customer model.
                name, email = generate_random_name_email(i)
                cust_dict['name'] = name
                cust_dict['email'] = email
                
                # Ensure customer_id_external is unique if your model enforces it
                # (The generator already makes it unique based on index 'i')
                # existing_cust_by_external_id = Customer.query.filter_by(customer_id_external=cust_dict['customer_id_external']).first()
                # if existing_cust_by_external_id:
                #     print(f"INFO: Customer with external_id {cust_dict['customer_id_external']} already exists. Skipping.")
                #     continue

                customer = Customer(**cust_dict)
                db.session.add(customer)
                
                # Commit one by one to catch unique email/id issues more easily for now
                # For bulk inserts, typically commit after the loop.
                db.session.commit() 
                customers_added_count += 1

            except IntegrityError as ie:
                db.session.rollback() # Rollback the failed addition
                print(f"IntegrityError while adding customer {cust_dict.get('customer_id_external', 'N/A')}: {ie}")
                print("This might be due to a duplicate email or customer_id_external if they have unique constraints.")
                print(f"Problematic data: {cust_dict}")
            except TypeError as te:
                db.session.rollback()
                print(f"TypeError while creating Customer object with data {cust_dict}: {te}")
                print("Ensure Customer model fields in 'app/models/data_models.py' match keys in the generated dictionary.")
                traceback.print_exc()
            except Exception as ex:
                db.session.rollback()
                print(f"An unexpected error occurred while creating Customer object {cust_dict.get('customer_id_external', 'N/A')}: {ex}")
                traceback.print_exc()
        
        # Final commit if not committing one by one
        # if customers_added_count > 0:
        #     try:
        #         db.session.commit()
        #     except Exception as e:
        #         db.session.rollback()
        #         print(f"Error during final commit of customer data: {e}")
        #         traceback.print_exc()

        print(f"Successfully attempted to add {customers_added_count} sample customer records.")
        print("Database seeding process completed.")

if __name__ == '__main__':
    seed()