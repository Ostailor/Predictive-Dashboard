import random
from datetime import datetime, timedelta
import math

def generate_random_date(start_date, end_date):
    """Generates a random datetime between two datetime objects."""
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    if days_between_dates < 0: # Should not happen if start_date <= end_date
        days_between_dates = 0
    random_number_of_days = random.randrange(days_between_dates + 1) # +1 to include end_date possibility if time is 00:00
    random_date = start_date + timedelta(days=random_number_of_days,
                                         hours=random.randint(0, 23),
                                         minutes=random.randint(0, 59),
                                         seconds=random.randint(0, 59))
    return min(random_date, end_date) # Ensure it does not exceed end_date

def generate_sample_sales_data(): # Removed num_records, as we generate daily for the full range
    """Generates a list of sample sales data with trend and seasonality for a fixed period."""
    sales = []
    categories = ['Electronics', 'Clothing', 'Groceries', 'Books', 'Home Goods']
    regions = ['North', 'South', 'East', 'West', 'Central']
    
    start_date_overall = datetime(2023, 1, 1)
    end_date_overall = datetime(2025, 5, 28) # Current date in project context

    current_date = start_date_overall
    
    while current_date <= end_date_overall:
        base_amount = random.uniform(50.0, 150.0)

        days_from_start = (current_date - start_date_overall).days
        trend_factor = 1 + (days_from_start / (365 * 2.5)) * 0.20 # 20% growth over ~2.5 years

        month = current_date.month
        monthly_factor = 1.0
        if month == 11: monthly_factor = 1.3
        elif month == 12: monthly_factor = 1.6 # Stronger December peak
        elif month == 1: monthly_factor = 1.1 # January sales
        elif month == 2: monthly_factor = 0.75 # February dip
        elif month in [6,7,8]: monthly_factor = 1.15 # Summer bump

        day_of_week = current_date.weekday() # Monday=0, Sunday=6
        weekly_factor = 1.0
        if day_of_week == 4: weekly_factor = 1.15 # Friday
        elif day_of_week == 5: weekly_factor = 1.45 # Saturday
        elif day_of_week == 6: weekly_factor = 1.25 # Sunday
        
        final_amount = base_amount * trend_factor * monthly_factor * weekly_factor
        final_amount *= random.uniform(0.85, 1.15) # Add some daily noise
        final_amount = max(10.0, final_amount) # Ensure a minimum amount, can be non-zero

        sale = {
            'timestamp': current_date,
            'amount': round(final_amount, 2),
            'product_category': random.choice(categories),
            'region': random.choice(regions)
        }
        sales.append(sale)
        
        current_date += timedelta(days=1) # Strictly advance by one day
            
    return sales

def generate_sample_customer_data(num_records=150): # Adjusted default
    """Generates a list of sample customer data."""
    customers = []
    genders = ['Male', 'Female', 'Other', 'Prefer not to say']
    contract_types = ['Monthly', 'Annual', 'Two Year']
    # Ensure signup dates are reasonable relative to sales data period
    start_signup_date = datetime(2022, 6, 1) 
    end_signup_date = datetime(2025, 2, 28) # Customers signed up before the latest sales data

    for i in range(num_records):
        signup_date = generate_random_date(start_signup_date, end_signup_date)
        
        # Ensure last_activity_date is after signup_date and not beyond current project date
        possible_last_activity_start = signup_date + timedelta(days=1)
        possible_last_activity_end = datetime(2025, 5, 28) # Current project date
        
        if possible_last_activity_start > possible_last_activity_end:
            last_activity_date = possible_last_activity_start # Or handle as an edge case
        else:
            last_activity_date = generate_random_date(possible_last_activity_start, possible_last_activity_end)

        monthly_charges = round(random.uniform(20.0, 150.0), 2)
        
        # Calculate months_active more robustly
        if last_activity_date < signup_date: # Should not happen with above logic
            months_active = 0
        else:
            months_active = (last_activity_date.year - signup_date.year) * 12 + \
                            (last_activity_date.month - signup_date.month) + \
                            (1 if last_activity_date.day >= signup_date.day else 0)
        months_active = max(0, months_active) # Ensure non-negative

        total_charges = round(monthly_charges * months_active * random.uniform(0.8, 1.2), 2)
        if months_active == 0: # If same month signup and activity, ensure total_charges is at least partial monthly
            total_charges = max(total_charges, monthly_charges * random.uniform(0.1,1.0))


        customer = {
            'customer_id_external': f'CUST{1000+i:04d}',
            'signup_date': signup_date,
            'last_activity_date': last_activity_date,
            'total_spend': round(random.uniform(50.0, 5000.0), 2), # This might be independent or derived
            'age': random.randint(18, 70),
            'gender': random.choice(genders),
            'contract_type': random.choice(contract_types),
            'monthly_charges': monthly_charges,
            'total_charges': total_charges, # This is now more consistent
            'has_support_ticket': random.choice([True, False, False, False]), # Skew towards fewer tickets
            'churned': random.choice([True, False, False, False, False]) # Skew towards not churned for sample
        }
        customers.append(customer)
    return customers