# Classification Project: Telco Churn

# Project Description
Telco is experiencing customer churn and the company wants to investigate to find possible causes and actionable solutions.

# Project Goal
- Discover drivers of churn in customers in the telco dataset
- Use drivers to develop a machine learning model to classify customers as customers who will churn or not churn
- A churned customer is defined as a customer who is no longer with the company
- This information could be used to reach a better understanding of why customers churn from Telco

# Initial Thoughts
My initial hypothesis is that drivers of churn will likely be features that relate to the personal life of the customer, such as having a family/partner, and being a senior citizen. I added two more features relating to having multiple lines with phone service and having internet service with Telco.

# Plan
- Acquire data from SQL database

- Prepare data by encoding strings, renaming columns, dropping unnecessary columns

- Explore data in search of drivers of upsets and answer the following:

>Is tenure significantly different in customers who have churned than customers who have not churned?

>Does a customer having a partner affectchurn?

>Does a customer having dependents play a role in churn?

>Is a customer's senior citizen status related to churn?

>Does a customer having internet service affect churn?

>Does a customer having multiple lines with phone service affect churn?

- Develop a model to predict if a customer will churn
> Use drivers identified in explore to build predictive models

> Evaluation of models on train and validate data

> Select best model based on highest accuracy

> Evaluation of best model on test data

- Draw conclusions

# Data Dictionary
| Feature | Definition |
| :- | :- |
| Churn | True or False, Churned customers have left the company |
| Multiple Lines | True or False, customer has phone service with multiple lines or not |
| Dependents | True or False, customer has dependents or not |
| Partner | True or False, customer has a partner or not |
| Senior Citizen | True or False, customer is a senior citizen or not |
| Internet Service | True or False, customer has internet service or not |
| Tenure | An integer that shows the number of months a customer has been with the company |

# Steps to Reproduce
1. Clone this repo
2. Acquire the data from SQL database
3. Place data in file containing the cloned repo
4. Run notebook

# Takeaways and Conclusions
- Churn occurs in 27% of customers
- Customers who do not have a partner have a 13% higher chance of churning
- Customer who do not have dependents have a 15% higher chance of churning
- Customers who are senior citizens have a 20% higher chance of churning
- Customers who do not have internet service have a 25% higher chance of churning
- Having multiple lines with phone service is a driver of churn with weak influence
- Model performs 6% higher than baseline predictions.

# Recommendations
- Possible limited time discounts to customers who are likely to churn within the featured demographics.
