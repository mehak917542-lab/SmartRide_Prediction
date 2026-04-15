# Used Car Price Prediction

This project provides a web application to predict the price of used cars based on various features. The application is built using Flask and leverages a machine learning model trained on a dataset of used car specifications.

## Features

**User-friendly Interface:** A clean and intuitive web form allows users to input car details for price prediction.
**Comprehensive Input Fields:** Predict car prices based on:
    * Car Name
    * Brand
    * Model
    * Vehicle Age (Years)
    * Kilometers Driven
    * Number of Seats
    * Seller Type
    * Fuel Type
    * Transmission
    * Mileage (km/l)
    * Engine (CC)
    * Max Power (BHP)
**Machine Learning Powered:** Utilizes a pre-trained machine learning model for accurate predictions.

## Live Website

You can access the live prediction website here:https://web-production-f5ded.up.railway.app/

## Technologies Used

* **Frontend:** HTML, CSS (implied by the web interface)
* **Backend:** Flask (Python web framework)
* **Machine Learning:** scikit-learn
* **Data Manipulation:** pandas, numpy
* **Data Visualization (for development/analysis):** matplotlib, seaborn
* **Model Persistence:** joblib
* **Web Server Gateway Interface:** gunicorn (for production deployment)
