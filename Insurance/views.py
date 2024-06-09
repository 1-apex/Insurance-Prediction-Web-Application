# views.py in your Django app
from django.http import HttpResponse
from django.shortcuts import render
import joblib
import numpy as np

def index(request):
    return render(request, 'predict_premium.html')

def predict_premium(request):
    if request.method == 'POST':
        age = int(request.POST.get('age'))
        diabetes = int(request.POST.get('diabetes'))
        blood_pressure_problems = int(request.POST.get('blood_pressure_problems'))
        transplants = int(request.POST.get('transplants'))
        chronic_diseases = int(request.POST.get('chronic_diseases'))
        height = int(request.POST.get('height'))
        weight = int(request.POST.get('weight'))
        known_allergies = int(request.POST.get('known_allergies'))
        history_of_cancer = int(request.POST.get('history_of_cancer'))
        num_major_surgeries = int(request.POST.get('num_major_surgeries')) 

        # Load the scaler and model
        scaler = joblib.load('C:/Users/prath/PycharmProjects/Insurance/Insurance/model/scaler.pkl')
        model = joblib.load('C:/Users/prath/PycharmProjects/Insurance/Insurance/model/insurance_premium_model.pkl')

        # Preprocess the user input
        user_data = np.array([[age, diabetes, blood_pressure_problems, transplants, chronic_diseases, 
                               height, weight, known_allergies, history_of_cancer, num_major_surgeries]])
        user_data_scaled = scaler.transform(user_data)

        # Make the prediction
        predicted_premium = model.predict(user_data_scaled)
        predicted_premium = round(predicted_premium[0], 2)

        return render(request, 'result.html', {'predicted_premium': predicted_premium})

    return render(request, 'predict_premium.html')
