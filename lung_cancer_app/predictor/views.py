from django.shortcuts import render
from .forms import LungCancerForm
from .models import PredictionRecord
import joblib
import numpy as np
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load the model (adjust path as needed)

model_path = os.path.join(BASE_DIR, 'predictor', 'predictor_model', 'models', 'final_model_logistic_regression.pkl')

model = joblib.load(model_path)

def home(request):
    prediction = None
    if request.method == 'POST':
        form = LungCancerForm(request.POST)
        if form.is_valid():
            cleaned = form.cleaned_data
            input_features = [
                int(cleaned['age']),
                1 if cleaned['gender'] == 'M' else 0,
                int(cleaned['air_pollution']),
                int(cleaned['alcohol_use']),
                int(cleaned['dust_allergy']),
                int(cleaned['occupational_hazards']),
                int(cleaned['genetic_risk']),
                int(cleaned['chronic_lung_disease']),
                int(cleaned['balanced_diet']),
                int(cleaned['obesity']),
                int(cleaned['smoking']),
                int(cleaned['passive_smoker']),
                int(cleaned['chest_pain']),
                int(cleaned['coughing_of_blood']),
                int(cleaned['fatigue']),
                int(cleaned['weight_loss']),
                int(cleaned['shortness_of_breath']),
                int(cleaned['wheezing']),
                int(cleaned['swallowing_difficulty']),
                int(cleaned['clubbing_of_finger_nails']),
                int(cleaned['frequent_colds']),
                int(cleaned['dry_cough']),
                int(cleaned['snoring']),
            ]
            prediction = model.predict([input_features])[0]

            # Save to DB if using model
            PredictionRecord.objects.create(
                age=cleaned['age'],
                gender=cleaned['gender'],
                air_pollution=cleaned['air_pollution'],
                alcohol_use=cleaned['alcohol_use'],
                dust_allergy=cleaned['dust_allergy'],
                occupational_hazards=cleaned['occupational_hazards'],
                genetic_risk=cleaned['genetic_risk'],
                chronic_lung_disease=cleaned['chronic_lung_disease'],
                balanced_diet=cleaned['balanced_diet'],
                obesity=cleaned['obesity'],
                smoking=cleaned['smoking'],
                passive_smoker=cleaned['passive_smoker'],
                chest_pain=cleaned['chest_pain'],
                coughing_of_blood=cleaned['coughing_of_blood'],
                fatigue=cleaned['fatigue'],
                weight_loss=cleaned['weight_loss'],
                shortness_of_breath=cleaned['shortness_of_breath'],
                wheezing=cleaned['wheezing'],
                swallowing_difficulty=cleaned['swallowing_difficulty'],
                clubbing_of_finger_nails=cleaned['clubbing_of_finger_nails'],
                frequent_colds=cleaned['frequent_colds'],
                dry_cough=cleaned['dry_cough'],
                snoring=cleaned['snoring'],
                prediction=prediction,
            )
    else:
        form = LungCancerForm()

    return render(request, 'predictor/home.html', {'form': form, 'prediction': prediction})
