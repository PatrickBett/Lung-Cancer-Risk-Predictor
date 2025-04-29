from django.db import models

class PredictionRecord(models.Model):
    GENDER_CHOICES = [('M', 'Male'), ('F', 'Female')]

    age = models.IntegerField()
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    air_pollution = models.IntegerField()
    alcohol_use = models.IntegerField()
    dust_allergy = models.IntegerField()
    occupational_hazards = models.IntegerField()
    genetic_risk = models.IntegerField()
    chronic_lung_disease = models.IntegerField()
    balanced_diet = models.IntegerField()
    obesity = models.IntegerField()
    smoking = models.IntegerField()
    passive_smoker = models.IntegerField()
    chest_pain = models.IntegerField()
    coughing_of_blood = models.IntegerField()
    fatigue = models.IntegerField()
    weight_loss = models.IntegerField()
    shortness_of_breath = models.IntegerField()
    wheezing = models.IntegerField()
    swallowing_difficulty = models.IntegerField()
    clubbing_of_finger_nails = models.IntegerField()
    frequent_colds = models.IntegerField()
    dry_cough = models.IntegerField()
    snoring = models.IntegerField()
    prediction = models.IntegerField()

    submitted_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction {self.id} - Risk: {self.prediction}"
