import pickle
import numpy as np

with open('pipeline.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

example = {
  "lead_source": "paid_ads",
  "industry": "technology",
  "employment_status": "employed",
  "location": "north_america",
  "number_of_courses_viewed": 2,
  "annual_income": 79276.0,
  "interaction_count": 4,
  "lead_score": 0.41
}
def predict_single(example, dv, model):
  X = dv.transform([example])  ## apply the one-hot encoding feature to the customer data 
  y_pred = model.predict_proba(X)[:, 1]
  return y_pred[0]

np.round(predict_single(example,dv,model),3)