from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score

def home(request):
    return render(request, 'home.html')

def predict(request):
    if request.method == 'POST':
        # Get form values
        a = float(request.POST.get('Temp'))
        b = float(request.POST.get('hum'))
        c = float(request.POST.get('ph'))

        # Load and train model each time
        df = pd.read_csv('data.csv')
        x = df[['Temp', 'hum', 'ph']]
        y = df['rain']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        res = r2_score(y_test, y_pred)

        # Make prediction
        ans = model.predict([[a, b, c]])
        if ans[0]==0:
            prediction = 'No Rain'
        else:
            prediction = 'Rain'

        return render(request, 'result.html', {
            'prediction': prediction,
            'score': res
        })
