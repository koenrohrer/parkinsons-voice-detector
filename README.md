# Parkinson’s Disease Detection

This repo contains two scripts plus the dataset:

- **model.py** trains and saves a Parkinson’s prediction model.  
- **predict.py** loads the saved model and lets you input new voice measurements to get a prediction.  
- **parkinsons_data.csv** is the raw dataset of voice features.

---

## model.py

1. **Reads** `parkinsons_data.csv`.  
2. **Extracts** all columns except `status` (and the `name` ID) as features, with `status` as the label (0 = healthy, 1 = Parkinson’s).  
3. **Scales** features to the range [-1, 1] with `MinMaxScaler`.  
4. **Splits** into 80% training / 20% testing.  
5. **Trains** an `XGBClassifier`.  
6. **Evaluates** accuracy on the test set.  
7. **Dumps** the fitted `scaler.pkl` and `model.pkl` via `joblib`.

---

## predict.py

1. **Loads** `scaler.pkl` and `model.pkl`.  
2. **Prompts** you to enter each voice feature (in the exact order below).  
3. **Applies** the same scaling.  
4. **Runs** `model.predict()` and prints:
   - “⚠️ Parkinson’s detected”  
   - “✅ No Parkinson’s detected”

---

## Dataset: parkinsons_data.csv headers

1. `name`  
   ­ Subject ID (not used as a feature).  
2. **MDVP:Fo(Hz)**  
   ­ Average vocal fundamental frequency.  
3. **MDVP:Fhi(Hz)**  
   ­ Maximum fundamental frequency.  
4. **MDVP:Flo(Hz)**  
   ­ Minimum fundamental frequency.  
5. **MDVP:Jitter(%)**  
   ­ Relative average perturbation.  
6. **MDVP:Jitter(Abs)**  
   ­ Absolute average perturbation.  
7. **MDVP:RAP**  
   ­ Relative amplitude perturbation.  
8. **MDVP:PPQ**  
   ­ Five-point period perturbation quotient.  
9. **Jitter:DDP**  
   ­ Average absolute difference of period differences.  
10. **MDVP:Shimmer**  
    ­ Amplitude variation.  
11. **MDVP:Shimmer(dB)**  
    ­ Amplitude variation in decibels.  
12. **Shimmer:APQ3**  
    ­ Three-point amplitude perturbation quotient.  
13. **Shimmer:APQ5**  
    ­ Five-point amplitude perturbation quotient.  
14. **MDVP:APQ**  
    ­ Eleven-point amplitude perturbation quotient.  
15. **Shimmer:DDA**  
    ­ Average absolute difference of amplitude differences.  
16. **NHR**  
    ­ Noise-to-Harmonics Ratio.  
17. **HNR**  
    ­ Harmonics-to-Noise Ratio.  
18. **status**  
    ­ Label (0 = healthy, 1 = Parkinson’s).  
19. **RPDE**  
    ­ Recurrence Period Density Entropy.  
20. **DFA**  
    ­ Detrended Fluctuation Analysis.  
21. **spread1**  
    ­ Nonlinear measure of signal spread.  
22. **spread2**  
    ­ Nonlinear measure of signal spread.  
23. **D2**  
    ­ Correlation dimension estimate.  
24. **PPE**  
    ­ Pitch Period Entropy.

> **IMPORTANT:** When you run `predict.py`, enter values in exactly this order (excluding `name` and `status`) to ensure the scaler and model align correctly.