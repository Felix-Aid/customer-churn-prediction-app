
# 🧠 Customer Churn Prediction App with SHAP

## 🖥️ Demo Preview

![Churn App Demo](assets/churn_app_demo.gif)


A user-friendly Streamlit web app...

This Streamlit-based web app allows users to **predict telecom customer churn** and **understand the prediction results** using **SHAP (SHapley Additive exPlanations)** visualizations.

It is designed for business users, data scientists, and AI/ML enthusiasts who want actionable insights and model explainability.

---

## 📌 Key Features

- Upload CSV files to predict churn for multiple customers.
- Visual display of:
  - ✅ Churn prediction results
  - 📈 Churn distribution
  - 🧠 SHAP-based feature importance
- Sample CSV download for quick testing.
- Built-in handling for encoded categorical variables (e.g., `Contract`).
- Download results as a CSV file.

---

## 🗂️ Project Structure

```
├── app.py / app_churn.py         # Main Streamlit app
├── model.pkl                     # Trained Random Forest model
├── scaler.pkl                    # StandardScaler used for features
├── sample_input.csv              # Working example file
├── requirements.txt              # All dependencies
└── README.md                     # This file
```

---

## 🧪 Sample Data Format

```csv
tenure,MonthlyCharges,TotalCharges,SeniorCitizen,Contract
12,70.35,842.2,0,1
24,89.45,2150.65,1,2
1,29.85,29.85,0,0
```

- `Contract`: Encoded as:
  - 0 → Month-to-month
  - 1 → One year
  - 2 → Two year

Make sure to match this format for predictions to work properly.

---

## ⚙️ How to Run Locally

1. **Create conda environment**:
   ```bash
   conda create -n churn-env python=3.10
   conda activate churn-env
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

---

## 🌐 Deployment (Streamlit Cloud)

You can deploy this project for free using [Streamlit Cloud](https://streamlit.io/cloud):

1. Push this repo to GitHub.
2. Connect your GitHub to Streamlit Cloud.
3. Choose `app.py` or `app_churn.py` as the entry point.
4. Add your `model.pkl`, `scaler.pkl`, and `requirements.txt`.

✅ That’s it!

---

## 🔍 Tech Stack

- Python 3.10
- Streamlit
- Scikit-learn
- SHAP
- Pandas, NumPy, Seaborn, Matplotlib

---

## 📌 Use Case

This app is ideal for:

- Telco customer churn analysis
- Data science portfolio projects
- Business insight presentations
- Model explainability demonstrations using SHAP

---

## 📧 Contact

**Felix Stephen Aidoo**  
Email: felixaidoo@example.com  
LinkedIn: [linkedin.com/in/your-profile](https://www.linkedin.com/in/felix-s-aidoo)
