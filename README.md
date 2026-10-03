# EduPulse
Student Dropout Prediction is a machine learning Streamlit app that predicts whether a student is likely to Dropout, remain Enrolled, or Graduate. It uses academic and admission details such as age, grades, approved units, evaluations, tuition status, and course information to generate predictions.

# 🎓 EduPulse: Student Success Predictor

**Early-warning insights for student success.**

EduPulse is an interactive Streamlit app that predicts whether a student is likely to
**Drop out**, stay **Enrolled**, or **Graduate**, based on admission details and
first-year academic performance.

## ✨ Features

- **Instant predictions** with a colour-coded result card
- **Probability breakdown** for all three outcomes
- **Profile signals**, such as pass rate, grade trend and tuition status
- **Suggested next steps** for each predicted outcome
- **Sample profiles** (strong, average, struggling) for quick demos
- **Insights tab** comparing semester grades and approved vs evaluated units
- **Session history** of predictions, with CSV export
- **Downloadable reports** (TXT and CSV)
- Input validation, for example approved units can't exceed evaluated units

## 🧠 Model Inputs

| Feature | Description |
|---|---|
| Age at enrollment | 15–100 |
| Admission grade | 0–200 |
| Course | Numeric course code |
| Tuition fees up to date | 0 = No, 1 = Yes |
| 1st / 2nd semester grade | 0–20 |
| 1st / 2nd semester units evaluated | Whole number |
| 1st / 2nd semester units approved | Whole number (≤ evaluated) |

**Output classes:** `0 = Dropout`, `1 = Enrolled`, `2 = Graduate`

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install dependencies
pip install streamlit pandas joblib scikit-learn

# 3. Run the app
streamlit run edupulse_app.py
```

> Make sure `student_dropout_model.pkl` is in the same folder as `edupulse_app.py`.

## 🛠️ Tech Stack

Python · Streamlit · Pandas · Scikit-learn · Joblib

## ⚠️ Disclaimer

This is a machine-learning demo. Predictions reflect patterns in the training data
and should support, not replace, decisions made by educators and advisors.

## 📄 License

MIT (or your preferred license)
