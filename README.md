🎯 About The Diabetes Detection & Health Assessment System is a comprehensive web application that uses advanced machine learning algorithms to assess diabetes risk and current blood sugar status based on user-reported symptoms, lifestyle factors, and health indicators. This system provides:

Long-term diabetes risk assessment based on lifestyle and chronic symptoms Real-time blood sugar status evaluation based on current symptoms Professional PDF reports with personalized recommendations High accuracy predictions (92-97% with real datasets)

🎓 Educational Purpose This tool is designed for educational purposes and health awareness. It is NOT a substitute for professional medical diagnosis or advice.

✨ Features 🤖 Dual AI Assessment

Long-term Risk Model: Predicts future diabetes risk based on lifestyle and medical history Current Status Model: Evaluates immediate blood sugar concerns based on current symptoms

📊 Symptom-Based Analysis

No medical equipment required Simple questionnaire format 13 common diabetes symptoms 5 current status indicators

📄 Professional PDF Reports

Beautiful, medical-grade formatting Color-coded risk levels Personalized recommendations Patient information and timestamps Downloadable and shareable

🎨 Modern UI/UX

Responsive design (mobile, tablet, desktop) Smooth animations Real-time validation Progress indicators Color-coded results (Green = Low Risk, Red = High Risk)

🔒 Privacy-Focused

No data stored on servers Client-side PDF generation Secure HTTPS support GDPR compliant

🎬 Demo Live Demo Coming Soon Screenshots See Screenshots section below

🛠️ Technology Stack Backend

Python 3.8+ Flask 3.0.0 - Web framework scikit-learn 1.3.0 - Machine learning NumPy 1.24.3 - Numerical computing Pandas 2.0.3 - Data manipulation Flask-CORS 4.0.0 - Cross-origin resource sharing

Frontend

HTML5 - Structure CSS3 - Styling with gradients and animations JavaScript (ES6+) - Interactivity jsPDF 2.5.1 - PDF generation

Machine Learning Models

Random Forest Classifier - Ensemble learning Gradient Boosting Classifier - Boosting algorithm Logistic Regression - Linear classification Voting Classifier - Soft voting ensemble

Development Tools

PyCharm - IDE Git - Version control GitHub - Repository hosting

📦 Installation Prerequisites

Python 3.8 or higher pip (Python package manager) Modern web browser (Chrome, Firefox, Edge, Safari)

Step 1: Clone the Repository bashgit clone https://github.com/abhishuman18/diabetes-detection-system.git cd diabetes-detection-system Step 2: Create Virtual Environment (Recommended) bash# Windows python -m venv venv venv\Scripts\activate

macOS/Linux
python3 -m venv venv source venv/bin/activate Step 3: Install Dependencies bashpip install -r requirements.txt Step 4: Download Dataset (Optional - For Higher Accuracy) For maximum accuracy (92-97%), download the Early Stage Diabetes Risk Prediction Dataset:

Go to Kaggle Dataset Download diabetes_data_upload.csv Place it in the project root folder

Note: The system will work without this dataset using synthetic data (88-92% accuracy). Step 5: Run the Application bash# Using the real dataset version (recommended) python app_real_data.py

OR using the improved synthetic version
python app_improved.py

OR using the basic version
python app.py The application will start on http://localhost:5000

🚀 Usage

Access the Web Interface Open your browser and navigate to: http://localhost:5000
Fill Out the Assessment Form Test Information
Name: Your full name (required) Email: Your email address (optional) Phone: Your phone number (optional) Test Date/Time: Auto-filled (editable)

Personal Information

Age: Your age in years Gender: Male/Female/Other BMI Category: Select your body mass index range

Lifestyle & Medical History

Family History: Diabetes in close relatives Physical Activity: Your exercise frequency Diet Quality: Overall diet assessment

Common Symptoms (Check all that apply)

Frequent urination Excessive thirst Unexplained weight loss Constant fatigue Blurred vision Slow healing wounds Tingling in hands/feet Frequent infections

Current Status

Recent Meal: When you last ate Meal Type: Size/type of meal Current Symptoms: What you're feeling right now

Get Your Results Click "🔍 Analyze My Health Status" to receive:
Long-term diabetes risk assessment Current blood sugar status Risk probability percentages Personalized recommendations

Export PDF Report Click "📄 Export Report as PDF" to download a professional medical report with:
Patient information Assessment results Recommendations Health data summary

🧠 Model Information Training Data Real Medical Dataset (Recommended)

Early Stage Diabetes Risk Prediction Dataset 520 real patient records 16 symptom-based features Accuracy: 92-97%

📡 API Endpoints GET / Returns the main web interface (HTML). POST /predict Performs diabetes risk and current status prediction. Request Body: json{ "patient_name": "John Doe", "age": 45, "gender": "male", "bmi_category": 2, "family_history": 1, "physical_activity": 1, "diet_quality": 1, "frequent_urination": 1, "excessive_thirst": 1, "unexplained_weight": 0, "fatigue": 1, "blurred_vision": 0, "slow_healing": 1, "tingling_hands": 0, "frequent_infections": 0, "recent_meal": 1, "meal_type": 2, "feeling_shaky": 0, "excessive_hunger": 1, "sweating": 0, "confusion": 0, "rapid_heartbeat": 0 } Response: json{ "diabetes_risk": { "prediction": 1, "probability": { "low_risk": 15.3, "high_risk": 84.7 } }, "current_status": { "prediction": 0, "probability": { "normal": 78.2, "concerning": 21.8 } }, "patient_info": { "name": "John Doe", "test_date": "2024-01-15T14:30", "email": "john@example.com", "phone": "+1234567890", "gender": "male" } } GET /model-info Returns information about loaded models. Response: json{ "status": "ready", "models": { "diabetes_risk": "Optimized Ensemble (RF + GB + LR)", "current_status": "Optimized Ensemble (RF + GB + LR)" } }

📁 Project Structure diabetes-detection-system/ ├── app_real_data.py # Real dataset version (recommended) ├── requirements.txt # Python dependencies ├── README.md # This file │ ├── templates/ │ └── index.html # Main web interface │ ├── static/ # (Optional) Static assets │ ├── css/ │ ├── js/ │ └── images/ │ ├── models/ # (Auto-generated) │ ├── diabetes_risk_model.pkl │ ├── current_status_model.pkl │ ├── scaler_risk.pkl │ └── scaler_status.pkl │ ├── data/ # (Optional) │ └── diabetes_data_upload.csv # Real dataset │ ├── docs/ # Documentation │ ├── API.md │ ├── INSTALLATION.md │ └── CONTRIBUTING.md │ └── tests/ # (Future) ├── test_models.py └── test_api.py

🤝 Contributing Contributions are welcome! Here's how you can help: Ways to Contribute

🐛 Report bugs 💡 Suggest new features 📝 Improve documentation 🔧 Submit pull requests

Development Setup

Fork the repository Create a feature branch: git checkout -b feature/AmazingFeature Commit your changes: git commit -m 'Add some AmazingFeature' Push to the branch: git push origin feature/AmazingFeature Open a Pull Request

Coding Standards

Follow PEP 8 for Python code Use meaningful variable names Add comments for complex logic Write docstrings for functions Test before submitting

📄 License This project is licensed under the MIT License - see the LICENSE file for details. MIT License Summary MIT License

Copyright (c) 2025 Abhishuman Roy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

👨‍💻 Credits Developer Arushi Sengupta & Abhishuman Roy

GitHub: @arushi23git & @abhishuman18 Email: arushisenguptaofficial@gmail.com & abhishumanr@gmail.com

Project Information

Created: December 2025 Version: 1.0.0 Status: Active Development

Acknowledgments

scikit-learn for machine learning capabilities Flask for the excellent web framework jsPDF for client-side PDF generation Kaggle for providing the diabetes dataset UCI Machine Learning Repository for medical datasets

Special Thanks

Medical professionals who provided guidance on symptom assessment Open-source community for amazing tools and libraries Beta testers for valuable feedback

⚠️ Disclaimer Medical Disclaimer This application is an AI screening tool for educational and informational purposes only. It is NOT:

❌ A medical diagnosis ❌ A replacement for professional medical advice ❌ A substitute for clinical testing (HbA1c, fasting glucose) ❌ Suitable for emergency medical situations

ALWAYS consult qualified healthcare professionals for:

Proper medical diagnosis Treatment recommendations Health concerns Emergency situations
