# **Six Human Emotions Detection App** 🎭  

## **Overview**  
The **Six Human Emotions Detection App** is a **machine learning-based web application** that predicts emotions from text using **Logistic Regression** and **TF-IDF vectorization**. Built with **Streamlit**, the app classifies input text into one of six emotions:

✅ **Joy**  
✅ **Fear**  
✅ **Anger**  
✅ **Love**  
✅ **Sadness**  
✅ **Surprise**  

---
## **Features**  
- **📝 Text Preprocessing**: Cleans input text by removing special characters, stopwords, and applying stemming.  
- **🔍 TF-IDF Vectorization**: Converts text into numerical features using a pre-trained TF-IDF vectorizer.  
- **🤖 Emotion Classification**: Uses a **Logistic Regression model** to predict emotions from text.  
- **📊 Confidence Score**: Displays the probability of the predicted emotion.  
- **🌐 Web Interface**: User-friendly **Streamlit** interface for easy interaction.  

---
## **Installation & Usage**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/your-username/emotion-detection-app.git
cd emotion-detection-app
```

### **2. Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3. Add Model Files**  
Place the following **pre-trained model files** in the project directory:  
- `logistic_regresion.pkl`
- `tfidf_vectorizer.pkl`
- `label_encoder.pkl`

### **4. Run the App**  
```bash
streamlit run app.py
```

### **5. Predict Emotion**  
Enter your text in the input box, click "Predict," and the app will display the detected emotion along with a confidence score.

---
## **Applications**  
🚀 **Sentiment Analysis**: Analyze customer reviews, social media comments, or user feedback.  
🤖 **Chatbot Emotion Recognition**: Enhance chatbot interactions with emotion-aware responses.  
💬 **Customer Feedback Analysis**: Gain insights into user sentiment in business applications.  
🧠 **Mental Health Monitoring**: Assist in detecting emotional patterns in text-based conversations.  

---
## **Contributing**  
Contributions are welcome! If you'd like to improve the project, feel free to **fork the repository** and submit a **pull request**.

---
## **Contact**  
For any questions or suggestions, feel free to reach out! 😊

