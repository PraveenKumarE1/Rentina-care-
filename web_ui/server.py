import os
import hashlib
import csv
import sqlite3
import re
import shutil
import threading
from datetime import datetime, timezone
from flask import Flask, request, jsonify, send_from_directory, session
from werkzeug.utils import secure_filename

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import python_dr_runner

WEB_UI_FOLDER = os.path.dirname(__file__)
app = Flask(__name__, static_folder=WEB_UI_FOLDER, static_url_path='/static')
app.secret_key = os.environ.get('RETINACARE_SESSION_SECRET', 'local-development-secret-change-me')
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
DATABASE_PATH = os.path.join(PROJECT_ROOT, 'database', 'upload_history.sqlite3')
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
RESULT_CACHE = {}
RESULT_DETAILS_CACHE = {}
TRAINING_DIR = os.path.join(PROJECT_ROOT, 'datasets', 'user_training')
FEEDBACK_DIR = os.path.join(PROJECT_ROOT, 'datasets', 'reviewer_feedback')
TRAINING_STATUS = {"state": "idle", "message": "No training has been started."}


def initialize_database():
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                email TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                date_of_birth TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS upload_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                image_hash TEXT NOT NULL,
                uploaded_at TEXT NOT NULL,
                width INTEGER,
                height INTEGER,
                screening_result TEXT,
                confidence REAL,
                referable INTEGER,
                quality_passed INTEGER,
                focus REAL,
                contrast REAL,
                signal_to_noise REAL,
                feedback TEXT,
                microaneurysms INTEGER,
                exudates INTEGER,
                haemorrhages INTEGER
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS reviewer_corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                filename TEXT NOT NULL,
                image_hash TEXT NOT NULL,
                corrected_grade INTEGER NOT NULL,
                corrected_label TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


def save_upload_record(filename, image_key, result):
    quality = result["quality"]
    grading = result["grading"]
    segmentation = result["segmentation"]
    image = result["image"]
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("""
            INSERT INTO upload_history (
                filename, image_hash, uploaded_at, width, height, screening_result,
                confidence, referable, quality_passed, focus, contrast, signal_to_noise,
                feedback, microaneurysms, exudates, haemorrhages
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            filename, image_key, datetime.now(timezone.utc).isoformat(),
            int(image.shape[1]), int(image.shape[0]), grading.get("label"),
            float(grading.get("confidence", 0.0)), int(bool(grading.get("referable", False))),
            int(quality.passed), float(quality.focus), float(quality.contrast),
            float(quality.snr), "; ".join(quality.feedback),
            int(segmentation["microaneurysms"]["count"]),
            int(segmentation["exudates"]["count"]),
            int(segmentation["haemorrhages"]["count"]),
        ))


def get_upload_history(limit=5):
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("""
            SELECT id, filename, uploaded_at, width, height, screening_result,
                   confidence, referable, quality_passed, feedback,
                   microaneurysms, exudates, haemorrhages
            FROM upload_history ORDER BY id DESC LIMIT ?
        """, (limit,)).fetchall()
    return [dict(row) for row in rows]


def save_reviewer_correction(email, filename, image_key, corrected_grade):
    labels = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"]
    os.makedirs(os.path.join(FEEDBACK_DIR, 'train_images'), exist_ok=True)
    source_path = os.path.join(UPLOAD_FOLDER, filename)
    feedback_filename = f"{image_key[:16]}_{secure_filename(filename)}"
    feedback_path = os.path.join(FEEDBACK_DIR, 'train_images', feedback_filename)
    shutil.copyfile(source_path, feedback_path)
    csv_path = os.path.join(FEEDBACK_DIR, 'train.csv')
    needs_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        if needs_header:
            writer.writerow(["id_code", "diagnosis"])
        writer.writerow([os.path.splitext(feedback_filename)[0], corrected_grade])
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("""
            INSERT INTO reviewer_corrections
            (email, filename, image_hash, corrected_grade, corrected_label, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (email, filename, image_key, corrected_grade, labels[corrected_grade], datetime.now(timezone.utc).isoformat()))


initialize_database()


def login_required():
    if not session.get('email'):
        return jsonify({"error": "Please sign in first."}), 401
    return None


def get_profile(email):
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT email, full_name, date_of_birth FROM user_profiles WHERE email = ?",
            (email,),
        ).fetchone()
    return dict(row) if row else None


def save_profile(email, full_name, date_of_birth):
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("""
            INSERT INTO user_profiles (email, full_name, date_of_birth, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(email) DO UPDATE SET
                full_name = excluded.full_name,
                date_of_birth = excluded.date_of_birth,
                updated_at = excluded.updated_at
        """, (email, full_name, date_of_birth, now, now))


def build_personal_guide(email, language='en'):
    profile = get_profile(email)
    uploads = get_upload_history(10)
    name = profile["full_name"] if profile else "there"
    latest_result = uploads[0]["screening_result"] if uploads else ""
    result_names = {
        'en': {'Severe NPDR': 'Severe diabetic retinal changes', 'Moderate NPDR': 'Moderate diabetic retinal changes', 'Mild NPDR': 'Mild diabetic retinal changes', 'No DR': 'No visible diabetic retinal changes', 'Ungradable': 'Image needs to be retaken'},
        'hi': {'Severe NPDR': 'डायबिटिक रेटिना में गंभीर बदलाव', 'Moderate NPDR': 'डायबिटिक रेटिना में मध्यम बदलाव', 'Mild NPDR': 'डायबिटिक रेटिना में हल्के बदलाव', 'No DR': 'डायबिटिक रेटिना में कोई दिखाई देने वाला बदलाव नहीं', 'Ungradable': 'चित्र दोबारा लेना आवश्यक है'},
        'ta': {'Severe NPDR': 'விழித்திரையில் கடுமையான நீரிழிவு மாற்றங்கள்', 'Moderate NPDR': 'விழித்திரையில் மிதமான நீரிழிவு மாற்றங்கள்', 'Mild NPDR': 'விழித்திரையில் லேசான நீரிழிவு மாற்றங்கள்', 'No DR': 'விழித்திரையில் தெளிவான நீரிழிவு மாற்றங்கள் இல்லை', 'Ungradable': 'படத்தை மீண்டும் எடுக்க வேண்டும்'},
        'te': {'Severe NPDR': 'రెటీనా లో తీవ్రమైన డయాబెటిక్ మార్పులు', 'Moderate NPDR': 'రెటీనా లో మితమైన డయాబెటిక్ మార్పులు', 'Mild NPDR': 'రెటీనా లో స్వల్ప డయాబెటిక్ మార్పులు', 'No DR': 'రెటీనా లో కనిపించే డయాబెటిక్ మార్పులు లేవు', 'Ungradable': 'చిత్రాన్ని మళ్లీ తీయాలి'},
        'ml': {'Severe NPDR': 'റെറ്റിനയിൽ ഗുരുതരമായ പ്രമേഹ മാറ്റങ്ങൾ', 'Moderate NPDR': 'റെറ്റിനയിൽ മിതമായ പ്രമേഹ മാറ്റങ്ങൾ', 'Mild NPDR': 'റെറ്റിനയിൽ നേരിയ പ്രമേഹ മാറ്റങ്ങൾ', 'No DR': 'റെറ്റിനയിൽ കാണാവുന്ന പ്രമേഹ മാറ്റങ്ങളില്ല', 'Ungradable': 'ചിത്രം വീണ്ടും എടുക്കണം'},
        'kn': {'Severe NPDR': 'ರೆಟಿನಾದಲ್ಲಿ ತೀವ್ರ ಮಧುಮೇಹ ಬದಲಾವಣೆಗಳು', 'Moderate NPDR': 'ರೆಟಿನಾದಲ್ಲಿ ಮಧ್ಯಮ ಮಧುಮೇಹ ಬದಲಾವಣೆಗಳು', 'Mild NPDR': 'ರೆಟಿನಾದಲ್ಲಿ ಸೌಮ್ಯ ಮಧುಮೇಹ ಬದಲಾವಣೆಗಳು', 'No DR': 'ರೆಟಿನಾದಲ್ಲಿ ಗೋಚರಿಸುವ ಮಧುಮೇಹ ಬದಲಾವಣೆಗಳಿಲ್ಲ', 'Ungradable': 'ಚಿತ್ರವನ್ನು ಮತ್ತೆ ತೆಗೆದುಕೊಳ್ಳಬೇಕು'},
    }
    language = language if language in {'en', 'hi', 'ta', 'te', 'ml', 'kn'} else 'en'
    result_text = result_names[language].get(latest_result, latest_result)
    if not uploads:
        first_steps = {
            'en': ['PERSONAL GUIDE FOR', 'Step 1: Upload a clear retinal image or use a supported retinal camera.', 'Step 2: Press Run screening and wait for the result.', 'Step 3: Share the result with your doctor; this tool cannot confirm diabetes.'],
            'hi': ['व्यक्तिगत मार्गदर्शिका:', 'चरण 1: एक स्पष्ट रेटिना चित्र अपलोड करें या रेटिना कैमरे का उपयोग करें।', 'चरण 2: स्क्रीनिंग चलाएँ और परिणाम की प्रतीक्षा करें।', 'चरण 3: परिणाम डॉक्टर के साथ साझा करें; यह उपकरण मधुमेह की पुष्टि नहीं कर सकता।'],
            'ta': ['தனிப்பட்ட வழிகாட்டி:', 'படி 1: தெளிவான விழித்திரை படத்தைப் பதிவேற்றவும் அல்லது விழித்திரை கேமராவைப் பயன்படுத்தவும்.', 'படி 2: திரையிடலை இயக்கி முடிவுக்காக காத்திருக்கவும்.', 'படி 3: முடிவை மருத்துவரிடம் பகிரவும்; இந்த கருவி நீரிழிவு நோயை உறுதிப்படுத்தாது.'],
            'te': ['వ్యక్తిగత మార్గదర్శి:', 'దశ 1: స్పష్టమైన రెటీనా చిత్రాన్ని అప్‌లోడ్ చేయండి లేదా రెటీనా కెమెరాను ఉపయోగించండి.', 'దశ 2: స్క్రీనింగ్ ప్రారంభించి ఫలితం కోసం వేచి ఉండండి.', 'దశ 3: ఫలితాన్ని వైద్యుడితో పంచుకోండి; ఈ సాధనం డయాబెటిస్‌ను నిర్ధారించదు.'],
            'ml': ['വ്യക്തിഗത ഗൈഡ്:', 'ഘട്ടം 1: വ്യക്തമായ റെറ്റിന ചിത്രം അപ്‌ലോഡ് ചെയ്യുക അല്ലെങ്കിൽ റെറ്റിന ക്യാമറ ഉപയോഗിക്കുക.', 'ഘട്ടം 2: സ്ക്രീനിംഗ് നടത്തി ഫലത്തിനായി കാത്തിരിക്കുക.', 'ഘട്ടം 3: ഫലം ഡോക്ടറുമായി പങ്കിടുക; ഈ ഉപകരണം പ്രമേഹം സ്ഥിരീകരിക്കില്ല.'],
            'kn': ['ವೈಯಕ್ತಿಕ ಮಾರ್ಗದರ್ಶಿ:', 'ಹಂತ 1: ಸ್ಪಷ್ಟವಾದ ರೆಟಿನಾ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ ಅಥವಾ ರೆಟಿನಾ ಕ್ಯಾಮೆರಾ ಬಳಸಿ.', 'ಹಂತ 2: ಸ್ಕ್ರೀನಿಂಗ್ ನಡೆಸಿ ಫಲಿತಾಂಶಕ್ಕಾಗಿ ಕಾಯಿರಿ.', 'ಹಂತ 3: ಫಲಿತಾಂಶವನ್ನು ವೈದ್ಯರೊಂದಿಗೆ ಹಂಚಿಕೊಳ್ಳಿ; ಈ ಸಾಧನವು ಮಧುಮೇಹವನ್ನು ದೃಢೀಕರಿಸುವುದಿಲ್ಲ.'],
        }
        return "\n".join([first_steps[language][0] + ' ' + name.upper(), *first_steps[language][1:]])
    latest = uploads[0]
    referable_count = sum(bool(item["referable"]) for item in uploads)
    if latest["screening_result"] == "Ungradable":
        next_step = "Retake the image with a proper retinal camera and ask an eye specialist to review it."
    elif latest["referable"]:
        next_step = "Arrange a prompt ophthalmology appointment and take the screening report with you."
    else:
        next_step = "Continue routine diabetes and eye-care follow-up and keep future screenings."
    guide_templates = {
        'en': ['PERSONAL GUIDE FOR', 'Step 1: Review your latest screening result.', 'Step 2: Your latest retinal result is {result}.', 'Step 3: You have {count} screening result(s) needing specialist attention in your saved history.', 'Step 4: {next_step}', 'Step 5: Confirm diabetes with HbA1c or blood-glucose testing; a retinal image alone cannot confirm diabetes.', 'This guide supports a conversation with a clinician and does not prescribe treatment.'],
        'hi': ['व्यक्तिगत मार्गदर्शिका:', 'चरण 1: अपनी नवीनतम स्क्रीनिंग का परिणाम देखें।', 'चरण 2: आपकी नवीनतम रेटिना जाँच का परिणाम {result} है।', 'चरण 3: आपके इतिहास में {count} परिणाम विशेषज्ञ की जाँच की आवश्यकता बताते हैं।', 'चरण 4: नेत्र विशेषज्ञ से जल्द अपॉइंटमेंट लें और रिपोर्ट साथ ले जाएँ।', 'चरण 5: HbA1c या रक्त-शर्करा जाँच से मधुमेह की पुष्टि करें; रेटिना चित्र अकेले मधुमेह की पुष्टि नहीं कर सकता।', 'यह मार्गदर्शिका डॉक्टर से बातचीत में सहायता करती है और उपचार नहीं बताती।'],
        'ta': ['தனிப்பட்ட வழிகாட்டி:', 'படி 1: உங்கள் சமீபத்திய திரையிடல் முடிவைப் பாருங்கள்.', 'படி 2: உங்கள் சமீபத்திய விழித்திரை முடிவு {result}.', 'படி 3: உங்கள் வரலாற்றில் {count} முடிவுகளுக்கு நிபுணர் கவனம் தேவை.', 'படி 4: கண் நிபுணரை விரைவில் சந்தித்து அறிக்கையை எடுத்துச் செல்லுங்கள்.', 'படி 5: HbA1c அல்லது இரத்த சர்க்கரை பரிசோதனை மூலம் நீரிழிவு நோயை உறுதிப்படுத்துங்கள்; விழித்திரை படம் மட்டும் போதாது.', 'இந்த வழிகாட்டி மருத்துவருடன் பேச உதவும்; சிகிச்சையை பரிந்துரைக்காது.'],
        'te': ['వ్యక్తిగత మార్గదర్శి:', 'దశ 1: మీ తాజా స్క్రీనింగ్ ఫలితాన్ని చూడండి.', 'దశ 2: మీ తాజా రెటీనా ఫలితం {result}.', 'దశ 3: మీ చరిత్రలో {count} ఫలితాలకు నిపుణుల శ్రద్ధ అవసరం.', 'దశ 4: కంటి నిపుణుడిని త్వరగా కలసి నివేదికను తీసుకెళ్లండి.', 'దశ 5: HbA1c లేదా రక్తంలో చక్కెర పరీక్షతో డయాబెటిస్‌ను నిర్ధారించండి; రెటీనా చిత్రం మాత్రమే సరిపోదు.', 'ఈ మార్గదర్శి వైద్యుడితో మాట్లాడటానికి సహాయపడుతుంది; చికిత్సను సూచించదు.'],
        'ml': ['വ്യക്തിഗത ഗൈഡ്:', 'ഘട്ടം 1: നിങ്ങളുടെ ഏറ്റവും പുതിയ സ്ക്രീനിംഗ് ഫലം പരിശോധിക്കുക.', 'ഘട്ടം 2: നിങ്ങളുടെ ഏറ്റവും പുതിയ റെറ്റിന ഫലം {result} ആണ്.', 'ഘട്ടം 3: നിങ്ങളുടെ ചരിത്രത്തിലെ {count} ഫലങ്ങൾക്ക് വിദഗ്ധ ശ്രദ്ധ ആവശ്യമാണ്.', 'ഘട്ടം 4: കണ്ണ് വിദഗ്ധനെ ഉടൻ കാണുകയും റിപ്പോർട്ട് കൊണ്ടുപോകുകയും ചെയ്യുക.', 'ഘട്ടം 5: HbA1c അല്ലെങ്കിൽ രക്തത്തിലെ പഞ്ചസാര പരിശോധനയിലൂടെ പ്രമേഹം സ്ഥിരീകരിക്കുക; റെറ്റിന ചിത്രം മാത്രം മതിയാകില്ല.', 'ഈ ഗൈഡ് ഡോക്ടറുമായി സംസാരിക്കാൻ സഹായിക്കുന്നു; ചികിത്സ നിർദ്ദേശിക്കുന്നില്ല.'],
        'kn': ['ವೈಯಕ್ತಿಕ ಮಾರ್ಗದರ್ಶಿ:', 'ಹಂತ 1: ನಿಮ್ಮ ಇತ್ತೀಚಿನ ಸ್ಕ್ರೀನಿಂಗ್ ಫಲಿತಾಂಶವನ್ನು ಪರಿಶೀಲಿಸಿ.', 'ಹಂತ 2: ನಿಮ್ಮ ಇತ್ತೀಚಿನ ರೆಟಿನಾ ಫಲಿತಾಂಶ {result}.', 'ಹಂತ 3: ನಿಮ್ಮ ಇತಿಹಾಸದಲ್ಲಿನ {count} ಫಲಿತಾಂಶಗಳಿಗೆ ತಜ್ಞರ ಗಮನ ಅಗತ್ಯವಿದೆ.', 'ಹಂತ 4: ಕಣ್ಣಿನ ತಜ್ಞರನ್ನು ಶೀಘ್ರವಾಗಿ ಭೇಟಿ ಮಾಡಿ ವರದಿಯನ್ನು ತೆಗೆದುಕೊಂಡು ಹೋಗಿ.', 'ಹಂತ 5: HbA1c ಅಥವಾ ರಕ್ತದ ಸಕ್ಕರೆ ಪರೀಕ್ಷೆಯಿಂದ ಮಧುಮೇಹವನ್ನು ದೃಢೀಕರಿಸಿ; ರೆಟಿನಾ ಚಿತ್ರ ಮಾತ್ರ ಸಾಕಾಗುವುದಿಲ್ಲ.', 'ಈ ಮಾರ್ಗದರ್ಶಿ ವೈದ್ಯರೊಂದಿಗೆ ಮಾತನಾಡಲು ಸಹಾಯ ಮಾಡುತ್ತದೆ; ಚಿಕಿತ್ಸೆಯನ್ನು ಸೂಚಿಸುವುದಿಲ್ಲ.'],
    }
    lines = guide_templates[language]
    return "\n".join([lines[0] + ' ' + name.upper(), lines[1], lines[2].format(result=result_text), lines[3].format(count=referable_count), lines[4].format(next_step=next_step), lines[5], lines[6]])


def build_case_review(result):
    grading = result["grading"]
    quality = result["quality"]
    segmentation = result["segmentation"]
    label = grading["label"]
    confidence = float(grading.get("confidence", 0.0))
    method = result.get('model_method', 'interpretable rule-based baseline')
    if not quality.passed:
        review_note = "No severity prediction is provided because this image did not pass quality checks."
    elif confidence < 0.70:
        review_note = "Confidence is limited. A clinician should review the original image before relying on this screening result."
    else:
        review_note = "This is a screening estimate and still requires clinician confirmation."
    finding_names = {
        "No DR": "No visible diabetic retinal changes",
        "Mild NPDR": "Mild diabetic retinal changes",
        "Moderate NPDR": "Moderate diabetic retinal changes",
        "Severe NPDR": "Severe diabetic retinal changes",
        "PDR": "Advanced diabetic retinal changes",
        "Ungradable": "Image needs to be retaken",
    }
    if label == "Ungradable":
        symptoms = "No symptom assessment is possible until a gradable image is obtained."
        action = "Retake or upload a properly acquired retinal fundus image."
    elif label == "No DR":
        symptoms = "Diabetic retinopathy may have no symptoms in its early stages."
        action = "Continue routine diabetes and eye-care follow-up."
    elif label == "Mild NPDR":
        symptoms = "Often no symptoms; blurred vision or small changes may occur."
        action = "Arrange a comprehensive eye examination and diabetes follow-up."
    elif label == "Moderate NPDR":
        symptoms = "Blurred or fluctuating vision may occur, but symptoms can still be absent."
        action = "Prompt ophthalmology review is recommended."
    elif label == "Severe NPDR":
        symptoms = "Blurred vision, floaters, or difficulty seeing may occur; urgent symptoms need care."
        action = "Prompt ophthalmology referral is recommended."
    else:
        symptoms = "Floaters, blurred vision, shadows, or sudden vision loss may occur."
        action = "Urgent ophthalmology assessment is recommended."

    diabetes_status = (
        "This image cannot confirm diabetes. A clinician must review symptoms, history, and blood tests."
        if quality.passed
        else "The image is not clear enough to review."
    )
    diabetes_risk = result.get("diabetes_risk", {})
    diabetes_risk_text = diabetes_risk.get("risk") or diabetes_risk.get("message", "Diabetes risk prediction is unavailable.")
    if label == "Ungradable":
        detail = "The retinal image did not meet the quality check, so disease severity cannot be assessed."
        tests = "Retake the fundus image; if diabetes is suspected, arrange medical review and blood-glucose testing."
        follow_up = "Do not use this image to make a treatment decision."
    else:
        detail = (
            f"The screening baseline detected approximately {segmentation['haemorrhages']['count']:,} "
            f"dark lesion candidates and {segmentation['exudates']['count']:,} bright lesion candidates. "
            "These are automated image findings and require ophthalmologist confirmation."
        )
        tests = "Ask a doctor for HbA1c and fasting or random plasma glucose testing to check diabetes status."
        follow_up = "Arrange a comprehensive dilated eye examination promptly; do not wait for symptoms."
    return "\n".join([
        f"SCREENING RESULT: {finding_names.get(label, label)}",
        f"MODEL: {method}",
        f"MODEL CONFIDENCE: {confidence:.0%}. {review_note}",
        f"DIABETES STATUS: {diabetes_status}",
        f"RETINAL-IMAGE DIABETES RISK: {diabetes_risk_text}",
        f"WHAT THIS MEANS: {detail}",
        f"POSSIBLE SYMPTOMS: {symptoms}",
        f"CONFIRM DIABETES: {tests}",
        f"RECOMMENDED NEXT STEP: {follow_up} {action}",
        "URGENT CARE: Seek immediate eye care for sudden vision loss, a curtain or shadow, many new floaters, flashes, severe eye pain, or rapidly worsening vision.",
        "IMPORTANT: This development baseline cannot prescribe medicine and does not replace a doctor or eye specialist.",
    ])


def build_result_details(result):
    """Return UI-safe model and quality metadata without image masks or arrays."""
    quality = result["quality"]
    grading = result["grading"]
    confidence = float(grading.get("confidence", 0.0))
    if not quality.passed:
        review_note = "Retake the image before any disease severity assessment."
    elif confidence < 0.70:
        review_note = "Limited confidence: clinician confirmation is especially important."
    else:
        review_note = "Screening support only; clinician confirmation is required."
    return {
        "model_method": result.get("model_method", "interpretable rule-based baseline"),
        "confidence": confidence,
        "review_note": review_note,
        "quality": {
            "passed": bool(quality.passed),
            "feedback": list(quality.feedback),
            "focus": round(float(quality.focus), 1),
            "contrast": round(float(quality.contrast), 3),
            "field_of_view": round(float(quality.field_of_view), 3),
            "snr": round(float(quality.snr), 2),
        },
        "diabetes_risk": result.get("diabetes_risk", {}),
    }

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/login', methods=['POST'])
def login():
    email = str(request.json.get('email', '')).strip().lower() if request.is_json else ''
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        return jsonify({"error": "Enter a valid email address."}), 400
    session['email'] = email
    return jsonify({"email": email})


@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.route('/me')
def me():
    email = session.get('email')
    return jsonify({"email": email, "profile": get_profile(email) if email else None})


@app.route('/profile', methods=['POST'])
def profile():
    auth_error = login_required()
    if auth_error:
        return auth_error
    payload = request.get_json(silent=True) or {}
    full_name = str(payload.get('full_name', '')).strip()
    date_of_birth = str(payload.get('date_of_birth', '')).strip()
    if len(full_name) < 2 or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_of_birth):
        return jsonify({"error": "Enter a name and date of birth."}), 400
    save_profile(session['email'], full_name, date_of_birth)
    return jsonify({"profile": get_profile(session['email'])})


@app.route('/personal-guide')
def personal_guide():
    auth_error = login_required()
    if auth_error:
        return auth_error
    language = request.args.get('language', 'en')
    return jsonify({"guide": build_personal_guide(session['email'], language)})


@app.route('/model-status')
def model_status():
    checkpoint_available = os.path.exists(python_dr_runner.MODEL_CHECKPOINT)
    diabetes_checkpoint_available = os.path.exists(python_dr_runner.DIABETES_MODEL_CHECKPOINT)
    return jsonify({
        "trained_checkpoint": checkpoint_available,
        "diabetes_risk_checkpoint": diabetes_checkpoint_available,
        "message": "Trained PyTorch CNN active." if checkpoint_available else "Rule-based development baseline active. Train a labeled checkpoint to enable the CNN.",
    })


@app.route('/teach-model', methods=['POST'])
def teach_model():
    auth_error = login_required()
    if auth_error:
        return auth_error
    payload = request.get_json(silent=True) or {}
    filename = secure_filename(str(payload.get('filename', '')))
    image_key = str(payload.get('image_hash', ''))
    try:
        corrected_grade = int(payload.get('corrected_grade'))
    except (TypeError, ValueError):
        return jsonify({"error": "Choose a confirmed DR grade from 0 to 4."}), 400
    if corrected_grade not in range(5) or not filename or not image_key:
        return jsonify({"error": "Choose a confirmed DR grade for the uploaded image."}), 400
    source_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(source_path):
        return jsonify({"error": "The uploaded image is no longer available."}), 404
    save_reviewer_correction(session['email'], filename, image_key, corrected_grade)
    return jsonify({"message": "Correction saved as a labeled training example for future retraining."})


@app.route('/history')
def history():
    auth_error = login_required()
    if auth_error:
        return auth_error
    return jsonify({"uploads": get_upload_history()})

@app.route('/run-model', methods=['POST'])
def run_model():
    auth_error = login_required()
    if auth_error:
        return auth_error
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(uploaded_file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    uploaded_file.save(save_path)

    try:
        with open(save_path, 'rb') as image_file:
            image_key = hashlib.sha256(image_file.read()).hexdigest()
        cached_output = RESULT_CACHE.get(image_key)
        if cached_output is not None:
            save_upload_record(filename, image_key, RESULT_DETAILS_CACHE[image_key])
            return jsonify({"output": cached_output, "details": build_result_details(RESULT_DETAILS_CACHE[image_key]), "cached": True, "filename": filename, "image_hash": image_key})

        result = python_dr_runner.run_pipeline(path=save_path)
        quality = result["quality"]
        grading = result["grading"]
        segmentation = result["segmentation"]
        output = "\n".join([
            build_case_review(result),
        ])
        RESULT_CACHE[image_key] = output
        RESULT_DETAILS_CACHE[image_key] = result
        save_upload_record(filename, image_key, result)
        return jsonify({"output": output, "details": build_result_details(result), "cached": False, "filename": filename, "image_hash": image_key})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route('/upload-training-data', methods=['POST'])
def upload_training_data():
    auth_error = login_required()
    if auth_error:
        return auth_error
    metadata = request.files.get('metadata')
    images = request.files.getlist('images')
    if metadata is None or not metadata.filename.lower().endswith('.csv'):
        return jsonify({"error": "Upload an APTOS-style CSV with id_code and diagnosis columns."}), 400

    import pandas as pd
    try:
        metadata.stream.seek(0)
        rows = pd.read_csv(metadata.stream)
    except Exception as exc:
        return jsonify({"error": f"Could not read CSV: {exc}"}), 400
    required = {"id_code", "diagnosis"}
    if not required.issubset(rows.columns):
        return jsonify({"error": "CSV must contain id_code and diagnosis columns."}), 400
    if rows.empty or not rows["diagnosis"].isin([0, 1, 2, 3, 4]).all():
        return jsonify({"error": "diagnosis values must be integers from 0 to 4."}), 400

    image_dir = os.path.join(TRAINING_DIR, 'train_images')
    os.makedirs(image_dir, exist_ok=True)
    csv_path = os.path.join(TRAINING_DIR, 'train.csv')
    metadata.stream.seek(0)
    with open(csv_path, 'wb') as target:
        shutil.copyfileobj(metadata.stream, target)
    for image in images:
        if image.filename and image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image.save(os.path.join(image_dir, secure_filename(image.filename)))

    return jsonify({
        "message": "Labeled training data saved. Check that every id_code has a matching image before training.",
        "rows": int(len(rows)),
        "images_uploaded": len(images),
        "path": TRAINING_DIR,
    })


@app.route('/training-status')
def training_status():
    auth_error = login_required()
    if auth_error:
        return auth_error
    return jsonify(TRAINING_STATUS)


def run_training_job(epochs):
    global TRAINING_STATUS
    TRAINING_STATUS = {"state": "running", "message": "Training on labeled retinal images."}
    try:
        csv_path = os.path.join(TRAINING_DIR, 'train.csv')
        image_dir = os.path.join(TRAINING_DIR, 'train_images')
        if not os.path.exists(csv_path) or not os.path.isdir(image_dir):
            raise FileNotFoundError("Upload train.csv and labeled retinal images first.")
        import subprocess
        trainer = os.path.join(PROJECT_ROOT, 'train_retinal_model.py')
        output = os.path.join(PROJECT_ROOT, 'models', 'user_retinal_cnn.pt')
        completed = subprocess.run(
            [sys.executable, trainer, '--data-dir', TRAINING_DIR, '--epochs', str(epochs), '--output', output],
            capture_output=True, text=True, check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or 'Training failed.')
        TRAINING_STATUS = {"state": "complete", "message": "Training complete.", "checkpoint": output, "log": completed.stdout[-2000:]}
    except Exception as exc:
        TRAINING_STATUS = {"state": "error", "message": str(exc)}


@app.route('/train-model', methods=['POST'])
def train_model():
    auth_error = login_required()
    if auth_error:
        return auth_error
    if TRAINING_STATUS["state"] == "running":
        return jsonify(TRAINING_STATUS), 409
    payload = request.get_json(silent=True) or {}
    epochs = max(1, min(int(payload.get('epochs', 5)), 50))
    thread = threading.Thread(target=run_training_job, args=(epochs,), daemon=True)
    thread.start()
    return jsonify({"state": "running", "message": "Training started."}), 202


if __name__ == '__main__':
    app.run(
        host=os.environ.get('HOST', '0.0.0.0'),
        port=int(os.environ.get('PORT', '5000')),
        debug=os.environ.get('FLASK_DEBUG', '').lower() == 'true',
    )
