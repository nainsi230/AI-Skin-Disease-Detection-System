import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from gtts import gTTS
import tempfile
import os
import textwrap
from reportlab.pdfgen import canvas
from datetime import datetime

st.set_page_config(
    page_title="AI Skin Disease Detection System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load AI model
model = load_model("skin_disease_model.h5")

# Supported classes
classes = [
    'Acne',
    'Actinic_Keratosis',
    'Benign_tumors',
    'Bullous',
    'Candidiasis',
    'DrugEruption',
    'Eczema',
    'Infestations_Bites',
    'Lichen',
    'Lupus',
    'Moles',
    'Psoriasis',
    'Rosacea',
    'Seborrh_Keratoses',
    'SkinCancer',
    'Sun_Sunlight_Damage',
    'Tinea',
    'Unknown_Normal',
    'Vascular_Tumors',
    'Vasculitis',
    'Vitiligo',
    'Warts'
]

# Disease information and recommendations
disease_info = {
    "Acne": {
        "description": "A common skin condition caused by clogged pores, excess oil, and inflammation leading to pimples, blackheads, and whiteheads.",
        "causes": ["Hormone changes during puberty or pregnancy", "Excess oil production", "Bacteria and clogged hair follicles"],
        "symptoms": ["Whiteheads or blackheads", "Inflamed red bumps", "Painful cysts or nodules"],
        "risk_factors": ["Teenage years", "Family history of acne", "Use of oily cosmetics"],
        "complications": ["Permanent scarring", "Dark spots", "Psychological distress"],
        "prevention": ["Cleanse your face gently twice daily", "Avoid heavy makeup and oily products", "Keep hair away from the face"],
        "precautions": ["Wash face regularly", "Avoid picking or squeezing lesions", "Choose oil-free skincare products"],
        "care": ["Use a gentle, non-comedogenic cleanser.", "Apply acne treatments as directed.", "Review treatments with a dermatologist if the condition worsens."],
        "when_to_consult": "See a dermatologist if acne is painful, widespread, or not improving with over-the-counter care."
    },
    "Actinic_Keratosis": {
        "description": "A rough, scaly patch on sun-exposed skin, often caused by long-term ultraviolet radiation exposure.",
        "causes": ["Chronic sun exposure", "UV radiation damage", "Fair skin that burns easily"],
        "symptoms": ["Dry, rough spots", "Scaly or crusty patches", "Pink or red discoloration"],
        "risk_factors": ["Age over 40", "History of sunburns", "Use of tanning beds"],
        "complications": ["Progression to squamous cell carcinoma", "Persistent inflammation", "Skin irritation"],
        "prevention": ["Use broad-spectrum sunscreen daily", "Wear protective clothing", "Limit time in direct sunlight"],
        "precautions": ["Protect skin from the sun", "Avoid prolonged UV exposure", "Check your skin regularly for new patches"],
        "care": ["Apply sunblock every day.", "Use moisturizers on dry patches.", "Schedule a skin check if lesions persist."],
        "when_to_consult": "Consult a dermatologist when new or changing rough patches appear, especially on sun-exposed areas."
    },
    "Benign_tumors": {
        "description": "Noncancerous growths in the skin that may appear as lumps, bumps, or raised spots but generally do not spread.",
        "causes": ["Inherited genetic factors", "Age-related skin changes", "Local skin irritation"],
        "symptoms": ["Soft or firm bump", "Slow growth", "Usually painless"],
        "risk_factors": ["Aging", "Family history of benign skin growths", "Previous skin injury"],
        "complications": ["Irritation from clothing", "Cosmetic concerns", "Rare infection if damaged"],
        "prevention": ["Monitor new skin growths", "Avoid unnecessary skin trauma", "Maintain a healthy skin care routine"],
        "precautions": ["Keep skin clean around the growth", "Avoid squeezing or scratching it", "Ask a doctor about any growth that changes"],
        "care": ["Watch for rapid size changes.", "Keep the area protected.", "Follow your dermatologist’s advice on removal if needed."],
        "when_to_consult": "Have a dermatologist evaluate growths that change color, size, or shape, or cause pain or bleeding."
    },
    "Bullous": {
        "description": "A group of skin conditions characterized by large fluid-filled blisters that can appear suddenly and may be fragile.",
        "causes": ["Autoimmune reactions", "Infections", "Medication sensitivity"],
        "symptoms": ["Large clear blisters", "Painful or itchy skin", "Blister rupture and raw areas"],
        "risk_factors": ["Advanced age", "Autoimmune disease history", "Recent medication changes"],
        "complications": ["Infection", "Fluid loss", "Scarring"],
        "prevention": ["Protect blistered skin from friction", "Avoid known medication triggers", "Practice gentle skin care"],
        "precautions": ["Avoid popping blisters", "Keep sores clean and covered", "Seek treatment for signs of infection"],
        "care": ["Use mild cleansers.", "Apply protective dressings.", "Follow medical recommendations for blister care."],
        "when_to_consult": "Consult a doctor immediately for widespread blistering, intense pain, or signs of infection."
    },
    "Candidiasis": {
        "description": "A fungal infection caused by Candida yeast, often affecting warm, moist skin areas with redness and irritation.",
        "causes": ["Yeast overgrowth", "Warm, damp skin", "Weakened immune system"],
        "symptoms": ["Red, itchy rash", "White patches or scaling", "Burning sensation"],
        "risk_factors": ["Diabetes", "Antibiotic use", "Excessive sweating"],
        "complications": ["Spread to nearby skin areas", "Secondary bacterial infection", "Chronic discomfort"],
        "prevention": ["Keep affected areas dry", "Change damp clothing quickly", "Use breathable fabrics"],
        "precautions": ["Avoid tight, non-breathable clothing", "Dry skin thoroughly after bathing", "See a provider if symptoms worsen"],
        "care": ["Use antifungal products as directed.", "Maintain good hygiene.", "Avoid hot, humid environments when possible."],
        "when_to_consult": "See a healthcare professional if the rash persists despite over-the-counter antifungal treatment."
    },
    "DrugEruption": {
        "description": "A skin reaction to medication that can produce rashes, redness, or itching and sometimes develops suddenly.",
        "causes": ["Allergic reaction to a medicine", "Sensitivity to a new drug", "Interaction between medications"],
        "symptoms": ["Widespread rash", "Itching or burning", "Swelling and redness"],
        "risk_factors": ["Multiple medications", "History of drug reactions", "Family history of drug allergies"],
        "complications": ["Severe allergic reactions", "Skin peeling", "Systemic symptoms"],
        "prevention": ["Inform doctors of all current medications", "Avoid known drug allergens", "Report suspicious reactions early"],
        "precautions": ["Stop the suspect drug if advised", "Monitor for breathing difficulty", "Seek prompt medical care for sudden rash"],
        "care": ["Use gentle skin care products.", "Keep the skin cool and moisturized.", "Follow medical guidance on medication changes."],
        "when_to_consult": "Consult a physician immediately if a new medication causes a sudden rash, swelling, or difficulty breathing."
    },
    "Eczema": {
        "description": "A chronic inflammatory skin condition that causes dry, itchy, and inflamed patches of skin.",
        "causes": ["Genetic predisposition", "Immune system sensitivity", "Environmental triggers"],
        "symptoms": ["Itchy, red patches", "Cracked or scaly skin", "Blisters or oozing"],
        "risk_factors": ["Family history of eczema or allergies", "Dry skin", "Exposure to irritants"],
        "complications": ["Skin infections", "Thickened skin", "Sleep disturbance due to itching"],
        "prevention": ["Use fragrance-free moisturizers", "Avoid known triggers", "Maintain gentle skin care"],
        "precautions": ["Keep skin hydrated", "Avoid harsh soaps", "Wear soft cotton clothing"],
        "care": ["Use fragrance-free moisturizers.", "Take lukewarm showers.", "Avoid known irritants and fragrances."],
        "when_to_consult": "See a dermatologist if eczema is severe, widespread, or not controlled with basic skin care."
    },
    "Infestations_Bites": {
        "description": "Skin irritation and rash caused by insect bites or small parasitic infestations, often leading to redness and itching.",
        "causes": ["Mosquito or flea bites", "Chigger exposure", "Scabies or lice infestations"],
        "symptoms": ["Localized itching", "Red bumps or welts", "Small blisters or burrow tracks"],
        "risk_factors": ["Outdoor exposure", "Contact with infected people or animals", "Warm, humid conditions"],
        "complications": ["Secondary infection", "Allergic reaction", "Persistent itching"],
        "prevention": ["Use insect repellent", "Wear protective clothing outdoors", "Avoid sharing bedding or clothing"],
        "precautions": ["Do not scratch bite areas", "Keep the area clean", "See a provider if swelling increases"],
        "care": ["Apply cold compresses.", "Use soothing lotions.", "Treat any underlying infestation promptly."],
        "when_to_consult": "Seek medical care if bites spread, are very painful, or show signs of infection."
    },
    "Lichen": {
        "description": "A condition that causes small, flat-topped bumps or scaly patches on the skin, often with itching.",
        "causes": ["Immune system activity", "Stress", "Viral triggers"],
        "symptoms": ["Purple or pink flat bumps", "Itching", "Rash on wrists, ankles, or lower back"],
        "risk_factors": ["Middle age", "Stress", "Existing autoimmune conditions"],
        "complications": ["Skin discoloration", "Scarring from scratching", "Chronic itching"],
        "prevention": ["Manage stress", "Avoid irritating fabrics", "Maintain good skin care"],
        "precautions": ["Avoid scratching affected areas", "Use mild skin products", "See a doctor if the rash changes"],
        "care": ["Use gentle, fragrance-free skincare.", "Apply topical treatment as prescribed.", "Manage stress and avoid triggers."],
        "when_to_consult": "Consult a healthcare professional if the rash is persistent, painful, or widespread."
    },
    "Lupus": {
        "description": "An autoimmune condition that often affects the skin with rashes, lesions, and sensitivity to sunlight.",
        "causes": ["Immune system attacking healthy tissue", "Genetic predisposition", "Sun exposure"],
        "symptoms": ["Butterfly-shaped facial rash", "Photosensitive skin", "Red or purple lesions"],
        "risk_factors": ["Female gender", "Family history of autoimmune disease", "Exposure to sunlight"],
        "complications": ["Joint pain", "Kidney inflammation", "Fatigue"],
        "prevention": ["Use high-SPF sunscreen", "Wear protective clothing", "Manage stress"],
        "precautions": ["Limit sun exposure", "Avoid harsh skincare products", "Consult a specialist for treatment"],
        "care": ["Keep sun-exposed skin protected.", "Use gentle cleansers.", "Follow prescribed treatment plans."],
        "when_to_consult": "See a specialist if you notice lupus-related rashes, joint pain, or increased sensitivity to sunlight."
    },
    "Moles": {
        "description": "Small skin growths caused by clusters of pigment-producing cells, typically benign but requiring monitoring for changes.",
        "causes": ["Genetics", "Sun exposure", "Hormonal changes"],
        "symptoms": ["Small brown or black spot", "Smooth or slightly raised bump", "Typically uniform color"],
        "risk_factors": ["Family history of moles", "Large number of moles", "Fair skin"],
        "complications": ["Rare malignant transformation", "Irritation from rubbing", "Cosmetic concern"],
        "prevention": ["Monitor moles regularly", "Use sunscreen", "Avoid excessive sun exposure"],
        "precautions": ["Watch for mole changes", "Avoid repetitive rubbing", "Ask a dermatologist about suspicious moles"],
        "care": ["Check moles monthly.", "Protect skin from sun.", "See a clinician for changing moles."],
        "when_to_consult": "Consult a dermatologist when a mole changes size, shape, color, or becomes itchy or bleeds."
    },
    "Psoriasis": {
        "description": "A chronic autoimmune condition that causes rapid skin cell growth, leading to red, scaly patches and discomfort.",
        "causes": ["Immune system overactivity", "Genetic predisposition", "Environmental triggers"],
        "symptoms": ["Red, scaly patches", "Itching or burning", "Dry, cracked skin that may bleed"],
        "risk_factors": ["Family history", "Stress", "Smoking"],
        "complications": ["Joint inflammation", "Infection", "Emotional distress"],
        "prevention": ["Moisturize regularly", "Avoid known triggers", "Manage stress"],
        "precautions": ["Keep skin moisturized", "Reduce stress", "Consult a dermatologist"],
        "care": ["Use gentle, soap-free cleansers.", "Apply emollients after bathing.", "Protect skin from cold, dry air."],
        "when_to_consult": "See a healthcare professional when psoriasis becomes painful, widespread, or interferes with daily life."
    },
    "Rosacea": {
        "description": "A chronic skin condition that causes facial redness, visible blood vessels, and sometimes acne-like bumps.",
        "causes": ["Blood vessel sensitivity", "Environmental triggers", "Genetics"],
        "symptoms": ["Facial redness", "Visible veins", "Small pustules"],
        "risk_factors": ["Fair skin", "Middle age", "Family history"],
        "complications": ["Eye irritation", "Thickened skin", "Persistent redness"],
        "prevention": ["Avoid trigger foods and drinks", "Protect skin from sun", "Use gentle skin care"],
        "precautions": ["Limit spicy foods and alcohol", "Avoid hot environments", "Use non-irritating skincare products"],
        "care": ["Use sunscreen daily.", "Choose rosacea-safe moisturizers.", "Avoid known triggers."],
        "when_to_consult": "Consult a dermatologist if facial redness or bumps persist or worsen."
    },
    "Seborrh_Keratoses": {
        "description": "A benign, often waxy or wart-like growth that appears on the skin and is usually harmless.",
        "causes": ["Age-related skin changes", "Genetics", "Sun exposure"],
        "symptoms": ["Waxy, raised bump", "Brown or tan coloration", "Often appears in multiple spots"],
        "risk_factors": ["Aging", "Family history", "Fair skin"],
        "complications": ["Irritation from friction", "Cosmetic concern", "Rare infection if scratched"],
        "prevention": ["Monitor growths for change", "Avoid trauma to affected areas", "Practice general skin care"],
        "precautions": ["Do not pick at growths", "Keep the area clean", "Discuss removal if it becomes irritated"],
        "care": ["Watch for changes.", "Protect skin from scraping.", "See a doctor if you are concerned."],
        "when_to_consult": "Ask a dermatologist if a growth changes rapidly, becomes painful, or bleeds."
    },
    "SkinCancer": {
        "description": "A serious condition where abnormal skin cells grow uncontrollably, often caused by ultraviolet radiation exposure.",
        "causes": ["Chronic sun exposure", "Tanning bed use", "Skin cell mutations"],
        "symptoms": ["A new spot or mole", "A sore that does not heal", "Changes in size, shape, or color"],
        "risk_factors": ["Fair skin", "History of sunburns", "Family history of skin cancer"],
        "complications": ["Spread to other tissues", "Tissue damage", "Life-threatening disease"],
        "prevention": ["Use broad-spectrum sunscreen", "Wear protective clothing", "Avoid tanning beds"],
        "precautions": ["Check your skin regularly", "Protect skin from UV exposure", "See a doctor for suspicious changes"],
        "care": ["Schedule regular skin exams.", "Apply sunscreen daily.", "Discuss any new lesions with a clinician."],
        "when_to_consult": "Seek medical evaluation for any suspicious or changing skin lesion as soon as possible."
    },
    "Sun_Sunlight_Damage": {
        "description": "Skin damage caused by prolonged sun exposure, including dryness, discoloration, and increased risk of more serious conditions.",
        "causes": ["Ultraviolet radiation", "Tanning without protection", "Cumulative sun exposure"],
        "symptoms": ["Red or tan patches", "Dry, leathery skin", "Dark spots or uneven pigmentation"],
        "risk_factors": ["Outdoor work", "Minimal sun protection", "Fair skin"],
        "complications": ["Premature aging", "Spots and discoloration", "Higher risk of skin cancer"],
        "prevention": ["Apply sunscreen every day", "Wear hats and long sleeves", "Seek shade during peak sun hours"],
        "precautions": ["Avoid intense midday sun", "Reapply sunscreen often", "Use protective clothing and sunglasses"],
        "care": ["Keep skin hydrated.", "Use moisturizers with antioxidants.", "Protect skin from further sun exposure."],
        "when_to_consult": "Talk to a skin specialist if sun-damaged areas are painful, blistered, or changing in appearance."
    },
    "Tinea": {
        "description": "A fungal infection of the skin that causes a ring-shaped rash and itching, commonly known as ringworm.",
        "causes": ["Dermatophyte fungi", "Direct skin contact", "Shared towels or clothing"],
        "symptoms": ["Ring-shaped red rash", "Itching", "Flaky or cracked skin"],
        "risk_factors": ["Warm, moist environments", "Close contact with infected people", "Weak immune system"],
        "complications": ["Spread to other body areas", "Secondary bacterial infection", "Chronic itching"],
        "prevention": ["Keep skin dry", "Avoid sharing personal items", "Wear breathable clothing"],
        "precautions": ["Wash hands frequently", "Use antifungal treatments as directed", "Avoid walking barefoot in public damp areas"],
        "care": ["Apply antifungal medications.", "Keep the affected area clean.", "Avoid tight clothing over the rash."],
        "when_to_consult": "See a provider if the rash spreads or does not improve with over-the-counter antifungal therapy."
    },
    "Unknown_Normal": {
        "description": "No active skin disease was detected. The skin appears within a normal and healthy range based on the image provided.",
        "causes": ["Normal skin variation", "Healthy skin tone and texture"],
        "symptoms": ["No abnormal rashes", "No lesions or significant discoloration", "Normal skin appearance"],
        "risk_factors": ["Healthy skin care routines", "No significant sun damage", "No known skin condition"],
        "complications": ["None specific", "Continue good skin care habits"],
        "prevention": ["Keep skin clean and hydrated", "Use sunscreen daily", "Maintain a balanced diet"],
        "precautions": ["Continue good hygiene", "Protect skin from sun", "Monitor your skin regularly"],
        "care": ["Use gentle skincare products.", "Protect skin from UV exposure.", "Maintain healthy lifestyle habits."],
        "when_to_consult": "If you notice new skin changes in the future, consult a dermatologist for a full evaluation."
    },
    "Vascular_Tumors": {
        "description": "Blood-vessel related growths in the skin that may appear red or purple and are usually benign but should be monitored.",
        "causes": ["Blood vessel proliferation", "Genetic factors", "Hormonal influences"],
        "symptoms": ["Red or purple raised lesions", "Soft skin growth", "May blanch when pressed"],
        "risk_factors": ["Infancy or childhood", "Hormonal changes", "Family history"],
        "complications": ["Bleeding if injured", "Infection if damaged", "Possible cosmetic concern"],
        "prevention": ["Monitor changes", "Protect vulnerable areas", "Avoid trauma to the lesion"],
        "precautions": ["Avoid squeezing or scratching", "Keep the area clean", "Seek medical advice for rapid growth"],
        "care": ["Protect lesions from injury.", "Use gentle skincare.", "Ask a clinician about treatment if it changes."],
        "when_to_consult": "Consult a dermatologist if a vascular growth bleeds easily or grows quickly."
    },
    "Vasculitis": {
        "description": "Inflammation of the blood vessels that can affect the skin and cause red or purple spots, sometimes with pain or tenderness.",
        "causes": ["Autoimmune inflammation", "Infection", "Medication reaction"],
        "symptoms": ["Red or purple spots", "Tender skin", "Possible ulceration"],
        "risk_factors": ["Autoimmune disease", "Recent infection", "Certain medications"],
        "complications": ["Skin ulcers", "Tissue damage", "Possible systemic involvement"],
        "prevention": ["Manage underlying health conditions", "Avoid triggers", "Follow medical advice for inflammatory conditions"],
        "precautions": ["Monitor skin changes closely", "Avoid trauma to affected areas", "Seek prompt care for new symptoms"],
        "care": ["Keep affected skin clean.", "Follow prescribed anti-inflammatory treatments.", "Protect skin from injury."],
        "when_to_consult": "See a healthcare provider if you develop painful or spreading skin spots or ulcers."
    },
    "Vitiligo": {
        "description": "A condition in which skin loses pigment in patches, causing areas of lighter skin that may gradually change over time.",
        "causes": ["Immune response against pigment cells", "Genetic predisposition", "Environmental triggers"],
        "symptoms": ["Patchy loss of skin color", "Irregular white patches", "Often starts on sun-exposed areas"],
        "risk_factors": ["Family history", "Autoimmune conditions", "Fair skin"],
        "complications": ["Sun sensitivity", "Social or emotional distress", "Dry or itchy depigmented skin"],
        "prevention": ["Protect depigmented skin from sun", "Use sunscreen regularly", "Avoid skin trauma"],
        "precautions": ["Use sunscreen", "Protect skin from injury", "Consult a dermatologist"],
        "care": ["Use broad-spectrum sunscreen daily.", "Wear protective clothing in sunlight.", "Avoid harsh skin treatments."],
        "when_to_consult": "Consult a dermatologist if the pigment changes rapidly or affects a large area of the body."
    },
    "Warts": {
        "description": "Small, rough growths caused by the human papillomavirus (HPV) that commonly appear on the hands, feet, or other skin surfaces.",
        "causes": ["Human papillomavirus infection", "Minor skin breaks", "Close contact with infected skin"],
        "symptoms": ["Rough, raised bump", "Small black dots inside the wart", "May be painful when pressed"],
        "risk_factors": ["Cut or damaged skin", "Weakened immune system", "Frequent exposure in communal areas"],
        "complications": ["Spread to adjacent skin", "Discomfort with pressure", "Persistent lesions"],
        "prevention": ["Avoid direct contact with warts", "Keep skin dry", "Do not share personal items"],
        "precautions": ["Avoid picking at warts", "Use protective coverings when needed", "Consult a doctor for persistent warts"],
        "care": ["Use wart removal products as directed.", "Keep the area clean.", "Seek medical advice for stubborn warts."],
        "when_to_consult": "See a healthcare provider if a wart is painful, changes in appearance, or persists despite treatment."
    }
}

# Supported languages
LANGUAGE_OPTIONS = {
    "English": "en",
    "Hindi": "hi"
}

TRANSLATIONS = {
    "en": {
        "navigation": "Navigation menu",
        "dashboard": "Dashboard",
        "scan_history": "Scan History",
        "model_info": "Model Info",
        "report": "AI Health Report",
        "select_language": "Select language",
        "model_information": "Model Information",
        "supported_diseases": "Supported diseases",
        "model_type": "Model type",
        "dataset": "Dataset",
        "last_update": "Last update",
        "why_trust": "Why Trust This System?",
        "project_details": "Project Details",
        "clinical_validation": "Clinical validation",
        "dermatologist_review": "Dermatologist review integration",
        "hospital_integration": "Hospital integration",
        "mobile_app_release": "Mobile app release",
        "version": "Version",
        "developer_info": "Developer information placeholder",
        "hero_title": "AI Skin Disease Detection System",
        "hero_subtitle": "Deep learning-based preliminary skin disease screening for educational and demo purposes. Upload an image to receive a fast prediction, confidence score, and summary report.",
        "feature_deep_learning": "Deep Learning Powered",
        "feature_categories": "22 Disease Categories",
        "feature_risk": "Instant Risk Assessment",
        "feature_report": "PDF Medical Report",
        "feature_voice": "Voice Assistant",
        "performance_overview": "Performance Overview",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1 Score",
        "why_trust_section": "Why trust this system?",
        "confidence_explanation": "Confidence score explanation",
        "confidence_explanation_text": "The confidence score represents the model's probability estimate for the predicted class. Higher confidence means the AI is more certain about its top prediction.",
        "project_roadmap": "Project Roadmap",
        "get_started": "Get started",
        "quick_guide": "Quick guide",
        "upload_image": "Upload a Skin Image",
        "choose_image_help": "Choose a clear photo of the affected skin area.",
        "prediction_result": "Prediction Result",
        "predicted_disease": "Predicted Disease",
        "confidence": "Confidence",
        "top_predictions": "Top 3 Predictions",
        "risk_assessment": "Risk Assessment",
        "purpose": "Purpose",
        "purpose_text": "Educational and preliminary screening only.",
        "disease_information": "Disease Information",
        "precautions": "Precautions",
        "care_recommendations": "Skin Care Recommendations",
        "expand_details": "View expanded details",
        "scan_history_title": "Scan History",
        "timestamp": "Timestamp",
        "severity": "Severity",
        "clear_history": "Clear history",
        "no_history": "No scan history yet.",
        "download_report": "Download Medical Report",
        "voice_assistant": "Read Result",
        "disclaimer": "This system does not replace professional medical diagnosis. Consult a dermatologist for medical decisions.",
        "severity_low": "Low Severity",
        "severity_moderate": "Moderate Severity",
        "severity_high": "High Severity",
        "recommendations_heading": "Personalized skin care recommendations",
        "report_summary": "AI Health Report Summary",
        "report_no_prediction": "No prediction available yet to generate a report.",
        "download_latest_report": "Download Latest Report",
        "learn_more_about_this_disease": "Learn More About This Disease",
        "sidebar_title": "Navigation menu",
        "model_info_title": "Model Information",
        "project_info_title": "Project Details",
        "developer_placeholder": "Developer: Your Name Here",
        "copyright": "© 2026 AI Skin Disease Detection System",
        "language_selector": "Language Selector"
    },
    "hi": {
        "navigation": "नेविगेशन मेनू",
        "dashboard": "डैशबोर्ड",
        "scan_history": "स्कैन इतिहास",
        "model_info": "मॉडल जानकारी",
        "report": "एआई हेल्थ रिपोर्ट",
        "select_language": "भाषा चुनें",
        "model_information": "मॉडल जानकारी",
        "supported_diseases": "समर्थित रोग",
        "model_type": "मॉडल प्रकार",
        "dataset": "डेटासेट",
        "last_update": "अंतिम अपडेट",
        "why_trust": "इस सिस्टम पर क्यों भरोसा करें?",
        "project_details": "परियोजना विवरण",
        "clinical_validation": "क्लिनिकल पुष्टि",
        "dermatologist_review": "त्वचा रोग विशेषज्ञ समीक्षा एकीकरण",
        "hospital_integration": "अस्पताल एकीकरण",
        "mobile_app_release": "मोबाइल ऐप रिलीज़",
        "version": "संस्करण",
        "developer_info": "डेवलपर जानकारी प्लेसहोल्डर",
        "hero_title": "एआई त्वचा रोग पहचान प्रणाली",
        "hero_subtitle": "शैक्षिक और डेमो उद्देश्यों के लिए डीप लर्निंग-आधारित प्रारंभिक त्वचा रोग स्क्रीनिंग। एक तेज़ पूर्वानुमान, आत्मविश्वास स्कोर और सारांश रिपोर्ट प्राप्त करने के लिए एक छवि अपलोड करें।",
        "feature_deep_learning": "डीप लर्निंग संचालित",
        "feature_categories": "22 रोग श्रेणियाँ",
        "feature_risk": "तत्काल जोखिम मूल्यांकन",
        "feature_report": "पीडीएफ मेडिकल रिपोर्ट",
        "feature_voice": "वॉयस असिस्टेंट",
        "performance_overview": "प्रदर्शन अवलोकन",
        "accuracy": "सटीकता",
        "precision": "प्रिसिजन",
        "recall": "रिकॉल",
        "f1_score": "एफ1 स्कोर",
        "why_trust_section": "इस सिस्टम पर भरोसा क्यों करें?",
        "confidence_explanation": "आत्मविश्वास स्कोर व्याख्या",
        "confidence_explanation_text": "आत्मविश्वास स्कोर मॉडल के अनुमान की संभावना को दर्शाता है। उच्च आत्मविश्वास का अर्थ है कि एआई अपने शीर्ष पूर्वानुमान के बारे में अधिक निश्चित है।",
        "project_roadmap": "परियोजना रोडमैप",
        "get_started": "शुरू करें",
        "quick_guide": "त्वरित मार्गदर्शिका",
        "upload_image": "त्वचा की छवि अपलोड करें",
        "choose_image_help": "प्रभावित त्वचा क्षेत्र की स्पष्ट फोटो चुनें।",
        "prediction_result": "पूर्वानुमान परिणाम",
        "predicted_disease": "पूर्वानुमानित रोग",
        "confidence": "आत्मविश्वास",
        "top_predictions": "शीर्ष 3 पूर्वानुमान",
        "risk_assessment": "जोखिम मूल्यांकन",
        "purpose": "उद्देश्य",
        "purpose_text": "केवल शैक्षिक और प्रारंभिक स्क्रीनिंग।",
        "disease_information": "रोग जानकारी",
        "precautions": "सावधानियां",
        "care_recommendations": "त्वचा देखभाल सिफ़ारिशें",
        "expand_details": "विस्तृत विवरण देखें",
        "scan_history_title": "स्कैन इतिहास",
        "timestamp": "समय टिकट",
        "severity": "गंभीरता",
        "clear_history": "इतिहास साफ़ करें",
        "no_history": "अभी तक कोई स्कैन इतिहास नहीं।",
        "download_report": "मेडिकल रिपोर्ट डाउनलोड करें",
        "voice_assistant": "परिणाम पढ़ें",
        "disclaimer": "यह सिस्टम पेशेवर चिकित्सा निदान को प्रतिस्थापित नहीं करता है। किसी भी चिकित्सा निर्णय के लिए त्वचा रोग विशेषज्ञ से परामर्श लें।",
        "severity_low": "कम गंभीरता",
        "severity_moderate": "मध्यम गंभीरता",
        "severity_high": "उच्च गंभीरता",
        "recommendations_heading": "व्यक्तिगत त्वचा देखभाल सिफ़ारिशें",
        "report_summary": "एआई हेल्थ रिपोर्ट सारांश",
        "report_no_prediction": "रिपोर्ट उत्पन्न करने के लिए अभी तक कोई पूर्वानुमान उपलब्ध नहीं है।",
        "download_latest_report": "नवीनतम रिपोर्ट डाउनलोड करें",
        "learn_more_about_this_disease": "इस रोग के बारे में और जानें",
        "sidebar_title": "नेविगेशन मेनू",
        "model_info_title": "मॉडल जानकारी",
        "project_info_title": "परियोजना विवरण",
        "developer_placeholder": "डेवलपर: आपका नाम यहाँ",
        "copyright": "© 2026 एआई त्वचा रोग पहचान प्रणाली",
        "language_selector": "भाषा चयनकर्ता"
    }
}


def t(key):
    return TRANSLATIONS.get(st.session_state.language, TRANSLATIONS["en"]).get(key, key)


def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "language" not in st.session_state:
        st.session_state.language = "en"
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None


def speak(text: str):
    """Generate audio playback for the given text."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_file:
            tts = gTTS(text=text, lang=st.session_state.language)
            tts.save(audio_file.name)
            st.audio(audio_file.name, format="audio/mp3")
    except Exception:
        st.error("Voice synthesis is not available at the moment.")


def get_disease_data(predicted_class: str):
    """Return structured disease details, falling back to a professional default when needed."""
    default_info = {
        "description": "Detailed information is currently unavailable for this condition.",
        "causes": [],
        "symptoms": [],
        "risk_factors": [],
        "complications": [],
        "prevention": [],
        "precautions": ["Consult a dermatologist for a professional review."],
        "care": ["Maintain skin hygiene.", "Avoid scratching or rubbing the area.", "Consult a dermatologist if the condition persists."],
        "when_to_consult": "If symptoms persist or worsen, seek professional medical advice."
    }
    disease_data = disease_info.get(predicted_class, {})
    merged = {**default_info, **disease_data}
    return merged


def create_pdf(predicted_class: str, confidence: float, timestamp: str, disease_data: dict):
    filename = "AI_Skin_Disease_Report.pdf"
    c = canvas.Canvas(filename)
    c.setTitle("AI Skin Disease Detection Report")
    c.setFont("Helvetica-Bold", 18)
    c.drawString(60, 800, "AI Skin Disease Detection Report")
    c.setFont("Helvetica", 12)
    c.drawString(60, 770, f"Predicted Disease: {predicted_class}")
    c.drawString(60, 750, f"Confidence: {confidence:.2f}%")
    c.drawString(60, 730, f"Timestamp: {timestamp}")

    y = 710
    c.setFont("Helvetica-Bold", 12)
    c.drawString(60, y, "Description:")
    y -= 18
    c.setFont("Helvetica", 11)
    for line in textwrap.wrap(disease_data["description"], width=90):
        c.drawString(80, y, line)
        y -= 14
        if y < 80:
            c.showPage()
            y = 760
            c.setFont("Helvetica", 11)

    sections = [
        ("Causes", disease_data.get("causes", [])),
        ("Symptoms", disease_data.get("symptoms", [])),
        ("Precautions", disease_data.get("precautions", [])),
    ]

    for title, items in sections:
        y -= 10
        if y < 90:
            c.showPage()
            y = 760
        c.setFont("Helvetica-Bold", 12)
        c.drawString(60, y, f"{title}:")
        y -= 18
        c.setFont("Helvetica", 11)
        if items:
            for item in items:
                for line in textwrap.wrap(item, width=90):
                    if y < 80:
                        c.showPage()
                        y = 760
                        c.setFont("Helvetica", 11)
                    c.drawString(80, y, f"- {line}" if line == textwrap.wrap(item, width=90)[0] else f"  {line}")
                    y -= 14
        else:
            c.drawString(80, y, "No additional details available.")
            y -= 14

    c.save()
    return filename


def get_severity_label(confidence: float):
    if confidence >= 80:
        return t("severity_low"), "🟢", "#22c55e"
    if confidence >= 60:
        return t("severity_moderate"), "🟡", "#f59e0b"
    return t("severity_high"), "🔴", "#ef4444"


def get_history_entry(predicted_class: str, confidence: float, severity: str, timestamp: str):
    return {
        "timestamp": timestamp,
        "disease": predicted_class,
        "confidence": confidence,
        "severity": severity
    }


def update_scan_history(predicted_class: str, confidence: float, severity: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.insert(0, get_history_entry(predicted_class, confidence, severity, timestamp))
    st.session_state.last_prediction = {
        "timestamp": timestamp,
        "disease": predicted_class,
        "confidence": confidence,
        "severity": severity
    }


def clear_history():
    st.session_state.history = []


def get_disease_care(predicted_class: str):
    disease_data = disease_info.get(predicted_class)
    if disease_data and "care" in disease_data:
        return disease_data["care"]
    return [
        "Maintain skin hygiene.",
        "Avoid scratching or rubbing the area.",
        "Consult a dermatologist for a professional review."
    ]


def get_disease_description(predicted_class: str):
    disease_data = disease_info.get(predicted_class)
    if disease_data:
        return disease_data["description"]
    return "Detailed information is currently unavailable."


def get_disease_precautions(predicted_class: str):
    disease_data = disease_info.get(predicted_class)
    if disease_data:
        return disease_data["precautions"]
    return [
        "Keep the area clean.",
        "Avoid scratching.",
        "Consult a dermatologist if the condition persists."
    ]


def generate_top3_chart(predictions: np.ndarray):
    values = predictions[0]
    top_indices = values.argsort()[-3:][::-1]
    top_labels = [classes[int(i)] for i in top_indices]
    top_values = [float(values[int(i)] * 100) for i in top_indices]
    df = pd.DataFrame({"Confidence": top_values}, index=top_labels)
    return df


init_session_state()

# Sidebar styling
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        background: #0f172a;
        color: #f8fafc;
    }
    .sidebar .css-1d391kg, .sidebar .css-ffhzg2 {
        color: #f8fafc;
    }
    .stApp {
        background-color: #f8fbff;
        color: #0f172a;
    }
    .stApp, .stApp .css-1d391kg, .stApp .css-ffhzg2, .stApp .stMarkdown, .stApp .stText {
        color: #0f172a !important;
    }
    .hero-section {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
        padding: 36px;
        border-radius: 24px;
        color: white;
        margin-bottom: 24px;
    }
    .hero-section h1, .hero-section p, .hero-section .badge-pill {
        color: #f8fafc !important;
    }
    .stButton>button, .stDownloadButton>button {
        color: #ffffff;
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%) !important;
        border: none !important;
        box-shadow: 0 10px 20px rgba(37, 99, 235, 0.2) !important;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
    }
    .badge-pill {
        display: inline-block;
        margin-right: 10px;
        margin-bottom: 10px;
        padding: 10px 16px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.16);
        color: #ffffff;
        font-weight: 600;
    }
    .metric-card, .roadmap-card {
        border-radius: 22px;
        padding: 24px;
        background: white;
        box-shadow: 0 22px 45px rgba(15, 23, 42, 0.08);
    }
    .metric-card.accuracy .metric-value {
        color: #16a34a;
    }
    .metric-card.precision .metric-value {
        color: #0ea5e9;
    }
    .metric-card.recall .metric-value {
        color: #f59e0b;
    }
    .metric-card.f1score .metric-value {
        color: #7c3aed;
    }
    .metric-card h4, .roadmap-card h4 {
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        color: #1d4ed8;
    }
    .metric-note {
        margin-top: 10px;
        color: #64748b;
    }
    .disclaimer-text {
        color: #dc2626;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main layout and sidebar navigation
with st.sidebar:
    st.title(t("navigation"))
    selected_section = st.radio(
        "Navigation sections",
        ["dashboard", "scan_history", "model_info", "report"],
        format_func=lambda x: t(x),
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("---")
    language_choice = st.selectbox(t("language_selector"), list(LANGUAGE_OPTIONS.keys()), index=0)
    st.session_state.language = LANGUAGE_OPTIONS[language_choice]
    st.markdown("---")
    st.subheader(t("model_information"))
    st.write(f"**{t('supported_diseases')}:** 22")
    st.write(f"**{t('model_type')}:** CNN / Deep Learning")
    st.write(f"**{t('dataset')}:** Placeholder dataset details")
    st.write(f"**{t('last_update')}:** Placeholder date")
    st.markdown("---")
    st.subheader(t("project_details"))
    st.write(f"- {t('clinical_validation')}")
    st.write(f"- {t('dermatologist_review')}")
    st.write(f"- {t('hospital_integration')}")
    st.write(f"- {t('mobile_app_release')}")
    st.markdown("---")
    st.write(f"**{t('version')}:** 1.0.0")
    st.write(t("developer_info"))
    st.write(t("copyright"))

# Dashboard section
if selected_section == "dashboard":
    st.markdown(
        "<div class='hero-section'>"
        f"<h1>{t('hero_title')}</h1>"
        f"<p style='font-size:1.05rem; line-height:1.7; max-width:760px;'>{t('hero_subtitle')}</p>"
        f"<div style='margin-top:18px;'>"
        f"<span class='badge-pill'>{t('feature_deep_learning')}</span>"
        f"<span class='badge-pill'>{t('feature_categories')}</span>"
        f"<span class='badge-pill'>{t('feature_risk')}</span>"
        f"<span class='badge-pill'>{t('feature_report')}</span>"
        f"<span class='badge-pill'>{t('feature_voice')}</span>"
        f"</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("## " + t("performance_overview"))
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    perf_col1.markdown(
        "<div class='metric-card accuracy'><h4>" + t("accuracy") + "</h4><p class='metric-value'>92.4%</p><p class='metric-note'>Placeholder value</p></div>",
        unsafe_allow_html=True,
    )
    perf_col2.markdown(
        "<div class='metric-card precision'><h4>" + t("precision") + "</h4><p class='metric-value'>91.0%</p><p class='metric-note'>Placeholder value</p></div>",
        unsafe_allow_html=True,
    )
    perf_col3.markdown(
        "<div class='metric-card recall'><h4>" + t("recall") + "</h4><p class='metric-value'>90.2%</p><p class='metric-note'>Placeholder value</p></div>",
        unsafe_allow_html=True,
    )
    perf_col4.markdown(
        "<div class='metric-card f1score'><h4>" + t("f1_score") + "</h4><p class='metric-value'>90.6%</p><p class='metric-note'>Placeholder value</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    trust_col, info_col = st.columns([2, 1])
    with trust_col:
        st.markdown("## " + t("why_trust_section"))
        st.write(t("confidence_explanation_text"))
    with info_col:
        st.markdown("## " + t("quick_guide"))
        st.write("1. " + t("upload_image"))
        st.write("2. " + t("confidence_explanation"))
        st.write("3. " + t("download_report"))

    st.markdown("---")
    roadmap1, roadmap2, roadmap3, roadmap4 = st.columns(4)
    roadmap1.markdown("<div class='roadmap-card'><h4>" + t("clinical_validation") + "</h4><p>Evaluate model performance with clinical data.</p></div>", unsafe_allow_html=True)
    roadmap2.markdown("<div class='roadmap-card'><h4>" + t("dermatologist_review") + "</h4><p>Integrate expert review for improved trust.</p></div>", unsafe_allow_html=True)
    roadmap3.markdown("<div class='roadmap-card'><h4>" + t("hospital_integration") + "</h4><p>Prepare for healthcare-friendly deployment.</p></div>", unsafe_allow_html=True)
    roadmap4.markdown("<div class='roadmap-card'><h4>" + t("mobile_app_release") + "</h4><p>Bring the system to mobile users.</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    upload_col, help_col = st.columns([2, 1])
    with upload_col:
        uploaded_file = st.file_uploader(
            t("upload_image"),
            type=["jpg", "jpeg", "png"],
            help=t("choose_image_help"),
        )
    with help_col:
        st.markdown("### " + t("get_started"))
        st.write("- " + t("upload_image"))
        st.write("- " + t("confidence_explanation"))
        st.write("- " + t("download_report"))

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=t("upload_image"), use_container_width=True)
        resized = image.resize((224, 224))
        img_array = np.array(resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = classes[int(np.argmax(prediction))]
        confidence = float(np.max(prediction) * 100)
        severity_label, severity_emoji, severity_color = get_severity_label(confidence)
        disease_data = get_disease_data(predicted_class)
        description = disease_data["description"]
        precautions = disease_data["precautions"]
        care_recommendations = disease_data.get("care", [])
        causes = disease_data.get("causes", [])
        symptoms = disease_data.get("symptoms", [])
        risk_factors = disease_data.get("risk_factors", [])
        complications = disease_data.get("complications", [])
        prevention = disease_data.get("prevention", [])
        when_to_consult = disease_data.get("when_to_consult", "If symptoms persist or worsen, seek professional medical advice.")

        update_scan_history(predicted_class, confidence, severity_label)

        st.markdown("---")
        st.markdown("## " + t("prediction_result"))
        left_col, right_col = st.columns([2, 1])
        with left_col:
            st.metric(t("predicted_disease"), predicted_class)
            st.metric(t("confidence"), f"{confidence:.2f}%")
            st.progress(min(100, int(confidence)))
            st.markdown("#### " + t("top_predictions"))
            top3_df = generate_top3_chart(prediction)
            st.bar_chart(top3_df)
        with right_col:
            st.markdown(
                f"<div class='metric-card'><h4>{t('risk_assessment')}</h4><p style='color:{severity_color}; font-weight:700;'>{severity_emoji} {severity_label}</p></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='metric-card'><h4>" + t("purpose") + "</h4><p>" + t("purpose_text") + "</p></div>",
                unsafe_allow_html=True,
            )

        st.markdown("## " + t("learn_more_about_this_disease"))
        with st.expander("📚 View Disease Details"):
            st.markdown("### 📖 Disease Overview")
            st.write(description)

            st.markdown("### 🦠 Causes")
            if causes:
                for item in causes:
                    st.write(f"- {item}")
            else:
                st.write("Detailed information is currently unavailable for this condition.")

            st.markdown("### 🤒 Common Symptoms")
            if symptoms:
                for item in symptoms:
                    st.write(f"- {item}")
            else:
                st.write("Detailed information is currently unavailable for this condition.")

            st.markdown("### ⚠ Risk Factors")
            if risk_factors:
                for item in risk_factors:
                    st.write(f"- {item}")
            else:
                st.write("Detailed information is currently unavailable for this condition.")

            st.markdown("### 🧩 Possible Complications")
            if complications:
                for item in complications:
                    st.write(f"- {item}")
            else:
                st.write("Detailed information is currently unavailable for this condition.")

            st.markdown("### 🛡 Prevention Tips")
            if prevention:
                for item in prevention:
                    st.write(f"- {item}")
            else:
                st.write("Detailed information is currently unavailable for this condition.")

            st.markdown("### 💊 Recommended Precautions")
            if precautions:
                for item in precautions:
                    st.write(f"- {item}")
            else:
                st.write("Detailed information is currently unavailable for this condition.")

            st.markdown("### 👨‍⚕ When to Consult a Doctor")
            st.write(when_to_consult)

        with st.expander(t("care_recommendations")):
            for item in care_recommendations:
                st.write(f"- {item}")

        if st.button(t("voice_assistant")):
            speak(
                f"The predicted disease is {predicted_class}. Confidence is {confidence:.2f} percent. "
                f"{description} "
                f"Key precautions include {', '.join(precautions[:3])}."
            )
            st.success("Voice playback complete.")

        report_filename = create_pdf(predicted_class, confidence, st.session_state.last_prediction["timestamp"], disease_data)
        with open(report_filename, "rb") as report_file:
            st.download_button(
                label=t("download_report"),
                data=report_file,
                file_name="AI_Skin_Disease_Report.pdf",
                mime="application/pdf",
            )

        st.markdown(
            f"<p class='disclaimer-text'>{t('disclaimer')}</p>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("## " + t("scan_history_title"))
    if st.button(t("clear_history")):
        clear_history()
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df[["timestamp", "disease", "confidence", "severity"]])
    else:
        st.info(t("no_history"))

elif selected_section == "scan_history":
    st.markdown("## " + t("scan_history_title"))
    if st.button(t("clear_history")):
        clear_history()
    if st.session_state.history:
        for entry in st.session_state.history:
            with st.expander(f"{entry['timestamp']} — {entry['disease']} ({entry['confidence']:.1f}%)"):
                st.write(f"**{t('severity')}:** {entry['severity']}")
                st.write(f"**{t('timestamp')}:** {entry['timestamp']}")
                st.write(f"**{t('confidence')}:** {entry['confidence']:.1f}%")
    else:
        st.info(t("no_history"))

elif selected_section == "model_info":
    st.markdown("## " + t("model_info_title"))
    st.write(f"**{t('supported_diseases')}:** 22")
    st.write(f"**{t('model_type')}:** CNN / Deep Learning")
    st.write(f"**{t('dataset')}:** Placeholder dataset details")
    st.write(f"**{t('last_update')}:** Placeholder date")
    st.markdown("---")
    st.markdown("## " + t("why_trust"))
    st.write(t("why_trust_section"))
    st.write(t("confidence_explanation_text"))
    st.markdown("---")
    st.markdown("## " + t("project_roadmap"))
    st.write(f"- {t('clinical_validation')}")
    st.write(f"- {t('dermatologist_review')}")
    st.write(f"- {t('hospital_integration')}")
    st.write(f"- {t('mobile_app_release')}")

else:
    st.markdown("## " + t("report_summary"))
    if st.session_state.last_prediction:
        last = st.session_state.last_prediction
        st.write(f"**{t('predicted_disease')}:** {last['disease']}")
        st.write(f"**{t('confidence')}:** {last['confidence']:.2f}%")
        st.write(f"**{t('timestamp')}:** {last['timestamp']}")
        st.write(f"**{t('severity')}:** {last['severity']}")
        last_disease_data = get_disease_data(last['disease'])
        report_filename = create_pdf(
            last['disease'],
            last['confidence'],
            last['timestamp'],
            last_disease_data,
        )
        with open(report_filename, "rb") as report_file:
            st.download_button(
                label=t("download_latest_report"),
                data=report_file,
                file_name="AI_Skin_Disease_Report.pdf",
                mime="application/pdf",
            )
    else:
        st.info(t("report_no_prediction"))

st.markdown("---")
st.markdown(
    f"<div style='text-align:center; color:#475569; padding:12px 0;'>"
    f"<strong>{t('version')}</strong>: 1.0.0 &nbsp;|&nbsp; {t('developer_info')} &nbsp;|&nbsp; {t('copyright')}"
    "</div>",
    unsafe_allow_html=True,
)
