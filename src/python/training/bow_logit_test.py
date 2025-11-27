################################################################################
# EXTERNAL VALIDATION SCRIPT
# 1. Load the pre-trained model
# 2. Define new, unseen data (simulating real-world inputs)
# 3. Predict and analyze results
################################################################################
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix

# --- 1. Load the Model ---

model_path = 'models/bow_logit_v1.joblib'

try:
    print(f"Loading model from {model_path}...")
    # This loads the full pipeline (CountVectorizer + LogisticRegression)
    model = joblib.load(model_path)
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print("❌ Error: Model file not found. Run the training script first.")
    exit()

# --- 2. New Data ---

# AI generated:

new_data = [
    # --- POLITICS & WORLD (TRUE) ---
    ("The United Nations Security Council unanimously voted to impose new sanctions on North Korea following its latest missile test.", "TRUE"),
    ("Senate Majority Leader announced a delay in the vote on the new healthcare bill until after the July recess.", "TRUE"),
    ("German Chancellor Angela Merkel met with French President Emmanuel Macron to discuss the future of the European Union.", "TRUE"),
    ("The U.S. Department of Labor reported that the economy added 150,000 jobs last month, beating analysts' expectations.", "TRUE"),
    ("Prime Minister Theresa May stated that Brexit negotiations will officially begin next Monday in Brussels.", "TRUE"),
    ("A massive earthquake of magnitude 7.1 struck central Mexico, causing damage to several buildings in the capital.", "TRUE"),
    ("China has officially opened the world's longest sea-crossing bridge, connecting Hong Kong to the mainland.", "TRUE"),
    ("The White House Press Secretary confirmed that the President will visit Japan and South Korea next month.", "TRUE"),
    ("NASA announced the discovery of a new exoplanet that could potentially support liquid water.", "TRUE"),
    ("Local authorities in California issued an evacuation order due to the rapidly spreading wildfire in the northern district.", "TRUE"),
    ("The International Monetary Fund (IMF) revised its global growth forecast down by 0.2% for the coming fiscal year.", "TRUE"),
    ("Thousands of protesters gathered in London to demand action on climate change ahead of the G7 summit.", "TRUE"),
    ("Russian officials denied allegations of interference in the upcoming general elections.", "TRUE"),
    ("The Supreme Court ruled 5-4 in favor of the new environmental protection regulations.", "TRUE"),
    ("Apple Inc. shares rose 2% after the company reported quarterly earnings that exceeded Wall Street estimates.", "TRUE"),
    ("Scientists at CERN have restarted the Large Hadron Collider after a two-year upgrade period.", "TRUE"),
    ("The World Health Organization declared the Ebola outbreak in Congo a global health emergency.", "TRUE"),
    ("Canada's Prime Minister Justin Trudeau unveiled a new plan to reduce carbon emissions by 2030.", "TRUE"),
    ("The Tokyo Olympic Committee announced the official schedule for the opening ceremony.", "TRUE"),
    ("Brazilian police arrested a former governor on charges of corruption and money laundering.", "TRUE"),

    # --- FAKE NEWS (CLICKBAIT & CONSPIRACY) ---
    ("BREAKING: Pope Francis endorses Donald Trump for President! Read the shocking statement here.", "FAKE"),
    ("Scientists admit that the new vaccine contains tracking microchips developed by Bill Gates.", "FAKE"),
    ("SHOCKING: Hillary Clinton found running a secret pizza shop in the basement of the Pentagon!", "FAKE"),
    ("Alien autopsy video leaked from Area 51 proves the government has been lying to us for 50 years!", "FAKE"),
    ("You won't believe what happened! The Statue of Liberty just opened its eyes!", "FAKE"),
    ("BANNED VIDEO: Doctors define a new miracle fruit that cures cancer in 24 hours!", "FAKE"),
    ("NASA confirms Earth will go dark for 6 days in December due to solar storm.", "FAKE"),
    ("Mark Zuckerberg is actually a reptilian shape-shifter according to leaked internal Facebook documents.", "FAKE"),
    ("ALERT: The government is poisoning our water supply to control our minds! Share this before it's deleted!", "FAKE"),
    ("Obama signed an executive order to ban the American flag in schools just before leaving office.", "FAKE"),
    ("The FBI just raided the White House and found gold bars stolen from Fort Knox!", "FAKE"),
    ("Celebrity shocking death! This famous actor was replaced by a clone 5 years ago.", "FAKE"),
    ("CONFIRMED: The moon landing was filmed in a studio in Hollywood by Stanley Kubrick.", "FAKE"),
    ("Drinking bleach is the secret cure for COVID-19 that Big Pharma doesn't want you to know.", "FAKE"),
    ("George Soros was arrested in Switzerland for trying to manipulate the World Cup.", "FAKE"),
    ("A mermaid washed up on the beach in Florida and the images will blow your mind!", "FAKE"),
    ("Breaking: United Nations plans to confiscate all guns from American citizens by next month.", "FAKE"),
    ("WikiLeaks dumps emails proving that dinosaurs never went extinct and are living underground.", "FAKE"),
    ("Yoko Ono admits she had an affair with Hillary Clinton in the 70s!", "FAKE"),
    ("North Korea lands first man on the Sun during a night mission to avoid burning.", "FAKE"),

    # --- TRICKY CASES (Hard for the model) ---
    
    # Fake written formally (The King of Spain case)
    ("The King of Spain announced yesterday that he will move his residence to the Moon by 2030.", "FAKE"),
    
    # Fake written with "trusted" words
    ("Reuters reports that the Prime Minister of Australia has declared war on Emus again.", "FAKE"),
    
    # True but sounds weird/alarming
    ("A Florida man was arrested for trying to trade a live alligator for a pack of beer.", "TRUE"),
    
    # True but has "Fake" trigger words (hacking, emails)
    ("Yahoo confirmed a massive data breach affecting 3 billion user accounts, exposing emails and passwords.", "TRUE"),
    
    # True but very short (Model lacks context)
    ("The stock market closed lower today.", "TRUE"),
    
    # Fake but uses economic jargon
    ("The Federal Reserve secretly printed 500 trillion dollars to buy Greenland from Denmark.", "FAKE"),
    
    # True but scientific clickbait-ish title
    ("NASA plans to crash a spacecraft into an asteroid to save Earth.", "TRUE"), # Refers to DART mission
    
    # Fake regarding specific Fake News tropes
    ("CNN finally admits they staged the moon landing footage.", "FAKE"),
    
    # True regarding a celebrity (often target of fakes)
    ("Actor Tom Hanks announced he and his wife tested positive for coronavirus.", "TRUE"),
    
    # Fake with "Medical" terminology
    ("New study proves that looking at the sun directly improves vision by 50%.", "FAKE")
]

df_external = pd.DataFrame(new_data, columns=['text', 'ground_truth'])

# --- 3. Prediction ---

print("\n--- Predicting on New Data ---")
predictions = model.predict(df_external['text'])
probs = model.predict_proba(df_external['text'])

# --- 4. Analysis ---

print(f"{'PREDICTION':<12} | {'CONFIDENCE':<10} | {'REALITY':<10} | {'TEXT (Truncated)'}")
print("-" * 80)

for text, pred, prob, truth in zip(df_external['text'], predictions, probs, df_external['ground_truth']):
    # prob[1] is the probability of class 1 (Fake)
    confidence = prob[max(pred, 0)] # Get the prob of the predicted class
    
    # Map 0/1 to labels
    pred_label = "FAKE" if pred == 1 else "TRUE"
    
    # Visual check
    match_icon = "✅" if pred_label == truth else "❌"
    
    print(f"{match_icon} {pred_label:<8} | {confidence:.1%} | {truth:<10} | {text[:40]}...")

y_real = df_external['ground_truth'].map({'FAKE': 1, 'TRUE': 0})

acc = accuracy_score(y_real, predictions)
print(f"\n📊 ACCURACY on AIgenerated data: {acc:.2%}")

cm = confusion_matrix(y_real, predictions)

tn, fp, fn, tp = cm.ravel()
print(f"✅(TP): {tp} of {sum(y_real==1)}")
print(f"✅(TN): {tn} of {sum(y_real==0)}")
print(f"❌(FN): {fn}")
print(f"❌(FP): {fp}")