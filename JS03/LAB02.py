# LAB 2 - TF-IDF Feature Extraction
corpus = [
    'the house had a tiny little mouse',
    'the cat saw the mouse',
    'the mouse ran away from the house',
    'the cat finally ate the mouse',
    'the end of the mouse story'
]


# Step 1 - Construct the TF-IDF Model
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TfidfVectorizer object
vect = TfidfVectorizer(stop_words='english')

# TF-IDF weighting
resp = vect.fit_transform(corpus)

# Print the results
print("=== TF-IDF MATRIX ===")
print(resp)


# Step 2 - Inspect the Terms Used
print("\n=== FEATURE NAMES ===")
print(vect.get_feature_names_out())