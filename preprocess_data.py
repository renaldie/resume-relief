import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.util import ngrams
import spacy
import time
import os

# Download NLTK resources (run once)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Download spaCy resources (run once)
nlp = spacy.load("en_core_web_trf")

##########
# TODO
# Process text for 'Job Description' and 'Requirements' - 
def process_text(text, custom_stopwords=None):
    if pd.isna(text):
        return ""
    
    # Convert to lowercase and remove punctuation
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    if custom_stopwords:
        stop_words.update(custom_stopwords)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # POS tag tokens to improve lemmatization
    tagged_tokens = pos_tag(tokens)
    
    # Lemmatize words based on POS tags
    lemmatized_words = []
    for word, tag in tagged_tokens:
        if word not in stop_words and len(word) > 2:
            # Convert POS tag to WordNet format
            if tag.startswith('J'):
                pos = 'a'  # adjective
            elif tag.startswith('V'):
                pos = 'v'  # verb
            elif tag.startswith('N'):
                pos = 'n'  # noun
            elif tag.startswith('R'):
                pos = 'r'  # adverb
            else:
                pos = 'n'  # default to noun
                
            lemmatized_words.append(lemmatizer.lemmatize(word, pos=pos))
    
    return " ".join(lemmatized_words)

# Process skills
def process_skills(skills_text, custom_stopwords=None):
    if pd.isna(skills_text):
        return ""
    
    # Convert to lowercase
    skills_text = str(skills_text).lower()
    
    # Split by comma
    skills = [skill.strip() for skill in skills_text.split(',')]
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    if custom_stopwords:
        stop_words.update(custom_stopwords)
    
    # Remove stop words and filter out short skills
    skills = [skill for skill in skills if skill and skill not in stop_words and len(skill) > 2]
    
    return " ".join(skills)
##########

def main():
    print("Starting data preprocessing...")
    start_time = time.time()
    
    # Custom stopwords relevant to job descriptions
    custom_stopwords = {'experience', 'job', 'work', 'skills', 'year', 'years', 'company',
                        'team', 'role', 'ability', 'required', 'requirements', 'responsibilities',
                        'candidate', 'position', 'including', 'using', 'knowledge'}
    
    # Load the original data
    jobs_associate = pd.read_csv('data/raw/Jobs_associate.csv', encoding="utf-8-sig")
    jobs_director = pd.read_csv('data/raw/Jobs_director.csv', encoding="utf-8-sig")
    jobs_entry_level = pd.read_csv('data/raw/Jobs_entry_level.csv', encoding="utf-8-sig")
    jobs_executive = pd.read_csv('data/raw/Jobs_executive.csv', encoding="utf-8-sig")
    jobs_internship_level = pd.read_csv('data/raw/Jobs_internship_level.csv', encoding="utf-8-sig")
    jobs_mid_senior_level = pd.read_csv('data/raw/Jobs_mid_senior_level.csv', encoding="utf-8-sig")

    # Process the original data
    df = pd.concat([jobs_associate, 
                            jobs_director, 
                            jobs_entry_level, 
                            jobs_executive, 
                            jobs_internship_level,
                            jobs_mid_senior_level], ignore_index=True)
    
    # Rule 1.
    # If employment_type is 'Part-time' and title contains intern-related terms and seniority is empty
    internship_keywords = ['intern', 'internship', '實習']
    mask_part_time = (df['employment_type'] == 'Part-time') & \
                    (df['seniority'].isna() | (df['seniority'] == ''))
    # Create a boolean mask for titles containing intern-related keywords
    mask_intern_title = df['title'].str.lower().apply(
        lambda x: any(keyword in str(x).lower() for keyword in internship_keywords)
    ) if 'title' in df.columns else False
    # Apply
    df.loc[mask_part_time & mask_intern_title, 'seniority'] = 'Internship'

    # Rule 2.
    # If employment_type is 'Internship', always set seniority to 'Internship'
    df.loc[df['employment_type'] == 'Internship', 'seniority'] = 'Internship'

    # Rule 3
    # If title contains intern-related terms and seniority is missing, fill with 'Internship'
    mask_missing_seniority = df['seniority'].isna() | (df['seniority'] == '')
    # Check if title contains any internship keywords
    mask_intern_title = df['title'].str.lower().apply(
        lambda x: any(keyword in str(x).lower() for keyword in internship_keywords)
    ) if 'title' in df.columns else False
    # Apply
    df.loc[mask_missing_seniority & mask_intern_title, 'seniority'] = 'Internship'

    # Rule 4
    # For remaining missing seniority values, fill with 'Entry level'
    still_missing_mask = df['seniority'].isna() | (df['seniority'] == '')
    df.loc[still_missing_mask, 'seniority'] = 'Entry level'

    # Rule 5
    # Create a mask for rows where title contains "Error code"
    mask_error_code = df['title'].str.contains("Error code", case=False, na=False)
    df = df[~mask_error_code]    
    
    # Process by category and seniority level
    categories = sorted(df['category_major'].dropna().unique())
    
    # Define seniority levels
    seniority_levels = [
        'Internship',
        'Entry level', 
        'Assistant', 
        'Mid-Senior level', 
        'Director',
        'Executive (VP, GM, C-Level)',
    ]
    
    # Create result dataframe
    results = []
    
    for category in categories:
        print(f"Processing category: {category}")
        filtered_data = df[df['category_major'] == category]
        
        for level in seniority_levels:
            level_data = filtered_data[filtered_data['seniority'] == level]
            
            if not level_data.empty:
                print(f"  - Processing {level} ({len(level_data)} jobs)")
                
                # Process skills
                processed_skills = " ".join(level_data['skills'].fillna('').apply(
                    lambda x: process_skills(x, custom_stopwords)
                ))
                
                # Process job description
                processed_description = " ".join(level_data['job_description'].fillna('').apply(
                    lambda x: process_text(x, custom_stopwords)
                ))
                
                # Process requirements
                processed_requirements = " ".join(level_data['requirements'].fillna('').apply(
                    lambda x: process_text(x, custom_stopwords)
                ))
                
                # Add to results
                results.append({
                    'category': category,
                    'seniority': level,
                    'count': len(level_data),
                    'processed_skills': processed_skills,
                    'processed_job_description': processed_description,
                    'processed_requirements': processed_requirements
                })
    
    # Create processed dataframe
    processed_df = pd.DataFrame(results)
    
    # Save to CSV
    processed_df.to_csv("data/jobs_processed.csv", index=False)
    
    elapsed_time = time.time() - start_time
    print(f"Preprocessing completed in {elapsed_time:.2f} seconds.")
    print(f"Processed data saved to data/jobs_processed.csv")

if __name__ == "__main__":
    main()