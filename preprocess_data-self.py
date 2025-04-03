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
import jieba
import googletrans
import string

# Download NLTK resources (run once)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Download spaCy resources (run once)
nlp = spacy.load("en_core_web_trf")
nlp_zh = spacy.load("zh_core_web_trf")

# Process text for 'Job Description' and 'Requirements' - 
def process_text(text, custom_stopwords=None, keep_phrases=True, min_word_length=2, 
               use_spacy=True, extract_entities=True, bigrams=True, handle_chinese=True,
               translate_chinese=True):
    """
    Process text data for NLP tasks with advanced options and multilingual support.
    
    Parameters:
    -----------
    text : str
        The input text to process
    custom_stopwords : set or list, optional
        Additional stopwords to remove
    keep_phrases : bool, default=True
        Whether to maintain important multi-word phrases
    min_word_length : int, default=2
        Minimum character length for words to keep
    use_spacy : bool, default=True
        Whether to use spaCy for advanced processing
    extract_entities : bool, default=True
        Whether to extract and preserve named entities
    bigrams : bool, default=True
        Whether to include common bigrams
    handle_chinese : bool, default=True
        Whether to handle Chinese text using jieba segmentation
    translate_chinese : bool, default=True
        Whether to translate Chinese text to English
    
    Returns:
    --------
    str
        Processed text
    """
    if pd.isna(text) or not text:
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Check if text contains Chinese characters
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text)) if handle_chinese else False
    
    # Handle Chinese text if detected
    if has_chinese:
        try:
            # Import Chinese processing libraries
            import jieba
            
            # For translation (if enabled)
            if translate_chinese:
                try:
                    from googletrans import Translator
                    translator = Translator()
                    
                    # Translate Chinese text to English
                    # Note: In production, should handle API limits and errors
                    translation = translator.translate(text, src='zh-cn', dest='en')
                    translated_text = translation.text.lower()
                    
                    # Combine original and translated text (preserving both)
                    combined_text = f"{text} {translated_text}"
                    text = combined_text
                except ImportError:
                    print("googletrans not installed. Install with: pip install googletrans==4.0.0-rc1")
                except Exception as e:
                    print(f"Translation failed: {e}. Proceeding with original text.")
            
            # Segment Chinese text
            seg_list = jieba.cut(text, cut_all=False)
            segmented_text = " ".join(seg_list)
            
            # Replace original text with segmented text for further processing
            text = segmented_text
            
        except ImportError:
            print("jieba not installed. Install with: pip install jieba")
            # Continue with unsegmented text if jieba is not available
        except Exception as e:
            print(f"Chinese text processing failed: {e}. Proceeding with original text.")
    
    # Preserve common phrases before tokenization if keep_phrases is True
    important_phrases = []
    if keep_phrases:
        # Job-related multi-word terms to preserve (English and Chinese equivalents)
        phrase_patterns = [
            # English terms
            r'machine learning', r'deep learning', r'data science', r'data analysis',
            r'artificial intelligence', r'business intelligence', r'product management',
            r'project management', r'software development', r'web development',
            r'full stack', r'front end', r'back end', r'devops', r'cloud computing',
            r'aws', r'azure', r'google cloud', r'data engineering', r'natural language processing',
            r'computer vision', r'user experience', r'user interface', r'scrum master',
            r'agile methodology', r'customer relationship management', r'search engine optimization',
            r'digital marketing', r'content marketing', r'social media', r'human resources',
            r'financial analysis', r'supply chain', r'quality assurance', r'quality control',
            r'business administration', r'operations management', r'customer service',
            r'power bi', r'tableau', r'sql server', r'ms excel', r'ms office',
            
            # Chinese equivalent terms
            r'机器学习', r'深度学习', r'数据科学', r'数据分析',
            r'人工智能', r'商业智能', r'产品管理',
            r'项目管理', r'软件开发', r'网站开发',
            r'全栈', r'前端', r'后端', r'运维', r'云计算',
            r'数据工程', r'自然语言处理',
            r'计算机视觉', r'用户体验', r'用户界面', r'敏捷开发',
            r'客户关系管理', r'搜索引擎优化',
            r'数字营销', r'内容营销', r'社交媒体', r'人力资源',
            r'财务分析', r'供应链', r'质量保证', r'质量控制',
            r'工商管理', r'运营管理', r'客户服务'
        ]
        
        for phrase in phrase_patterns:
            if re.search(r'\b' + phrase + r'\b', text):
                # Replace spaces with underscores to preserve the phrase
                text = re.sub(r'\b' + phrase + r'\b', phrase.replace(' ', '_'), text)
                important_phrases.append(phrase.replace(' ', '_'))
    
    # Use spaCy for advanced NLP if enabled
    if use_spacy and nlp is not None:
        try:
            # Process with spaCy
            doc = nlp(text)
            
            processed_tokens = []
            
            # Extract named entities if requested
            entities = []
            if extract_entities:
                entities = [e.text.replace(' ', '_') for e in doc.ents 
                           if e.label_ in ['ORG', 'PRODUCT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'GPE']]
            
            # Process tokens
            for token in doc:
                # Skip stopwords and short words
                if (custom_stopwords and token.text in custom_stopwords) or len(token.text) <= min_word_length:
                    continue
                
                # Skip if it's a stopword (unless it's in our important phrases)
                if token.is_stop and token.text not in important_phrases and token.text not in entities:
                    continue
                
                # Use lemma
                lemma = token.lemma_
                
                # Keep original form for proper nouns and technical terms
                if token.pos_ in ['PROPN', 'X'] or token.text in important_phrases or token.text in entities:
                    processed_tokens.append(token.text)
                else:
                    processed_tokens.append(lemma)
            
            # Add extracted entities to ensure they're preserved
            processed_tokens.extend([e for e in entities if e not in processed_tokens])
            
            # Generate bigrams if enabled
            if bigrams and len(processed_tokens) > 1:
                bigram_list = ['_'.join(bg) for bg in zip(processed_tokens[:-1], processed_tokens[1:])]
                # Only keep meaningful bigrams (filter out common word combinations)
                meaningful_bigrams = [bg for bg in bigram_list 
                                     if not any(common in bg for common in ['the_', '_the', '_and', 'and_', '_of', 'of_'])]
                # Add top bigrams (limit to avoid explosion)
                top_bigrams = meaningful_bigrams[:10]  # Limit to top 10 bigrams
                processed_tokens.extend(top_bigrams)
            
            return " ".join(processed_tokens)
            
        except Exception as e:
            # Fallback to traditional method if spaCy fails
            print(f"spaCy processing failed: {e}. Falling back to traditional method.")
            pass
    
    # Traditional method (fallback)
    # Remove punctuation but preserve underscores (for phrases we've already processed)
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
        # Skip stopwords and short words (unless they're our preserved phrases)
        if (word in stop_words and '_' not in word) or len(word) <= min_word_length:
            continue
        
        # Keep phrases that we preserved earlier
        if word in important_phrases:
            lemmatized_words.append(word)
            continue
            
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
    
    # Generate bigrams if enabled
    if bigrams and len(lemmatized_words) > 1:
        bigram_list = ['_'.join(bg) for bg in zip(lemmatized_words[:-1], lemmatized_words[1:])]
        # Filter meaningful bigrams
        meaningful_bigrams = [bg for bg in bigram_list 
                             if not any(common in bg for common in ['the_', '_the', '_and', 'and_', '_of', 'of_'])]
        # Add top bigrams (limit to avoid explosion)
        lemmatized_words.extend(meaningful_bigrams[:10])
    
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

def main():
    print("Starting data preprocessing...")
    start_time = time.time()
    
    # Custom stopwords relevant to job descriptions
    custom_stopwords = {'experience', 'job', 'work', 'skills', 'year', 'years', 'company',
                      'team', 'role', 'ability', 'required', 'requirements', 'responsibilities',
                      'candidate', 'position', 'including', 'using', 'knowledge'}
    
    # Add Chinese stopwords to your custom stopwords
    from multilingual_config import CHINESE_STOPWORDS
    custom_stopwords.update(CHINESE_STOPWORDS)
    
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
    
    # Add Chinese stopwords to your custom stopwords
    from multilingual_config import CHINESE_STOPWORDS
    custom_stopwords.update(CHINESE_STOPWORDS)

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
                
                # Process job description with enhanced options
                processed_description = " ".join(level_data['job_description'].fillna('').apply(
                    lambda x: process_text(
                        x, 
                        custom_stopwords=custom_stopwords,
                        keep_phrases=True,
                        use_spacy=True,
                        extract_entities=True,
                        bigrams=True,
                        handle_chinese=True,
                        translate_chinese=False
                    )
                ))

                
                # Process requirements with enhanced options
                processed_requirements = " ".join(level_data['requirements'].fillna('').apply(
                    lambda x: process_text(
                        x, 
                        custom_stopwords=custom_stopwords,
                        keep_phrases=True,
                        use_spacy=True,
                        extract_entities=True,
                        bigrams=True,
                        handle_chinese=True,
                        translate_chinese=False
                    )
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