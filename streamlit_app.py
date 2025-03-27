import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.util import ngrams
import spacy

# Download NLTK resources (run once)
nltk.data.find('corpora/stopwords')
nltk.download('stopwords', quiet=True)
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Download spaCy resources (run once)
nlp = spacy.load("en_core_web_sm")

# Set page config
st.set_page_config(
    page_title="Resume Relief",
    page_icon="😮‍💨",
    layout="wide",
)
st.title("What do I need to get a job? 😭")

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv("data/jobs_complete.csv", encoding="utf-8-sig")
    return df

# Process text to extract meaningful terms
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

# Process skills (comma-separated)
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

# Generate wordcloud with specific colormap
def generate_wordcloud(text, title, colormap):
    if not text or len(text.strip()) == 0:
        return None
        
    font_path = "data/simhei.ttf"
    
    wc = WordCloud(
        font_path=font_path,
        background_color="white",
        max_words=100,
        max_font_size=80,
        width=800,
        height=400,
        colormap=colormap
    ).generate(text)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, fontsize=16)
    return plt

# Main app
def main():
    # Load data
    df = load_data()
    
    # Custom stopwords relevant to job descriptions
    custom_stopwords = {'experience', 'job', 'work', 'skills', 'year', 'years', 'company',
                        'team', 'role', 'ability', 'required', 'requirements', 'responsibilities',
                        'candidate', 'position', 'including', 'using', 'knowledge'}
    
    # Get unique category_major values
    categories_major = sorted(df['category_major'].dropna().unique())
    if not categories_major:
        st.error("No category_major data found in the dataset.")
        return
    
    # Get unique seniority levels with a specific order
    seniority_order = [
        'Internship',
        'Entry level', 
        'Assistant', 
        'Mid-Senior level', 
        'Director',
        'Executive (VP, GM, C-Level)',
    ]

    # Create a mapping for seniority display names with emojis
    seniority_labels = {
        'Internship': '🎓 Internship',
        'Entry level': '🌱 Entry Level',
        'Assistant': '🔍 Assistant',
        'Mid-Senior level': '⚙️ Mid-Senior Level',
        'Director': '🚀 Director',
        'Executive (VP, GM, C-Level)': '👑 Executive (VP, GM, C-Level)',
    }

    # Map the actual seniority levels to display
    seniority_levels = seniority_order.copy()

    # Check if we have any seniority data
    if not seniority_levels:
        st.error("No seniority data found in the dataset.")
        return

    # 1. Category major buttons at the top
    st.write("## A Job in 🏢")

    # Create a mapping for category display names
    category_labels = {
        "Bio, Medical": "Bio / Medical 🧬",
        "Catering / Food & Beverage": "Catering / Food & Beverage 🍔",
        "Construction": "Construction 🏗️",
        "Customer Service": "Customer Service 🛎️",
        "Design": "Design 🎨",
        "Education": "Education 📚",
        "Engineering": "Engineering 🔧",
        "Finance": "Finance 💵",
        "Game Production": "Game Production 🎮",
        "HR": "HR 👩‍💼",
        "IT": "IT 💻",
        "Law": "Law ⚖️",
        "Logistics / Trade": "Logistics / Trade 🚚",
        "Management / Business": "Management / Business 👔",
        "Manufacturing": "Manufacturing 🏭",
        "Marketing / Advertising": "Marketing / Advertising 📢",
        "Media / Communication": "Media / Communication 📡",
        "Public Social Work": "Public Social Work 💞",
        "Sales": "Sales 📈",
        "Other": "Other 💅",
    }

    # Calculate number of columns to use (max 4)
    num_columns = min(len(categories_major), 4)
    category_cols = st.columns(num_columns)

    # Initialize session state for selected category if not exists
    if 'selected_category' not in st.session_state:
        st.session_state.selected_category = categories_major[0]

    # Function to update the selected category
    def set_category(category):
        st.session_state.selected_category = category

    # Create buttons side by side in columns
    for i, category in enumerate(categories_major):
        col_idx = i % num_columns  # Determine which column to place this button
        with category_cols[col_idx]:
            # Use the custom label if available, otherwise use the original category name
            display_name = category_labels.get(category, category)
            
            st.button(
                display_name,
                key=f"cat_{category}",
                use_container_width=True,
                type="primary" if st.session_state.selected_category == category else "secondary",
                on_click=set_category,
                args=(category,)
            )

    # Use the selected category from session state
    selected_category = st.session_state.selected_category
    
    # 2. Comparison criteria buttons
    st.write("## Looking at 📊")
    comparison_metrics = [
        {"id": "skills", "label": "Skills Required 🛠️"},
        {"id": "job_description", "label": "Job Description 📄"},
        {"id": "requirements", "label": "Job Requirements 📋"}
    ]

    # Create columns for the metric buttons - one for each metric
    metric_cols = st.columns(len(comparison_metrics))

    # Initialize session state for selected metric if not exists
    if 'selected_metric' not in st.session_state:
        st.session_state.selected_metric = comparison_metrics[0]["id"]

    # Function to update the selected metric
    def set_metric(metric_id):
        st.session_state.selected_metric = metric_id

    # Create buttons for metrics side by side
    for i, metric in enumerate(comparison_metrics):
        with metric_cols[i]:
            st.button(
                metric["label"], 
                key=f"metric_{metric['id']}", 
                use_container_width=True,
                type="primary" if st.session_state.selected_metric == metric["id"] else "secondary",
                on_click=set_metric,
                args=(metric["id"],)
            )

    # Use the selected metric from session state
    selected_metric = st.session_state.selected_metric
    
    # Filter data by selected category
    filtered_by_category = df[df['category_major'] == selected_category]
    
    if filtered_by_category.empty:
        st.warning(f"No data available for the selected category: {selected_category}")
        return
    
    # 3. Display data for all seniority levels
    st.markdown("## Climbing the Ladder 📈")
    display_category = category_labels.get(selected_category, selected_category)
    st.write(f"Oh, wow! {len(filtered_by_category)} jobs in {display_category}")
    
    # Assign color maps to different seniority levels
    color_maps = {
        'Internship': 'Blues',
        'Entry level': 'GnBu',
        'Assistant': 'Greens',
        'Mid-Senior level': 'Purples',
        'Director': 'Reds',
        'Executive (VP, GM, C-Level)': 'Oranges'
    }
    
    # Default color map if specific level not found
    default_colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    
    # Fixed layout: 2 rows with 3 columns each
    columns_per_row = 3
    rows = 2
    
    # First filter the seniority levels to only include those with data
    seniority_with_data = []
    for level in seniority_levels:
        # Get data for this seniority level (no special handling needed for Internship now)
        level_data = filtered_by_category[filtered_by_category['seniority'] == level]
        
        # Only include levels that have data
        if not level_data.empty:
            seniority_with_data.append(level)
    
    # Check if we have any data to display
    if not seniority_with_data:
        st.warning(f"No seniority level data available for the selected category: {display_category}")
    else:
        # Process each seniority level that has data
        level_index = 0
        for row in range(rows):
            # Only create row if there are levels to display
            if level_index >= len(seniority_with_data):
                break
                
            # Create columns for this row
            cols = st.columns(columns_per_row)
            
            # Process the levels for this row
            for col in range(columns_per_row):
                # Check if we still have levels to process
                if level_index < len(seniority_with_data):
                    level = seniority_with_data[level_index]
                    
                    # Get data for this seniority level in the selected category
                    level_data = filtered_by_category[filtered_by_category['seniority'] == level]
                    
                    with cols[col]:
                        display_level = seniority_labels.get(level, level)
                        st.subheader(display_level)
                        st.write(f"Found: {len(level_data)}")
                        
                        # Process text based on selected comparison metric
                        if selected_metric == 'skills':
                            processed_text = " ".join(level_data['skills'].fillna('').apply(
                                lambda x: process_skills(x, custom_stopwords)
                            ))
                        else:
                            processed_text = " ".join(level_data[selected_metric].fillna('').apply(
                                lambda x: process_text(x, custom_stopwords)
                            ))

                        # Check if we have enough text data
                        if not processed_text or len(processed_text.strip()) < 50:  # Minimum threshold
                            st.info(f"Found {len(level_data)} job postings, but the extracted terms were insufficient.")
                        else:
                            # Generate word cloud
                            # Select colormap based on seniority level or default
                            colormap = color_maps.get(level, default_colormaps[level_index % len(default_colormaps)])
                            
                            # Generate and display word cloud
                            fig = generate_wordcloud(
                                processed_text,
                                f"{level} - Top Terms",
                                colormap
                            )
                            if fig:
                                st.pyplot(fig)
                                
                                # Display top 10 terms
                                word_freq = Counter(processed_text.split())
                                if word_freq:
                                    # Get the most common 10 terms
                                    most_common = word_freq.most_common(10)
                                    
                                    # Create dataframe with 1-based indexing
                                    freq_df = pd.DataFrame(most_common, columns=['Term', 'Frequency'])
                                    freq_df.index = range(1, len(freq_df) + 1)  # Start index from 1 instead of 0
                                    
                                    st.write("Top 10 Terms:")
                                    st.dataframe(freq_df)
                                else:
                                    st.info("No meaningful terms could be extracted after removing common stopwords.")
                    
                    # Move to the next level
                    level_index += 1
                else:
                    # No more levels to display
                    break
            
            # Only add separator if we have another row with data
            if level_index < len(seniority_with_data):
                st.markdown("---")
    
    # Insights section at the bottom
    st.markdown("## Insights from the Comparison")
    st.write("""
    ### Observations:
    - Compare the tag clouds to see how job requirements evolve with seniority within the same category
    - Note which skills are common across all levels versus those that appear only at specific levels
    - Pay attention to terms that increase in prominence as seniority increases
    
    ### Career Path Planning:
    - Use these visualizations to understand skill progression for career advancement in your chosen field
    - Identify skills to develop for moving to the next seniority level
    - Compare how requirements differ across job categories to identify transferable skills
    """)

if __name__ == "__main__":
    main()