import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from collections import Counter
import string

st.title("Job Insights ✨")

# """Find out what is trending 📊"""

# Load the data
@st.cache_data
def load_data():
    # Load data just once
    df = pd.read_csv("data/jobs_processed.csv", encoding="utf-8-sig")
    return df

# Generate wordcloud with specific colormap
def generate_wordcloud(text, title, colormap):
    if text is None or pd.isna(text) or len(str(text).strip()) == 0:
        return None
    
    # Ensure text is a string
    text = str(text)
    
    # Filter out non-English characters (remove Chinese)
    text = re.sub(r'[\u4e00-\u9fff]+', '', text)  # Remove Chinese characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)    # Remove other non-ASCII characters
    
    # Clean and normalize text as in your friend's code
    text = re.sub(r"[【|】]", "", text)                        # Remove 【 and 】
    text = re.sub(r"：", ":", text)                           # Replace full-width colon
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)                          # Remove all digits
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Check if any meaningful content remains after filtering
    if not text or len(text) < 10:  # Require more content for meaningful wordcloud
        return None
    
    try:
        wc = WordCloud(
            background_color="white",
            max_words=100,
            max_font_size=80,
            width=800,
            height=400,
            colormap=colormap,
            prefer_horizontal=0.9,
            relative_scaling=0.5,
            random_state=42
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        plt.tight_layout(pad=0)
        return fig
    except Exception as e:
        print(f"Error generating word cloud: {e}")
        return None

# Load data
df = load_data()
df = load_data()

# Get unique category values
categories_major = sorted(df['category'].dropna().unique())
if not categories_major:
    st.error("No category data found in the dataset.")
    st.stop()

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

# 1. Category major buttons at the top
st.write("## A Job in 🏢")

# Create a mapping for category display names
category_labels = {
    "Bio, Medical": "🧬 Bio / Medical",
    "Catering / Food & Beverage": "🍔 Catering / Food & Beverage",
    "Construction": "🏗️ Construction",
    "Customer Service": "🛎️ Customer Service",
    "Design": "🎨 Design",
    "Education": "📚 Education",
    "Engineering": "🔧 Engineering",
    "Finance": "💵 Finance",
    "Game Production": "🎮 Game Production",
    "HR": "👩‍💼 HR",
    "IT": "💻 IT",
    "Law": "⚖️ Law",
    "Logistics / Trade": "🚚 Logistics / Trade",
    "Management / Business": "👔 Management / Business",
    "Manufacturing": "🏭 Manufacturing",
    "Marketing / Advertising": "📢 Marketing / Advertising",
    "Media / Communication": "📡 Media / Communication",
    "Public Social Work": "💞 Public Social Work",
    "Sales": "📈 Sales",
    "Other": "💅 Other",
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
st.write("## Looking at 😳")
comparison_metrics = [
    {"id": "skills", "label": "🛠️ Skills Required"},
    {"id": "job_description", "label": "📄 Job Description"},
    {"id": "requirements", "label": "📋 Job Requirements"}
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

# Map selected_metric to the processed column name
metric_column_map = {
    'skills': 'processed_skills',
    'job_description': 'processed_job_description',
    'requirements': 'processed_requirements'
}

processed_column = metric_column_map[selected_metric]

# Filter data by selected category
filtered_data = df[df['category'] == selected_category]

if filtered_data.empty:
    st.warning(f"No data available for the selected category: {selected_category}")
    st.stop()

# 3. Display data for all seniority levels
display_category = category_labels.get(selected_category, selected_category)

# Get total job count for this category from the dataset
total_jobs = filtered_data['count'].sum() if 'count' in filtered_data.columns else len(filtered_data)
st.markdown(f"## Oh, wow! {total_jobs} jobs in {display_category}")

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

# Get seniority levels that have data for this category
seniority_with_data = filtered_data['seniority'].tolist()

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
                level_row = filtered_data[filtered_data['seniority'] == level].iloc[0]
                job_count = level_row['count']
                processed_text = level_row[processed_column]
                
                with cols[col]:
                    display_level = seniority_labels.get(level, level)
                    st.subheader(display_level)
                    st.write(f"{job_count} job(s)")
                    
                    # Check if we have enough text data
                    if processed_text is None or pd.isna(processed_text):
                        st.info(f"Found {job_count} job(s), but no text data was available.")
                    else:
                        # Convert to string and check length
                        processed_text = str(processed_text)
                        if len(processed_text.strip()) < 30:  # Minimum threshold
                            st.info(f"Found {job_count} job(s), but the extracted terms were insufficient.")
                        else:
                            # Generate word cloud
                            # Select colormap based on seniority level or default
                            colormap = color_maps.get(level, default_colormaps[level_index % len(default_colormaps)])
                            
                            # Generate and display word cloud
                            fig = generate_wordcloud(
                                processed_text,
                                f"{display_level} - Top Terms",
                                colormap
                            )
                            if fig:
                                st.pyplot(fig)
                            else:
                                st.info(f"Found {job_count} job(s), but couldn't generate a word cloud from the available terms.")
                                                        
                # Move to the next level
                level_index += 1
            else:
                # No more levels to display
                break
