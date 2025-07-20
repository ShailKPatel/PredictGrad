import streamlit as st
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import json
import os
from collections import Counter, deque
import re
import numpy as np

# File Paths
REVIEW_PATH = "reviews/recent_reviews.json"
WORD_COUNT_PATH = "reviews/word_count.json"
REVIEW_LIMIT = 6

# Expanded filter words to prevent "lol" variations and vandalism
FILTER_WORDS = {
    "lol", "lolis", "laughing", "out", "loud",
    "haha", "hehe", "lmao", "rofl"
}

# Load reviews
def load_reviews():
    if os.path.exists(REVIEW_PATH):
        with open(REVIEW_PATH, "r", encoding="utf-8") as file:
            return deque(json.load(file), maxlen=REVIEW_LIMIT)
    return deque(maxlen=REVIEW_LIMIT)

# Save reviews
def save_reviews(reviews):
    with open(REVIEW_PATH, "w", encoding="utf-8") as file:
        json.dump(list(reviews), file)

# Load word count
def load_word_count():
    if os.path.exists(WORD_COUNT_PATH):
        with open(WORD_COUNT_PATH, "r", encoding="utf-8") as file:
            return Counter(json.load(file))
    return Counter()

# Save word count
def save_word_count(counter):
    with open(WORD_COUNT_PATH, "w", encoding="utf-8") as file:
        json.dump(dict(counter), file)

# Filter words: must be longer than 1 character, alphabetic, and not in stop/filter words
def should_count_word(word):
    word = word.lower()
    if re.search(r'l+o+l+[!@#$%^&*]*', word):
        return False
    if word in STOPWORDS or word in FILTER_WORDS:
        return False
    if len(word) <= 1:
        return False  # Remove single characters
    if word.isdigit():
        return False  # Remove numbers
    return True

# Check if review contains "lol" or filtered words
def contains_filtered_words(review):
    words = review.split()
    for word in words:
        if re.search(r'l+o+l+[!@#$%^&*]*', word.lower()):
            return True
        if any(fw in word.lower() for fw in FILTER_WORDS) or word.lower() in FILTER_WORDS:
            return True
    return False

# Initialize data
review_queue = load_reviews()
word_count = load_word_count()

# Streamlit UI
st.set_page_config(page_title="Feedback", layout="centered", page_icon='💬')

st.title("📝 Feedbacks")

st.subheader("📜 Recent Reviews")
if review_queue:
    for review in review_queue:
        st.write(f"✍️ {review}")
else:
    st.write("No reviews yet. Be the first to leave your mark! ✨")

st.subheader("📊 Word Cloud")
if word_count:
    # Filter word_count to remove unwanted words
    filtered_word_count = Counter()
    for word, freq in word_count.items():
        if should_count_word(word):
            filtered_word_count[word.lower()] += freq

    if filtered_word_count:
        # Custom color function
        def random_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            colors = ["rgb(73,11,61)", "rgb(189,30,81)", "rgb(241,184,20)", "rgb(128,173,204)"]
            return np.random.choice(colors)

        # Create word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            max_words=50,
            min_font_size=10,
            scale=15,
            stopwords=STOPWORDS.union(FILTER_WORDS),
            color_func=random_color_func
        ).generate_from_frequencies(filtered_word_count)

        # Show word cloud
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.write("No valid words to display in the word cloud after filtering.")
else:
    st.write("No words to display in the word cloud yet. Submit a review!")

# Submit new review
user_review = st.text_area("Got a complaint or suggestion? Drop it here", "")
if st.button("Submit Review"):
    if user_review:
        if not contains_filtered_words(user_review):
            review_queue.append(user_review)
            save_reviews(review_queue)

            words = re.findall(r'\b\w+\b', user_review.lower())
            for word in words:
                if should_count_word(word):
                    word_count[word] += 1
            save_word_count(word_count)

        st.balloons()
        st.success("Review submitted! Thanks for sharing your thoughts!")
        st.rerun()
    else:
        st.warning("Please write something before submitting.")
