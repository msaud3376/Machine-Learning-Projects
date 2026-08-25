import streamlit as st
import nltk
import string
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data with error handling
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab')
        except:
            nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

# Download NLTK data at the start
download_nltk_data()

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

class NLPToolkit:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vocabulary = None
        self.vectorizer = None
        self.naive_bayes_model = None
        self.word_probabilities = None
        self.class_probabilities = None
        
    def text_preprocessing(self, text, method='lemmatization', remove_stopwords=True):
        """
        Preprocess text with lower-casing, tokenization, stop-word removal, and stemming/lemmatization
        """
        # Lower casing
        text = text.lower()
        
        # Tokenization with error handling
        try:
            tokens = word_tokenize(text)
        except LookupError:
            tokens = text.split()
        
        # Remove punctuation and numbers
        tokens = [token for token in tokens if token not in string.punctuation and not token.isdigit()]
        
        # Stop-word removal
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Stemming or Lemmatization
        if method == 'stemming':
            tokens = [self.stemmer.stem(token) for token in tokens]
        elif method == 'lemmatization':
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def create_vocabulary(self, documents, min_freq=1):
        """
        Create vocabulary from a list of documents
        """
        all_tokens = []
        for doc in documents:
            tokens = self.text_preprocessing(doc)
            all_tokens.extend(tokens)
        
        # Count word frequencies
        word_freq = Counter(all_tokens)
        
        # Create vocabulary with words meeting minimum frequency
        self.vocabulary = {word: idx for idx, (word, freq) in enumerate(word_freq.items()) 
                          if freq >= min_freq}
        
        return self.vocabulary
    
    def bow_representation(self, documents):
        """
        Create Bag-of-Words representation
        """
        if self.vectorizer is None:
            self.vectorizer = CountVectorizer(
                preprocessor=lambda x: ' '.join(self.text_preprocessing(x)),
                min_df=1
            )
            bow_matrix = self.vectorizer.fit_transform(documents)
        else:
            bow_matrix = self.vectorizer.transform(documents)
        
        return bow_matrix, self.vectorizer.get_feature_names_out()
    
    def tfidf_representation(self, documents):
        """
        Create TF-IDF representation
        """
        tfidf_vectorizer = TfidfVectorizer(
            preprocessor=lambda x: ' '.join(self.text_preprocessing(x)),
            min_df=1
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
        return tfidf_matrix, tfidf_vectorizer.get_feature_names_out()
    
    def train_naive_bayes(self, documents, labels):
        """
        Train Naive Bayes classifier using scikit-learn's reliable implementation
        """
        # Preprocess documents
        preprocessed_docs = [' '.join(self.text_preprocessing(doc)) for doc in documents]
        
        # Train Naive Bayes model
        self.vectorizer = CountVectorizer()
        X = self.vectorizer.fit_transform(preprocessed_docs)
        
        self.naive_bayes_model = MultinomialNB(alpha=1.0)
        self.naive_bayes_model.fit(X, labels)
        
        return self.naive_bayes_model
    
    def predict_phrase_probability(self, phrase, target_class):
        """
        Calculate probability of a phrase using the trained model
        """
        if self.naive_bayes_model is None:
            return 0.0
        
        # Preprocess and transform the phrase
        tokens = self.text_preprocessing(phrase)
        preprocessed_phrase = ' '.join(tokens)
        X_phrase = self.vectorizer.transform([preprocessed_phrase])
        
        # Get probabilities from the trained model
        probabilities = self.naive_bayes_model.predict_proba(X_phrase)[0]
        class_probs = dict(zip(self.naive_bayes_model.classes_, probabilities))
        
        return class_probs.get(target_class, 0.0)
    
    def calculate_all_probabilities(self, phrase):
        """
        Calculate probabilities for all classes
        """
        if self.naive_bayes_model is None:
            return {}
        
        tokens = self.text_preprocessing(phrase)
        preprocessed_phrase = ' '.join(tokens)
        X_phrase = self.vectorizer.transform([preprocessed_phrase])
        
        probabilities = self.naive_bayes_model.predict_proba(X_phrase)[0]
        return dict(zip(self.naive_bayes_model.classes_, probabilities))
    
    def analyze_text_statistics(self, text):
        """Analyze basic text statistics"""
        # Original text stats
        try:
            original_words = word_tokenize(text)
            sentence_count = len(sent_tokenize(text))
        except LookupError:
            original_words = text.split()
            sentence_count = len([s for s in text.split('.') if s.strip()])
        
        original_word_count = len(original_words)
        original_char_count = len(text)
        
        # Preprocessed text stats
        preprocessed_tokens = self.text_preprocessing(text)
        preprocessed_word_count = len(preprocessed_tokens)
        unique_words = len(set(preprocessed_tokens))
        
        return {
            'original_word_count': original_word_count,
            'original_char_count': original_char_count,
            'sentence_count': sentence_count,
            'preprocessed_word_count': preprocessed_word_count,
            'unique_words': unique_words,
            'tokens': preprocessed_tokens
        }

def main():
    st.set_page_config(
        page_title="NLP Toolkit",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'toolkit' not in st.session_state:
        st.session_state.toolkit = NLPToolkit()
    if 'user_text' not in st.session_state:
        st.session_state.user_text = ""
    if 'nb_trained' not in st.session_state:
        st.session_state.nb_trained = False
    
    # Scenario-based dataset for Spam/Ham classification
    scenario_documents = [
        # Spam messages
        "Congratulations! You've won a $1000 gift card. Click here to claim your prize now!",
        "URGENT: Your bank account needs verification. Reply with your account details immediately.",
        "FREE iPhone 15 for our lucky winner! Claim your free device by clicking the link below.",
        "You have been selected for a special offer! Limited time discount on luxury watches.",
        "Your package delivery failed. Update your shipping information to receive your parcel.",
        "Investment opportunity: Double your money in 24 hours. Guaranteed returns!",
        "You qualify for a government grant! Apply now to receive $5000 in financial aid.",
        "Final notice: Your subscription will be canceled unless you update your payment method.",
        
        # Ham (legitimate) messages
        "Hi John, just checking if we're still meeting for lunch tomorrow at 1 PM?",
        "Your Amazon order #12345 has been shipped and will arrive by Friday.",
        "Reminder: Your doctor's appointment is scheduled for tomorrow at 3:30 PM.",
        "Thanks for your email. I'll review the documents and get back to you by EOD.",
        "Your monthly bank statement is now available for download in your online banking.",
        "Meeting rescheduled: The team meeting has been moved to Thursday at 10 AM.",
        "Your Netflix subscription will renew on May 15th for $15.99.",
        "Weather alert: Thunderstorms expected in your area this evening. Stay safe."
    ]
    scenario_labels = ['spam'] * 8 + ['ham'] * 8
    
    # Sidebar
    st.sidebar.title("📚 NLP Toolkit")
    st.sidebar.markdown("---")
    
    # Main content
    st.title("🧠 Natural Language Processing Toolkit")
    st.markdown("Perform various NLP tasks on your text data using this interactive toolkit.")
    
    # Text input section
    st.header("📝 Input Text")
    user_text = st.text_area(
        "Enter your text here:",
        value=st.session_state.user_text,
        height=150,
        placeholder="Type or paste your text here...",
        key="text_input"
    )
    
    if user_text:
        st.session_state.user_text = user_text
    
    st.markdown("---")
    
    # NLP Operations
    st.header("🔧 NLP Operations")
    
    # Create tabs for different operations (removed Classification tab)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Text Analysis", 
        "🔤 Text Preprocessing", 
        "📖 Vocabulary", 
        "🎯 Text Representations", 
        "🤖 Naive Bayes"
    ])
    
    with tab1:
        st.subheader("Text Statistics Analysis")
        if st.session_state.user_text:
            stats = st.session_state.toolkit.analyze_text_statistics(st.session_state.user_text)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Original Words", stats['original_word_count'])
            with col2:
                st.metric("Characters", stats['original_char_count'])
            with col3:
                st.metric("Sentences", stats['sentence_count'])
            with col4:
                st.metric("Unique Words", stats['unique_words'])
            
            st.subheader("Preprocessed Tokens")
            st.write(stats['tokens'])
        else:
            st.warning("Please enter some text to analyze.")
    
    with tab2:
        st.subheader("Text Preprocessing")
        if st.session_state.user_text:
            col1, col2 = st.columns(2)
            with col1:
                method = st.radio(
                    "Preprocessing Method:",
                    ["lemmatization", "stemming"],
                    horizontal=True
                )
            with col2:
                remove_stopwords = st.checkbox("Remove Stopwords", value=True)
            
            if st.button("Preprocess Text"):
                tokens = st.session_state.toolkit.text_preprocessing(
                    st.session_state.user_text, 
                    method=method, 
                    remove_stopwords=remove_stopwords
                )
                
                st.subheader("Preprocessed Result")
                st.write("**Tokens:**", tokens)
                
                # Show before and after
                col1, col2 = st.columns(2)
                with col1:
                    st.text_area("Original Text", st.session_state.user_text, height=150, disabled=True)
                with col2:
                    st.text_area("Processed Text", " ".join(tokens), height=150, disabled=True)
        else:
            st.warning("Please enter some text to preprocess.")
    
    with tab3:
        st.subheader("Vocabulary Creation")
        if st.session_state.user_text:
            min_freq = st.slider("Minimum Frequency", min_value=1, max_value=5, value=1)
            
            if st.button("Create Vocabulary"):
                vocab = st.session_state.toolkit.create_vocabulary([st.session_state.user_text], min_freq=min_freq)
                
                st.subheader("Vocabulary")
                st.write(f"**Total Words:** {len(vocab)}")
                
                # Display vocabulary in a nice format
                words_per_row = 5
                vocab_words = list(vocab.keys())
                
                for i in range(0, len(vocab_words), words_per_row):
                    cols = st.columns(words_per_row)
                    for j, word in enumerate(vocab_words[i:i+words_per_row]):
                        with cols[j]:
                            st.info(word)
        else:
            st.warning("Please enter some text to create vocabulary.")
    
    with tab4:
        st.subheader("Text Representations")
        if st.session_state.user_text:
            representation_type = st.radio(
                "Select Representation:",
                ["Bag-of-Words (BOW)", "TF-IDF"],
                horizontal=True
            )
            
            if st.button("Generate Representation"):
                if representation_type == "Bag-of-Words (BOW)":
                    bow_matrix, feature_names = st.session_state.toolkit.bow_representation([st.session_state.user_text])
                    st.subheader("Bag-of-Words Representation")
                else:
                    bow_matrix, feature_names = st.session_state.toolkit.tfidf_representation([st.session_state.user_text])
                    st.subheader("TF-IDF Representation")
                
                # Create a DataFrame for better visualization
                df = pd.DataFrame(
                    bow_matrix.toarray(),
                    columns=feature_names,
                    index=['Your Text']
                )
                
                st.write("**Feature Matrix:**")
                st.dataframe(df)
                
                st.write("**Shape:**", bow_matrix.shape)
                st.write("**Features:**", list(feature_names))
        else:
            st.warning("Please enter some text to generate representations.")
    
    with tab5:
        st.subheader("Naive Bayes Classifier - Spam Detection")
        
        st.info("""
        **Scenario: Spam vs Ham (Legitimate) Message Classification**
        This model is trained to distinguish between spam messages and legitimate messages.
        """)
        
        # Display the training dataset
        st.subheader("📋 Training Dataset")
        st.write("The model is trained on the following scenario-based data:")
        
        dataset_df = pd.DataFrame({
            'Message': scenario_documents,
            'Type': scenario_labels
        })
        st.dataframe(dataset_df, height=400)
        
        # Training section
        st.subheader("🚀 Train the Model")
        if st.button("Train Naive Bayes Model"):
            with st.spinner("Training Naive Bayes classifier on spam detection dataset..."):
                model = st.session_state.toolkit.train_naive_bayes(scenario_documents, scenario_labels)
                st.session_state.nb_trained = True
            
            st.success("✅ Naive Bayes model trained successfully!")
            st.write(f"**Classes:** {list(model.classes_)}")
            st.write(f"**Number of features:** {len(st.session_state.toolkit.vectorizer.get_feature_names_out())}")
        
        # Probability analysis section
        if st.session_state.nb_trained:
            st.subheader("🔍 Analyze Message Probability")
            
            # Option to use sample phrases from dataset
            st.write("**Try these sample phrases from the dataset:**")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Sample Spam Message"):
                    st.session_state.user_text = "Congratulations! You've won a $1000 gift card. Click here to claim your prize now!"
                    st.rerun()
            
            with col2:
                if st.button("Sample Ham Message"):
                    st.session_state.user_text = "Hi John, just checking if we're still meeting for lunch tomorrow at 1 PM?"
                    st.rerun()
            
            if st.session_state.user_text:
                st.write(f"**Current text to analyze:** '{st.session_state.user_text}'")
                
                # Calculate probabilities
                probabilities = st.session_state.toolkit.calculate_all_probabilities(st.session_state.user_text)
                
                if probabilities:
                    # Display probabilities with clear interpretation
                    st.subheader("📊 Probability Results")
                    
                    col1, col2 = st.columns(2)
                    
                    for i, (class_name, prob) in enumerate(probabilities.items()):
                        with col1 if i % 2 == 0 else col2:
                            if class_name == 'spam':
                                icon = "🚫"
                                color = "red" if prob > 0.5 else "green"
                            else:
                                icon = "✅"
                                color = "green" if prob > 0.5 else "red"
                            
                            st.metric(
                                f"{icon} P({class_name.upper()})", 
                                f"{prob:.4f}",
                                delta=f"{(prob-0.5)*100:+.1f}%" if len(probabilities) == 2 else None,
                                delta_color="normal" if color == "green" else "inverse"
                            )
                    
                    # Probability bar chart
                    prob_data = pd.DataFrame({
                        'Message Type': list(probabilities.keys()),
                        'Probability': list(probabilities.values())
                    })
                    st.bar_chart(prob_data.set_index('Message Type'))
                    
                    # Show interpretation
                    max_class = max(probabilities, key=probabilities.get)
                    max_prob = probabilities[max_class]
                    
                    if max_class == 'spam':
                        st.error(f"🚫 **This message is likely SPAM** (probability: {max_prob:.4f})")
                        st.write("**Characteristics:** Contains promotional language, urgency, or suspicious links")
                    else:
                        st.success(f"✅ **This message is likely HAM (Legitimate)** (probability: {max_prob:.4f})")
                        st.write("**Characteristics:** Normal conversation, personal communication, or legitimate notifications")
                    
                    # Show feature analysis
                    with st.expander("🔍 View Feature Analysis"):
                        tokens = st.session_state.toolkit.text_preprocessing(st.session_state.user_text)
                        st.write(f"**Preprocessed tokens:** {tokens}")
                        
                        feature_names = st.session_state.toolkit.vectorizer.get_feature_names_out()
                        present_features = [token for token in tokens if token in feature_names]
                        st.write(f"**Features found in vocabulary:** {present_features}")
                
                else:
                    st.error("Could not calculate probabilities. Please try again.")
            
            else:
                st.warning("Please enter some text to analyze or use the sample buttons above.")
        
        elif not st.session_state.nb_trained:
            st.warning("⚠️ Please train the Naive Bayes model first using the button above.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with ❤️ using Streamlit and NLTK | Spam Detection Scenario</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()