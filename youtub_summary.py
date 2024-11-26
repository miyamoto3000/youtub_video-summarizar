import pyttsx3  # Text-to-speech library
from sklearn.feature_extraction.text import TfidfVectorizer
from youtube_transcript_api import YouTubeTranscriptApi
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Function to change voice to a female voice
def set_female_voice():
    voices = engine.getProperty('voices')
    for voice in voices:
        if 'female' in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break
    engine.setProperty('rate', 150)  # Set speaking rate
    engine.setProperty('volume', 0.9)  # Set volume level

# Set the engine to use a gentle female voice
set_female_voice()

def get_video_id_from_url(url):
    """Extract video ID from YouTube URL."""
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    return None

def get_youtube_captions(video_id):
    """Fetch captions for a YouTube video."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        captions = " ".join([t['text'] for t in transcript])
        return captions
    except Exception as e:
        return str(e)

def segment_text(text, chunk_size=15):
    """Break text into chunks of a fixed size."""
    words = word_tokenize(text)
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def summarize_with_tfidf(context, summary_ratio=0.40):
    """Summarize text using TF-IDF."""
    sentences = segment_text(context)

    # Apply TF-IDF to the sentences
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    scores = X.sum(axis=1).A1  # Sum up the TF-IDF scores for each sentence

    # Select the top sentences based on their TF-IDF scores
    select_len = max(1, round(len(sentences) * summary_ratio))
    top_sentences = sorted(range(len(scores)), key=lambda i: -scores[i])[:select_len]

    summary = " ".join([sentences[i] for i in sorted(top_sentences)])
    return summary

def summarize_youtube_video(url):
    """Fetch captions from a YouTube video and summarize."""
    video_id = get_video_id_from_url(url)
    if not video_id:
        return {"error": "Invalid YouTube URL"}

    captions = get_youtube_captions(video_id)

    if "Could not retrieve" in captions:
        return {"error": "Error retrieving captions"}

    summary = summarize_with_tfidf(captions)

    # Calculate word counts
    original_word_count = len(word_tokenize(captions))
    summary_word_count = len(word_tokenize(summary))

    return {
        "summary": summary,
        "original_text": captions,
        "original_word_count": original_word_count,
        "summary_word_count": summary_word_count
    }

def ask_for_voice_output():
    """Ask the user if they want to hear the summary."""
    user_input = input("Do you want to hear the summary? (yes/no): ").strip().lower()
    return user_input == "yes"

def read_summary_aloud(summary):
    """Read the summary aloud using text-to-speech."""
    engine.say(summary)
    engine.runAndWait()

# Example Usage
youtube_url = "https://www.youtube.com/watch?v=6XUKdOSu_G8"  # Input YouTube URL here

result = summarize_youtube_video(youtube_url)

if "error" in result:
    print("Error:", result["error"])
else:
    print("Original Captions:\n", result["original_text"])
    print("\nSummary:\n", result["summary"])
    print("\nOriginal Word Count:", result["original_word_count"])
    print("Summary Word Count:", result["summary_word_count"])

    # Ask the user if they want to hear the summary
    if ask_for_voice_output():
        read_summary_aloud(result["summary"])
