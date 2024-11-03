#!/usr/bin/env python
# coding: utf-8

# In[1]:


text = "Written collaboratively by largely anonymous volunteers known as Wikipedians, Wikipedia articles can be edited by anyone with Internet access, except in limited cases where editing is restricted to prevent disruption or vandalism. Since its creation on January 15, 2001, it has grown into the world's largest reference website, attracting over a billion visitors each month. Wikipedia currently has more than sixty-three million articles in more than 300 languages, including 6,886,397 articles in English, with 116,686 active contributors in the past month.Wikipedias fundamental principles are summarized in its five pillars. While the Wikipedia community has developed many policies and guidelines, new editors do not need to be familiar with them before they start contributing."


# In[2]:


len(text)


# In[3]:


pip install spacy


# In[7]:


import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation


# In[8]:


nlp = spacy.load('en_core_web_sm')


# In[9]:


nlp(text)


# In[10]:


doc = nlp(text)


# In[11]:


tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct and token.text!= "\n"]


# In[12]:


tokens


# In[13]:


from collections import Counter


# In[14]:


Counter(tokens)


# In[15]:


word_freq = Counter(tokens)


# In[16]:


max_freq = max(word_freq.values())


# In[17]:


max_freq


# In[18]:


for word in word_freq.keys():
    word_freq[word] = word_freq[word]/max_freq


# In[19]:


word_freq


# In[20]:


sent_toekn = [sent.text for sent in doc.sents]


# In[21]:


sent_toekn


# In[22]:


sent_score = {}
for sent in sent_toekn:
    for word in sent.split():
        if word in word_freq:
            if sent not in sent_score:
                sent_score[sent] = word_freq[word]  # Initialize score
            else:
                sent_score[sent] += word_freq[word]  # Increment score
print(sent_score)


# In[23]:


import pandas as pd


# In[24]:


pd.DataFrame(list(sent_score.items()),columns = ['Sentence','Score'])


# In[25]:


from heapq import nlargest


# In[26]:


num_sentences = 3
n=nlargest(num_sentences,sent_score, key=sent_score.get)


# In[27]:


" ".join(n)


# In[28]:


import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
from tkinter import END

# Importing the summarization code
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest

def summarize_text():
    # Get text from the text box
    text = text_box.get("1.0", "end-1c")
    
    # Loading spaCy model
    nlp = spacy.load('en_core_web_sm')
    
    # Tokenization and removing stopwords
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc 
              if not token.is_stop and not token.is_punct and token.text != '\n']

    # Calculating word frequency
    word_freq = Counter(tokens)
    if not word_freq:
        messagebox.showerror("Error", "No words found in the text.")
        return
    
    max_freq = max(word_freq.values())
    for word in word_freq.keys():
        word_freq[word] = word_freq[word]/max_freq
    
    # Sentence tokenization
    sent_token = [sent.text for sent in doc.sents]

    sent_score = {}
    for sent in sent_token:
        for word in sent.split():
            if word.lower() in word_freq.keys():
                if sent not in sent_score.keys():
                    sent_score[sent] = word_freq[word]
                else:
                    sent_score[sent] += word_freq[word]

    # Select top-scoring sentences based on user input
    num_sentences = int(num_sentences_entry.get())
    summarized_sentences = nlargest(num_sentences, sent_score, key=sent_score.get)

    # Display summarized text in the result box
    result_box.delete(1.0, END)
    result_box.insert(END, " ".join(summarized_sentences))

# GUI setup
root = tk.Tk()
root.title("Text Summarizer")

# Text box for input
text_box = scrolledtext.ScrolledText(root, width=50, height=10, wrap=tk.WORD)
text_box.pack(pady=10)

# Entry field for the number of sentences
num_sentences_label = tk.Label(root, text="Number of Sentences:")
num_sentences_label.pack()
num_sentences_entry = tk.Entry(root, width=10)
num_sentences_entry.insert(END, "3")  # Default value
num_sentences_entry.pack()

# Button to summarize
summarize_button = tk.Button(root, text="Summarize", command=summarize_text)
summarize_button.pack(pady=5)

# Result box for output
result_box = scrolledtext.ScrolledText(root, width=50, height=5, wrap=tk.WORD)
result_box.pack(pady=10)

root.mainloop()


# In[29]:


pip install pipline 


# In[30]:


from transformers import pipeline


# In[ ]:


summarizer =pipeline("summarization", model="t5-base", tokenizers ="th-base", framework="pt")


# In[ ]:


text="""In a world often dominated by negativity, it's important to remember the power of kindness and compassion. Small acts of kindness have the ability to brighten someone's day, uplift spirits, and create a ripple effect of positivity that can spread far and wide. Whether it's a smile to a stranger, a helping hand to a friend in need, or a thoughtful gesture to a colleague, every act of kindness has the potential to make a difference in someone's life.Beyond individual actions, there is also immense power in collective efforts to create positive change. When communities come together to support one another, incredible things can happen. From grassroots initiatives to global movements, people are uniting to tackle pressing social and environmental issues, driving meaningful progress and inspiring hope for a better future.It's also important to recognize the strength that lies within each and every one of us. We all have the ability to make a positive impact, no matter how small our actions may seem. By tapping into our innate compassion and empathy, we can cultivate a culture of kindness and empathy that enriches our lives and those around us.So let's embrace the power of kindness, and strive to make the world a better place one small act at a time. Together, we can create a brighter, more compassionate future for all."""


# In[ ]:


text


# In[ ]:


summary = summarizer(text,max_length=100,min_length=10,do_sample=False)


# In[ ]:


print(summary[0]['summary_text'])


# In[ ]:


import tkinter as tk
from transformers import pipeline

def summarize_text():
    # Get text from the input text box
    text = text_entry.get("1.0", "end-1c")

    # Summarize the text
    summary = summarizer(text, max_length=100, min_length=10, do_sample=False)

    # Update the output text box with the summary
    output_text.delete("1.0", "end")
    output_text.insert("1.0", summary[0]['summary_text'])

# Create a Tkinter window
window = tk.Tk()
window.title("Text Summarizer")

# Create input text box
text_entry = tk.Text(window, height=10, width=60)
text_entry.pack(pady=10)

# Create a button to trigger text summarization
summarize_button = tk.Button(window, text="Summarize", command=summarize_text)
summarize_button.pack()

# Create output text box
output_text = tk.Text(window, height=10, width=60)
output_text.pack(pady=10)

# Initialize the summarizer pipeline
summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="pt")

# Run the Tkinter event loop
window.mainloop()


# In[ ]:





# In[ ]:





# In[ ]:




