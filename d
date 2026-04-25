#content analysis


    import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

#INSPECT
df = pd.read_csv("sample_engagement.csv")
print("\n Information of Dataset \n")
print(df.info)
print("\n Description of Dataset \n")
print(df.describe())
print("\n First 5 tuples of Dataset \n")
print(df.head())
print("\n Data Types of Dataset \n")
print(df.dtypes)
print("\n Size of Dataset \n")
print(df.shape)


#CLEANING
df=df.dropna().drop_duplicates()

def cleaning(text):
    text=re.sub(r"http\S+|#\w+|@\w+|[^a-z\s]","",str(text).lower())
    return text

df["review"]=df["review"].apply(cleaning)
df["content"]=df["content"].apply(cleaning)

q1= df["likes"].quantile(0.25)
q3= df["likes"].quantile(0.75)
iqr=q3-q1
df=df[(df["likes"]>=q1-1.5*iqr) &  (df["likes"]<= q3+1.5*iqr)]

df["location"]=df["location"].fillna("Unkown").astype(str).str.strip().str.title()

df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce")
df["date"]=df["timestamp"].dt.date
        
vector = CountVectorizer(
    stop_words="english",
    min_df=2,               # ignore rare words
    max_df=0.9,              # ignore overly common words
    ngram_range=(1, 2)      # consider both unigrams and bigrams
)
X = vector.fit_transform(df["clean_text"])

lda = LatentDirichletAllocation(
    n_components=3,
    random_state=1,
    max_iter=10
)

lda.fit(X)

feature_names = vector.get_feature_names_out()
print(feature_names)

for i, topic in enumerate(lda.components_):
    words = topic.argsort()[-5:]
    print("Topic", i, ":", [feature_names[j] for j in words])

# Visualization
vis = lda_model.prepare(lda, X, vector)
pyLDAvis.save_html(vis, "lda_vis.html")

print("Visualization saved: lda_vis.html")
print("Interpretation: read top words and label topics.")

#Location

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# -----------------------------
# 1. Inspect Data
# -----------------------------
print(df.head())
print(df.info())
print(df.describe())

print("\nNull values:\n", df.isnull().sum())
print("\nDuplicates BEFORE:\n", df.duplicated().sum())

# -----------------------------
# 2. Handle Location
# -----------------------------

locations = ["Mumbai", "Delhi", "Pune", "Bangalore", "Chennai", "Hyderabad", "Kolkata", "New York"]

def get_location(text):
    text = str(text)
    for loc in locations:
        if loc.lower() in text.lower():
            return loc
    return "Unknown"

df["location"] = df["tweet"].apply(get_location)

# -----------------------------
# 3. Clean Data
# -----------------------------
df=df[df['location'].notna()]
df = df[df["location"] != ""]
df = df.drop_duplicates()
df["location"] = df["location"].fillna("").astype(str).str.strip().str.title()
#df["location"] = df["location"].fillna("Unknown")

print("\nDuplicates AFTER:\n", df.duplicated().sum())

# -----------------------------
# 4. Analysis
# -----------------------------
location_counts = df["location"].value_counts()
top_locations = location_counts.head(8)

print("\nTop locations:\n", top_locations)

# -----------------------------
# 5. Visualization
# -----------------------------

# Bar Chart
top_locations.plot(kind="barh")
plt.title("Top Locations")
plt.show()

# Pie Chart
top_locations.plot(kind="pie", autopct="%1.1f%%")
plt.ylabel("")
plt.title("Location Share")
plt.show()

# Heatmap
sns.heatmap([top_locations.values], annot=True, cmap="YlOrRd",xticklabels =top_locations.index,yticklabels="")
plt.title("Heatmap")
plt.show()


plt.plot(top_locations.index, top_locations.values, marker='o')
plt.title("Location Trend Line")
plt.xlabel("Location")
plt.ylabel("Count")

plt.xticks(rotation=45)
plt.show()

# -----------------------------
# 6. Interpretation
# -----------------------------
print("\nInterpretation:")
print("1) Bar chart shows highest tweet locations.")
print("2) Pie chart shows location share in %.")
print("3) Heatmap highlights high and low location counts.")
print("4) Line plot shows trend of tweet frequency across locations.")


#Trend timeline

print(df.head())
print(df.info())
print(df.isnull().sum())

# 4) Clean + convert to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"]).drop_duplicates()

# 5) Group by date/hour/day
df["date"] = df["timestamp"].dt.date
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day_name()
#df["month"]=df["timestamp"].dt.month

# 6) Visualize


df.groupby('date').size().plot()
plt.title("Tweet Activity by Date")
plt.xlabel("Date")
plt.ylabel("Number of Tweets")
plt.show()

df.groupby('hour').size().plot()
plt.show()

# df.groupby('month').size().plot()
# plt.xticks(range(12), ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], rotation=45)
# plt.show()

days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
df.groupby('day').size().reindex(days).plot(kind='bar', color='coral')
plt.show()


# hashtag

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

#df = pd.read_csv("sample_hashtag.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())

df = df.dropna(subset=["tweet", "user_group"]).drop_duplicates()
if "hashtag" in df.columns:
	df["hashtags"] = df["hashtag"].fillna("").str.lower().str.replace(",", " ").str.findall(r"#\w+")
else:
	df["hashtags"] = df["tweet"].str.lower().str.findall(r"#\w+")
df = df.explode("hashtags").dropna(subset=["hashtags"])

overall_counts = df["hashtags"].value_counts().head(10)
group_counts = (
	df.groupby("user_group")["hashtags"].value_counts().rename("freq").reset_index()
)

print("\nTop hashtags overall:\n", overall_counts)
print("\nTop hashtags by user group:\n")
print(group_counts.sort_values(["user_group", "freq"], ascending=[True, False]).groupby("user_group").head(3))

overall_counts.plot(kind="bar")
plt.show()

df["hashtags"] = df["hashtags"].str.replace("#", "", regex=False)

freq = df["hashtags"].value_counts()
wc = WordCloud(width=900, height=450, background_color="white").generate_from_frequencies(freq.to_dict())
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()


print("\nInterpretation: Most frequent hashtags are the most popular. Group table shows popularity by user group.")


#sentiment 

import re
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

#df = pd.read_csv("sample_sentiment.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())

df = df.dropna(subset=["text"]).drop_duplicates()
df["clean_text"] = df["text"].str.lower().str.replace(r"[^a-z\s]", " ", regex=True)

sentiments = []
for t in df["clean_text"]:
    p = TextBlob(t).sentiment.polarity
    if p > 0:
        sentiments.append("Positive")
    elif p < 0:
        sentiments.append("Negative")
    else:
        sentiments.append("Neutral")

df["sentiment"] = sentiments
sent_counts = df["sentiment"].value_counts()
print("\nSentiment counts:\n", sent_counts)

sent_counts.plot(kind="pie", autopct="%1.1f%%")
plt.ylabel("")
plt.show()

sent_counts.plot(kind='bar',title='sentiment')
plt.show()

# stop = set(stopwords.words('english'))
stop = set({"a","an","the","and","or","but","if","while","is","am","are","was","were",
"be","been","being","have","has","had","do","does","did",
"of","to","in","for","on","with","as","by","at","from",
"this","that","these","those","it","its","i","me","my","we","our","you","your",
"he","him","his","she","her","they","them","their",
"so","because","about","into","over","after","before","between","during",
"very","just","more","most","some","such","no","not","only","own","same"})

all_words = ' '.join(df['clean_text'].astype(str)).lower()
all_words = re.sub(r'http\S+', '', all_words)   # remove URLs
all_words = re.sub(r'@\w+', '', all_words)       # remove mentions
all_words = re.sub(r'[^a-z\s]', '', all_words)  # remove punctuation
words = [w for w in all_words.split() if w not in ENGLISH_STOP_WORDS and len(w) > 2]

freq = pd.Series(words).value_counts()
wc = WordCloud(width=900, height=450, background_color="white").generate_from_frequencies(freq.to_dict())
plt.imshow(wc)
plt.axis('off')
plt.title('Keyword Cloud')
plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words="english")
X = tfidf.fit_transform(df["clean_text"])

scores = X.sum(axis=0).A1
words = tfidf.get_feature_names_out()
top = pd.Series(scores, index=words).sort_values(ascending=False).head(10)
print(top)
majority = df['sentiment'].value_counts().idxmax()
print("\nMajority sentiment:", majority)

print("\nInterpretation: Pie chart shows overall sentiment; word cloud shows most frequent keywords.")

#user engagement

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print(df.head())
print(df.info())
print(df.isnull().sum())

df = df.dropna().drop_duplicates()
df["engagement"] = df["likes"] + df["comments"] + df["shares"]

content_eng = df.groupby("content_type")["engagement"].mean().sort_values(ascending=False)
user_eng = df.groupby("user")["engagement"].mean().sort_values(ascending=False)

top_post = df.sort_values("engagement", ascending=False).head(1)
print("\nHighest engagement post:\n", top_post)
print("\nTop content by avg engagement:\n", content_eng)
print("\nTop users by avg engagement:\n", user_eng.head(5))

# Chart 1: Scatter (likes vs engagement)
plt.scatter(df["likes"], df["engagement"])
plt.show()

# Chart 2: Bar (content type vs engagement)
content_eng.plot(kind="bar")
plt.show()

# Chart 3: Correlation heatmap
corr = df[["likes", "comments", "shares", "engagement"]].corr()
sns.heatmap(corr, annot=True)
plt.show()

totals = pd.Series(
    [df['likes'].sum(), df['shares'].sum(), df['comments'].sum()],
    index=["Likes", "Shares", "Comments"]
)
totals.plot(kind="pie", autopct="%1.1f%%")
plt.ylabel("")
plt.show()

print("\nInterpretation: Content types and users with higher average engagement are more engaging.")

#eda

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Shape:", df.shape)
print("\nDtypes:\n", df.dtypes)
print("\nNulls:\n", df.isnull().sum())
print("\nSummary stats:\n", df.describe(include="all"))

# Missing values heatmap
sns.heatmap(df.isnull(),  yticklabels=False)
plt.show()

# Histograms
num = df.select_dtypes(include="number")
if num.shape[1] > 0:
    num.hist(bins=10)
    plt.suptitle("Distribution of Numeric Features")
    plt.show()


# Bar chart
if num.shape[1] > 0:
	num.mean().plot(kind="bar")
	plt.show()

# Scatter chart
if num.shape[1] >= 2:
	plt.scatter(num.iloc[:, 0], num.iloc[:, 1])
	plt.xlabel(num.columns[0])
	plt.ylabel(num.columns[1])
	plt.show()

# Pie chart
cat = df.select_dtypes(exclude="number")
if cat.shape[1] > 0:
	df[cat.columns[0]].astype(str).value_counts().head(5).plot(kind="pie", autopct="%1.1f%%")
elif num.shape[1] > 0:
	num.iloc[:, 0].value_counts().head(5).plot(kind="pie", autopct="%1.1f%%")
plt.ylabel("")
plt.show()



# Box plot
if num.shape[1] > 0:
	sns.boxplot(data=num, orient="h")
	plt.show()

# Correlation heatmap
if num.shape[1] > 1:
	sns.heatmap(num.corr(), annot=True)
	plt.show()

print("\nInterpretation: EDA shows data quality, spread, outliers, and relationships between numeric features.")

#brand
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

brand_name = "TechNova"
# df = pd.read_csv("sample_brand.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())

df = df.dropna(subset=["brand", "text", "timestamp"]).drop_duplicates()
df = df[df["brand"].str.lower()==(brand_name.lower())].copy()
print(f"\nPosts for {brand_name}:", len(df))
if df.empty:
    raise SystemExit("No brand posts found")

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])
df["clean"] = df["text"].str.lower().str.replace(r"[^a-z\s]", " ", regex=True)
df["engagement"] = df[["likes", "comments", "shares"]].sum(axis=1)

sentiments = []
for t in df["clean"]:
    p = TextBlob(t).sentiment.polarity
    if p > 0:
        sentiments.append("Positive")
    elif p < 0:
        sentiments.append("Negative")
    else:
        sentiments.append("Neutral")

df["sentiment"] = sentiments

sent_counts = df["sentiment"].value_counts()
print("\nSentiment counts:\n", sent_counts)

sent_counts.plot(kind="pie", autopct="%1.1f%%")
plt.ylabel("")
plt.show()

words = [
    w
    for t in df["clean"]
    for w in t.split()
    if w not in ENGLISH_STOP_WORDS and len(w) > 2
]
freq = pd.Series(words).value_counts()
wc = WordCloud(width=900, height=450, background_color="white").generate_from_frequencies(freq.to_dict())
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()

daily = df.groupby(df["timestamp"].dt.date)["engagement"].sum()
daily.plot(marker="o")
plt.xticks(rotation=45)
plt.show()

top_post = df.sort_values("engagement", ascending=False).head(1)
print("\nHighest engagement brand post:\n", top_post)

#competitive

import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# df = pd.read_csv("sample_brand.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())

df = df.dropna(subset=["brand", "text", "likes", "comments", "shares"]).drop_duplicates()
df["brand_tag"] = df["brand"].str.strip().str.title()
df["engagement"] = df[["likes", "comments", "shares"]].sum(axis=1)
df["clean"] = df["text"].str.lower().str.replace(r"[^a-z\s]", " ", regex=True)

sentiment = []
for t in df["clean"]:
    p = TextBlob(t).sentiment.polarity
    if p > 0:
        sentiment.append("Positive")
    elif p < 0:
        sentiment.append("Negative")
    else:
        sentiment.append("Neutral")
df["sentiment"] = sentiment

mentions = df["brand_tag"].value_counts()
engagement_avg = df.groupby("brand_tag")["engagement"].mean().sort_values(ascending=False)
sent_table = pd.crosstab(df["brand_tag"], df["sentiment"])

print("\nMentions by brand:\n", mentions)
print("\nAverage engagement by brand:\n", engagement_avg)
print("\nSentiment by brand:\n", sent_table)

# Plot 1: Mentions comparison
mentions.plot(kind="pie", autopct="%1.1f%%")
plt.ylabel("")
plt.show()

# Plot 2: Engagement comparison
engagement_avg.plot(kind="bar")
plt.show()

# Plot 3: Sentiment comparison
sent_table.plot(kind="bar", stacked=True)
plt.show()

print("\nInterpretation: Compare mentions, engagement, sentiment, and top keywords to analyze competitor activity.")

#network graph

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community import girvan_newman
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Extract mentions
df["mentions"] = df["text"].str.findall(r"@\w+")

# --- Create Graph (simpler using explode) ---
edges = df.explode("mentions")
edges["mentioned"] = edges["mentions"].str[1:].str.lower()
edges["user"] = edges["user"].str.lower()

G = nx.from_pandas_edgelist(edges, "user", "mentioned")

# Top users
print("Top users:\n", pd.Series(nx.degree_centrality(G)).nlargest(5))

# Communities
if G.number_of_edges() > 0:
    print("Communities:", list(next(girvan_newman(G))))
else:
    print("Nodes:", list(G.nodes()))

# Draw graph
nx.draw(G, with_labels=True)
plt.title("Mention Network")
plt.show()

# --- KMeans Clustering ---
vec = TfidfVectorizer(stop_words="english")
X = vec.fit_transform(df["text"])

km = KMeans(n_clusters=3, random_state=42)
df["cluster"] = km.fit_predict(X)

# Top words per cluster (shorter)
terms = vec.get_feature_names_out()
for i, center in enumerate(km.cluster_centers_):
    print("Cluster", i, ":", [terms[j] for j in center.argsort()[-5:]])

# Plot cluster sizes
df["cluster"].value_counts().plot(kind="bar", title="Cluster Sizes")
plt.show()
