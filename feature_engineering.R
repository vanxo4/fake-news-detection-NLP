################################################################################
# FEATURE ENGINEERING PIPELINE
# 1. Stylistic Features (Punctuation, Case)
# 2. Lexical Complexity (Word length, Diversity)
# 3. Subjectivity (Pronouns)
# 4. Sentiment Analysis (Polarity, Emotions)
# 5. Structural Consistency (Title-Body overlap)
################################################################################

# --- 1. Setup & Loading ---

if (!require("stringi")) install.packages("stringi")
if (!require("syuzhet")) install.packages("syuzhet")
library(stringi)
library(dplyr)
library(syuzhet)

cat("Loading prepared dataset...\n")
df <- read.csv("news_prepared.csv", stringsAsFactors = FALSE)

# Helper function for safe division
safe_div <- function(a, b) ifelse(b > 0, a / b, 0)

# --- 2. Stylistic Features (Punctuation & Case) ---

cat("Generating stylistic features...\n")

# Title Uppercase Ratio (Clickbait detection)
df$title_n_char <- nchar(df$title)
df$title_n_cap  <- stri_count_regex(df$title, "[A-Z]")
df$title_cap_ratio <- safe_div(df$title_n_cap, df$title_n_char)

# Text Punctuation (Emotional indicators)
df$n_exclam <- stri_count_fixed(df$text, "!")
df$n_quest  <- stri_count_fixed(df$text, "?")

df$exclam_ratio <- safe_div(df$n_exclam, df$n_char)
df$quest_ratio  <- safe_div(df$n_quest, df$n_char)

# Text Uppercase Ratio ("Shouting")
df$text_n_cap   <- stri_count_regex(df$text, "[A-Z]")
df$text_cap_ratio <- safe_div(df$text_n_cap, df$n_char)

# --- 3. Lexical Complexity ---

cat("Generating lexical complexity features...\n")

# Tokenization for statistics
lista_palabras <- stri_split_boundaries(df$text, type = "word", skip_word_none = TRUE)

# Average Word Length (Lower length often correlates with lower complexity)
df$avg_word_len <- sapply(lista_palabras, function(x) {
  if(length(x) == 0) return(0)
  mean(nchar(x))
})

# Lexical Diversity (Type-Token Ratio)
# Low ratio = repetitive vocabulary (common in bots/poorly written fakes)
df$lexical_diversity <- sapply(lista_palabras, function(x) {
  if(length(x) == 0) return(0)
  length(unique(tolower(x))) / length(x)
})

# --- 4. Subjectivity (Pronoun Usage) ---

cat("Counting personal pronouns...\n")

# Real news usually uses 3rd person. Fakes/Opinion often use 1st/2nd (I, we, you).
df$n_i   <- stri_count_regex(tolower(df$text), "\\bi\\b")
df$n_we  <- stri_count_regex(tolower(df$text), "\\bwe\\b")
df$n_you <- stri_count_regex(tolower(df$text), "\\byou\\b")

df$pronoun_sum <- df$n_i + df$n_we + df$n_you
df$pronoun_ratio <- safe_div(df$pronoun_sum, df$n_word)

# --- 5. Sentiment Analysis (Polarity) ---

cat("Calculating sentiment polarity (Syuzhet)...\n")

# Score: >0 (Positive), <0 (Negative), 0 (Neutral)
df$sentiment_score <- get_sentiment(df$text, method = "syuzhet")

# Magnitude: Absolute intensity of emotion
df$sentiment_magnitude <- abs(df$sentiment_score)

# --- 6. Title-Body Consistency ---

cat("Calculating Title-Body relationship...\n")

# Jaccard Overlap: How many title words appear in the body?
calc_overlap <- function(t, b) {
  w_title <- unlist(stri_split_boundaries(tolower(t), type="word", skip_word_none=TRUE))
  w_body  <- unlist(stri_split_boundaries(tolower(b), type="word", skip_word_none=TRUE))
  
  if(length(w_title) == 0 || length(w_body) == 0) return(0)
  
  intersect_len <- length(intersect(unique(w_title), unique(w_body)))
  return(intersect_len / length(unique(w_title)))
}

df$title_text_overlap <- mapply(calc_overlap, df$title, df$text)

# Length Ratio: Checks for disproportionate titles
df$title_text_len_ratio <- safe_div(df$title_n_char, df$n_char)

# --- 7. Granular Emotions (NRC Lexicon) ---

cat("Extracting specific emotions (NRC)...\n")

# Extracts specific counts for Anger, Fear, Joy, etc.
emotions <- get_nrc_sentiment(df$text)

# Normalize by text length
df$ratio_anger   <- safe_div(emotions$anger, df$n_word)
df$ratio_fear    <- safe_div(emotions$fear, df$n_word)
df$ratio_disgust <- safe_div(emotions$disgust, df$n_word)
df$ratio_joy     <- safe_div(emotions$joy, df$n_word)

# --- 8. Uncertainty Indicators (Hedge Words) ---

cat("Searching for hedge words...\n")

hedge_words <- "allegedly|reportedly|apparently|purportedly|suggests|seems|maybe|perhaps|possibly"

df$n_hedge <- stri_count_regex(tolower(df$text), hedge_words)
df$hedge_ratio <- safe_div(df$n_hedge, df$n_word)

# --- 9. Feature Selection & Export ---

cols_to_keep <- c(
  "id", "is_fake", 
  # Original Metadata
  "n_word", "n_char", "n_url", "n_num",
  # Stylistic (Title)
  "title_cap_ratio",
  # Stylistic (Text)
  "exclam_ratio", "quest_ratio", "text_cap_ratio",
  # Lexical
  "avg_word_len", "lexical_diversity",
  # Subjectivity
  "pronoun_ratio",
  # Sentiment & Emotion
  "sentiment_score", "sentiment_magnitude", 
  "ratio_anger", "ratio_fear", "ratio_disgust", "ratio_joy",
  # Consistency & Uncertainty
  "title_text_overlap", "title_text_len_ratio",
  "hedge_ratio"
)

df_features <- df[, cols_to_keep]

write.csv(df_features, "news_features_numeric.csv", row.names = FALSE)

cat("------------------------------------------------------\n")
cat("Success → news_features_numeric.csv\n")
cat("Total numeric features generated:", (ncol(df_features)-2), "\n")
cat("------------------------------------------------------\n")