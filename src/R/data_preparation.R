################################################################################
# DATA PREPARATION PIPELINE
# 1. Ingestion & Merging
# 2. Quality Filtering (NA removal, Length checks)
# 3. Standardization (Encoding, Date parsing)
# 4. Deduplication
# 5. Text Cleaning & Masking
# 6. Feature Engineering (BERT/TF-IDF formats)
################################################################################

library(here)

# --- 1. Dataset Loading & Labeling ---

fake <- read.csv("data/raw/fake.csv",
                 header = TRUE,
                 fileEncoding = "UTF-8",
                 na.strings = c("", "NA", "N/A", " ", ".", "?"))

true <- read.csv("data/raw/true.csv",
                 header = TRUE,
                 fileEncoding = "UTF-8",
                 na.strings = c("", "NA", "N/A", " ", ".", "?"))

fake$is_fake <- 1
true$is_fake <- 0
df <- rbind(fake, true)
write.csv(df, "data/raw/news.csv", row.names = FALSE, na="")

# --- 2. Quality Control & Filtering ---

for (var in c("title", "text", "subject", "date")) {
  cat("\nColumna:", var, "\n")
  print(any(is.na(df[[var]]))) 
  print(sum(is.na(df[[var]]))) 
}

df <- na.omit(df)

df$text_length  <- nchar(df$text)
df$title_length <- nchar(df$title)

lower_than <- function(long, x) {
  long < x
}

for (var in c("title_length", "text_length")) {
  cat("\nColumna:", var, "\n")
  if (var == "title_length"){ umbral <- 20 }else { umbral <- 140}
  print(any(lower_than(df[[var]], umbral))) 
  print(sum(lower_than(df[[var]], umbral)))   
}

cond <- df$text_length < 140 | df$title_length < 20
any(cond)
sum(cond)
df <- df[!cond, ]

# --- 3. Text Encoding Normalization ---

if (!require("stringi")) install.packages("stringi")
library(stringi)

char_cols <- names(df)[vapply(df, is.character, logical(1))]
for (col in char_cols) {
  x <- df[[col]]
  x <- stri_enc_toutf8(x)
  x <- stri_replace_all_regex(x, "^(?:\\x{FEFF})", "") 
  x <- stri_replace_all_regex(x, "\\p{Cf}", "")        
  df[[col]] <- x
}

if (!"id" %in% names(df) || anyDuplicated(df$id)) {
  df$id <- seq_len(nrow(df))
} else {
  df$id <- as.integer(df$id)
}

# --- 4. Date Parsing & Sorting ---

expand_2digit_year <- function(y, pivot = 30, low_century = 1900, high_century = 2000) {
  if (is.na(y)) return(NA_integer_)
  if (y < 100) {
    if (y <= pivot) return(high_century + y)
    return(low_century + y)
  }
  y
}

normalize_date_simple <- function(v) {
  x <- tolower(trimws(v))
  x <- gsub(",", "", x)
  x <- gsub("([0-9])(st|nd|rd|th)\\b", "\\1", x, perl = TRUE)
  x <- gsub("\\.", "", x)
  
  month_map <- c(
    "january"="01","february"="02","march"="03","april"="04","may"="05","june"="06",
    "july"="07","august"="08","september"="09","october"="10","november"="11","december"="12",
    "jan"="01","feb"="02","mar"="03","apr"="04","may"="05","jun"="06",
    "jul"="07","aug"="08","sep"="09","sept"="09","oct"="10","nov"="11","dec"="12"
  )
  for (m in names(month_map)) x <- gsub(paste0("\\b", m, "\\b"), month_map[m], x)
  
  x <- gsub("[/\\-]", " ", x)
  x <- gsub("\\s+", " ", x)
  
  toks <- strsplit(x, " ", fixed = TRUE)
  
  parse_one <- function(tt) {
    nums <- suppressWarnings(as.integer(tt[grepl("^\\d+$", tt)]))
    if (length(nums) < 3) return(NA_character_)
    if (length(nums) > 3) nums <- nums[1:3]
    
    yr_i <- which(nchar(nums) == 4 & nums >= 1800 & nums <= 2100)
    if (length(yr_i) == 0) yr_i <- which.max(nums)
    year <- expand_2digit_year(nums[yr_i], pivot = 30) 
    rest <- nums[-yr_i]
    if (length(rest) < 2) return(NA_character_)
    
    if (rest[1] >= 1 && rest[1] <= 12 && !(rest[2] >= 1 && rest[2] <= 12)) {
      month <- rest[1]; day <- rest[2]
    } else if (rest[2] >= 1 && rest[2] <= 12 && !(rest[1] >= 1 && rest[1] <= 12)) {
      month <- rest[2]; day <- rest[1]
    } else {
      month <- rest[1]; day <- rest[2]
    }
    
    sprintf("%04d-%02d-%02d", year, month, day)
  }
  
  as.Date(vapply(toks, parse_one, character(1)))
}

df$date <- normalize_date_simple(df$date)
df <- df[order(df$date), ]

# --- 5. Deduplication Strategy ---

# Remove exact matches (Title + Body)
df$.__combo__ <- paste0(trimws(df$title), "|||", trimws(df$text))
keep <- !duplicated(df$.__combo__) 
dup_n <- sum(!keep)
if (dup_n > 0) message("Exact duplicates removed: ", dup_n)
df <- df[keep, ]
df$.__combo__ <- NULL

# Remove duplicate titles (keep longest text)
df$.__key__  <- tolower(trimws(df$title))
df$.__tlen__ <- nchar(df$text)

o <- order(df$.__key__, -df$.__tlen__, df$date)
df <- df[o, ]

keep <- !duplicated(df$.__key__)
dropped_title <- sum(!keep)
df <- df[keep, ]

df$.__key__ <- df$.__tlen__ <- NULL
df <- df[order(df$date), ]

cat("Removed via duplicate titles:", dropped_title, "\n")

# --- 6. Text Cleaning & Masking ---

clean_text_base <- function(x, replace_numbers = TRUE) {
  x <- as.character(x)
  x <- trimws(x)
  
  # HTML Entities to ASCII
  from <- c("&amp;","&lt;","&gt;","&quot;","&#39;","&apos;","&nbsp;")
  to   <- c("&","<",">","\"",   "'",    "'",      " ")
  for (i in seq_along(from)) x <- gsub(from[i], to[i], x, fixed = TRUE)
  
  # Normalize quotes and dashes
  x <- gsub("[\u2018\u2019]", "'", x)
  x <- gsub("[\u201C\u201D]", "\"", x)
  x <- gsub("[\u2013\u2014]", "-", x)
  x <- gsub("\u00A0", " ", x)
  
  # Masking (URLs, Emails, Handles)
  x <- gsub("(https?://|www\\.)\\S+", "__URL__", x, perl = TRUE)
  x <- gsub("\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}\\b", "__EMAIL__", x, perl = TRUE)
  x <- gsub("@[A-Za-z0-9_]{2,}", "__USER__", x, perl = TRUE)
  
  # Remove agency boilerplate
  x <- gsub("^\\s*\\(?(reuters|ap|associated press)\\)?\\s*[-—:]\\s*", "", x,
            ignore.case = TRUE, perl = TRUE)
  x <- gsub("(?i)\\badvertisement\\b", "", x, perl = TRUE)
  x <- gsub("(?i)all rights reserved\\.?","", x, perl = TRUE)
  x <- gsub("(?i)^photo:\\s*", "", x, perl = TRUE)
  
  # Optional number masking
  if (isTRUE(replace_numbers)) {
    x <- gsub("\\d+", "__NUM__", x, perl = TRUE)
  }
  
  x <- gsub("\\s+", " ", x)
  x <- gsub("(^\\s+|\\s+$)", "", x)
  x
}

df$title_raw <- df$title
df$text_raw  <- df$text

df$title <- clean_text_base(df$title, replace_numbers = TRUE)
df$text  <- clean_text_base(df$text,  replace_numbers = TRUE)

df$full_text <- paste0(df$title, ". ", df$text)

# --- 7. Final Feature Preparation & Export ---

# Secondary length filter
min_title_chars <- 20
min_text_chars  <- 140
keep <- (nchar(df$title) >= min_title_chars) & (nchar(df$text) >= min_text_chars)
dropped <- sum(!keep)
if (dropped > 0) message("Rows dropped by secondary length filter: ", dropped)
df <- df[keep, ]

# Format 1: BERT (Preserve punctuation/case)
df$text_bert <- df$full_text

# Format 2: TF-IDF (Lowercase, no punctuation except masks)
to_tfidf_plain <- function(x) {
  x <- tolower(x)
  x <- gsub("[^[:alnum:]_]+", " ", x, perl = TRUE)
  x <- gsub("\\s+", " ", x)
  x <- gsub("(^\\s+|\\s+$)", "", x)
  x
}
df$text_tfidf <- to_tfidf_plain(df$full_text)

# Simple Metadata Features
count_token <- function(x, token) lengths(regmatches(x, gregexpr(token, x, fixed = TRUE)))
df$n_char <- nchar(df$full_text)
df$n_word <- lengths(strsplit(trimws(df$full_text), "\\s+"))
df$n_url  <- count_token(df$full_text, "__URL__")
df$n_num  <- count_token(df$full_text, "__NUM__")

# Export
cols_out <- intersect(c(
  "id","date","is_fake","subject","lang",
  "title","text","full_text","text_bert","text_tfidf",
  "n_char","n_word","n_url","n_num","split"
), names(df))

write.csv(df[ , cols_out], "data/processed/news_prepared.csv", row.names = FALSE)
cat("Success → news_prepared.csv\n")