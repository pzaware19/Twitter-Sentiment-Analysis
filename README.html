Sentiment Analysis of Indian Political Tweets: 2019 General Elections
================
Piyush Zaware

- [INTRODUCTION](#introduction)
- [DATA DESCRIPTION](#data-description)
- [DATA CLEANING AND PREPROCESSING](#data-cleaning-and-preprocessing)
- [WORD FREQUENCY ANALYSIS](#word-frequency-analysis)
- [SENTIMENT TRAJECTORY](#sentiment-trajectory)
- [SENTIMENT WORD CLOUD](#sentiment-word-cloud)
- [PARTYWISE CLASSIFICATION](#partywise-classification)
- [SENTIMENT BY PARTY](#sentiment-by-party)
- [EMOTIONAL BREAKDOWN](#emotional-breakdown)
- [CONCLUSION](#conclusion)
- [HELP LOGS](#help-logs)

## INTRODUCTION

During the 2019 Indian General Elections, I noticed how significantly
Twitter transformed into a political battleground. Political parties and
their supporters increasingly turned to the platform to influence
discourse, shape narratives, and connect with voters. In this project, I
analyze a large collection of tweets posted between January and May
2019, with the aim of understanding how sentiment and emotional tone
varied across different political parties. Using tidy text analysis, I
focus on word frequencies, sentiment patterns, and emotional vocabulary
to explore the linguistic footprint of Indian political campaigning on
social media.

## DATA DESCRIPTION

The dataset I used consists of tweets
[data](https://www.kaggle.com/datasets/yogesh239/twitter-data-about-2019-indian-general-election)
during the election campaign window from January 1 to May 23, 2019.
These tweets were extracted based on keywords, hashtags, and usernames
linked to political parties and leaders. Each entry includes the tweet
content, timestamp, and the Twitter handle of the user who posted it. To
narrow the scope, I filtered the dataset to focus only on tweets from
the official campaign period and removed tweets that were empty or
improperly formatted.

``` r
tweets <- read_csv("IndianElection19TwitterData.csv")       #this dataset was downloaded from Kaggle https://www.kaggle.com/datasets/yogesh239/twitter-data-about-2019-indian-general-election
tweets <- tweets %>%
  rename(text = Tweet, timestamp = Date, user = User) %>%   #renaming some columns for better understanding, columns are tweet, date and user
  mutate(timestamp = as.POSIXct(timestamp, format = "%Y-%m-%d %H:%M:%S"),
         date = as.Date(timestamp)) %>%                        #drops time stamping from the date part and only keeps the dates
  filter(date >= as.Date("2019-01-01") & date <= as.Date("2019-05-23")) %>%
  drop_na(text)     #only select tweets that I have selected in my timeline
                    #dropping blank columns which have no text
```

## DATA CLEANING AND PREPROCESSING

Before diving into the analysis, I cleaned the data to prepare it for
text mining. I tokenized the tweets into individual words, converted
them to lowercase, removed punctuation, and eliminated common stop words
such as “the” or “and.” This preprocessing step allowed me to focus on
politically meaningful words that reveal discourse patterns, rather than
on grammatical fillers.

``` r
data("stop_words")   #tidytext package list of words, conjunctions, verbs and adverbs which are pretty redundunt in textual analysis. I got senser of this from https://stackoverflow.com/questions/72498240/removing-stop-words-from-text-in-r
tweet_words <- tweets %>%
  select(user, date, text) %>%               # I have only selected user, date and text column
  unnest_tokens(word, text) %>%              # convert the text into a format that is one word per row format for analysis
  filter(str_detect(word, "^[a-z']+$")) %>%  
  anti_join(stop_words, by = "word")   #remove out hashtags, numerical words and others using antijoin
```

## WORD FREQUENCY ANALYSIS

To begin, I identified the most common words across all tweets. As
expected, names like “modi,” “rahul,” “bjp,” “vote,” and “congress”
topped the list. These high frequency words confirm the dataset’s strong
political focus and show how central party figures were to the online
discussion. It also reflects the dominance of national issues and
personalities in voters’ minds during the campaign.

To go beyond basic word frequency, I calculated TF-IDF scores to find
unique words used by individual users. This technique helps spotlight
the most distinctive words for each handle—whether political parties,
media houses, or individuals. For instance, some users used words like
“chowkidar,” “priyanka,” or “surgicalstrike” far more often than others,
suggesting targeted narratives and ideological leanings. It also gave me
a sense of who was pushing which messages more aggressively.

``` r
top_words <- tweet_words %>%count(word, sort = TRUE) %>%top_n(17) %>%
  mutate(party_color = case_when(word %in% c("bjp", "narendramodi", "modi") ~ "BJP",word %in% c("rahulgandhi", "incindia", "congress") ~ "INC",
TRUE ~ "Other"))                                          # words related to bjp and words related inc are grouped together for better visualisation purpose
                                                          # frequency of words appearing in the word df that i had created initially, I select top 17 words
ggplot(top_words, aes(x = reorder(word, n), y = n, fill = party_color)) +
  geom_col() +coord_flip() +scale_fill_manual(values = c("BJP" = "orange", "INC" = "lightblue", "Other" = "black")) +
  labs(title = "Most Common Words in Political Tweets (2019)",x = NULL, y = "Frequency", fill = "Party"
  ) +
  theme_grey()        #plot having all BJP related things in orange and INC related things in blue after comments from instructor meeting
```

![](README_files/figure-gfm/frequently%20used%20words-1.png)<!-- -->

Clearly from the above histogram plot it is inferrable that the word BJP
which corresponds to the ruling party in India and the party that won a
magnanimous mandate in 2019 was the most used word on twitter(now X)
which is not a surprise given how they had shaped their entire campaign
around social media tactics. Narendra Modi the then PM candidate comes
second followed by the present leader opposition and the then Indian
National Congress President Mr Rahul Gandhi. Out of the most requently
used words one word that stands out to me was ‘ji’, while ji is a
respectful address that is mostly used as a sufix to convey respect
after your name e.g Rahul ji, Modi ji and hence this word although not
so important appears in the list. Ideally this should have also been one
of the stopwords but as it happens to be in Hindi this was not included
in the tinytext package

``` r
user_tf_idf <- tweet_words %>%  count(user, word) %>%  bind_tf_idf(word, user, n) %>%  arrange(desc(tf_idf))    #numer of times a particular word was used by a user
top_tf_idf <- user_tf_idf %>%  group_by(user) %>%  slice_max(tf_idf, n = 1) %>%
  ungroup()               #term frequency score for each word for each user, high tf-df means the word is frequent for particular user but rare for others
head(top_tf_idf, 17)
```

    ## # A tibble: 17 × 6
    ##    user            word                       n     tf   idf tf_idf
    ##    <chr>           <chr>                  <int>  <dbl> <dbl>  <dbl>
    ##  1 0007_CJ         cows                       3 0.111   7.07  0.786
    ##  2 001_chandan     chandnichowk               1 0.05    8.50  0.425
    ##  3 001amitsingh    sudhanshubjp               2 0.1    10.8   1.08 
    ##  4 001ankitG       abhinandan                 1 0.5     4.71  2.36 
    ##  5 002_akash       answerable                 1 0.0909  7.41  0.673
    ##  6 004Pruth        pass                       2 0.1     5.48  0.548
    ##  7 0062a04e1ceb458 scale                      1 0.143   6.24  0.892
    ##  8 007Bhas         aamaadmi                   1 0.0455  8.73  0.397
    ##  9 007Fahadkhan    electioncommission         1 0.333   3.45  1.15 
    ## 10 007__AK         opposit                    1 0.0714 10.8   0.772
    ## 11 007_hemal       trusted                    2 0.0625  6.63  0.415
    ## 12 007_joshh       supports                   2 0.0345  5.44  0.188
    ## 13 007_vasu        unitedpakistanalliance     1 0.0208 10.8   0.225
    ## 14 007dibs         depends                    1 0.0909  6.43  0.584
    ## 15 007shekharsuman jaipur                     1 0.0588  6.59  0.388
    ## 16 007vidyadhar    sakhshimaharaj             1 0.143  10.8   1.54 
    ## 17 0099Swami       prestigious                1 0.0625  7.20  0.450

These are random twitter users,not someone who I think are famous or
have a celebrity status. Fir 001amitsingh, word sudhanshubjp has a high
tf-idf score which means its his personal favourite word, as sudhansubjp
happens to be a BJP party member and maybe 001amitsingh has always
tagged him in his tweets while others usually dont. Similarly, 001ankitG
has used abhinandan just once but its uncommon among other users hence
the 2.35 tf-idf score. Words like cows, aamadmi are policy releated
themes and show how certain users focus on specific policy issues or
political narrative building.

## SENTIMENT TRAJECTORY

Using the Bing sentiment lexicon, I calculated daily sentiment scores by
subtracting the number of negative words from positive ones. When I
plotted these over time, I noticed spikes in both directions around mid
April and early May, which align with key phases of the election
campaign. While there were brief sentiment surges, the overall tone
remained relatively balanced across the period. This suggested to me
that praise and criticism were flowing in almost equal measure during
the campaign, possibly driven by debates, manifesto releases, and
campaign rallies.

``` r
sentiment_words <- tweet_words %>% inner_join(get_sentiments("bing"))      #using bing for sentiment analysis https://www.tidytextmining.com/sentiment, using inner join (referenced from penultimate class lecture or I was getting this wrong)
sentiment_by_day <- sentiment_words %>% count(date, sentiment) %>% pivot_wider(names_from = sentiment, values_from = n, values_fill = 0) %>% mutate(net_sentiment = positive - negative)                                    # counting the no of positve and negative words for each date. Formating wife data for each data to have its own positive and negative, fill any missing values with zero such that it will still appear. Add a new sentiment column: sentiment positive if move positive words, sentiment negative if more negative words
ggplot(sentiment_by_day, aes(x = date, y = net_sentiment)) + 
  geom_vline(xintercept = as.Date("2019-02-14"), linetype = "dashed", color = "red") + annotate("text", x = as.Date("2019-02-14"), y = 100, label = "Pulwama Attack", angle = 90, vjust = -0.6, size = 3) + geom_vline(xintercept = as.Date("2019-02-26"), linetype = "dashed", color = "red") + annotate("text", x = as.Date("2019-02-26"), y = 100, label = "Balakot Strikes", angle = 90, vjust = -0.6, size = 3) + geom_vline(xintercept = as.Date("2019-03-10"), linetype = "dashed", color = "red") + annotate("text", x = as.Date("2019-03-10"), y = 100, label = "Onset of elections", angle = 90, vjust = -0.6, size = 3) + geom_vline(xintercept = as.Date("2019-05-19"), linetype = "dashed", color = "red") + annotate("text", x = as.Date("2019-05-19"), y = 100, label = "End of polling", angle = 90, vjust = -0.6, size = 3) +
  geom_line(color = "black") +                                             #these lines are added just to mark important dates with context to indian elections
  labs(title = "Daily Net Sentiment of Political Tweets",
       x = "Date", y = "Net Sentiment") +
  theme_grey()
```

![](README_files/figure-gfm/sentiment-1.png)<!-- -->

Trends in the sentiment show that during January it starts out fairly
negative, indicating anti incumbancy and backlash from the late-2018
political events in India and also opposition winning 3 state assembly
elections during december 2018. By mid February theres a sharp dip
possibly due to the Pulwama Terrorist Attack that happened on 14th
February which triggered anger, grief and sorrow. Late February and
early march shows a spike possibly linked to Balakot airstrike and the
government’s response which generated praise and leadership
upheavel.March to April was the build up to elections and hence the
negative or neutral spike is explained with party campaigns, manifestos
and rising levels of polarisaiton. April to May was the election phase
where there are several negative sentiments due to criticism of parties
and leaders from both the sides.

## SENTIMENT WORD CLOUD

To visualize sentiment rich words, I generated a word cloud using terms
matched to either positive or negative sentiment. Words like
“development,” “trust,” and “victory” stood out on the positive side,
while “lies,” “corrupt,” and “scam” featured among the negative. This
mix of aspirational and accusatory language felt typical of high stakes
electoral contests, especially in India’s charged political climate.

``` r
wordcloud_data <- sentiment_words %>% count(word, sentiment, sort = TRUE)              #this counts the number of words in sentiment_words dataset grouping them by sentiments and then sorting it 
wordcloud(words = wordcloud_data$word,   #this code just creates the word cloud, the last line ensures that biggest words appear at the center than being scatter around
 freq = wordcloud_data$n,  min.freq = 30, max.words = 100, colors = c("black", "grey"),random.order = FALSE)
```

![](README_files/figure-gfm/sentiment%20word%20cloud-1.png)<!-- -->

Words like support, win, shame, corruption, proud, lost appear bigger
possibly as these are the most commonly used emotionally charged up
words by political parties and leaders during election. We can see that
highly polarized words like support, win, proud, respect, love, shame,
corruption, defeat, scam, terrorism comprise the wordcloud suggesting it
was highly polarized election seasion.

``` r
nrc <- get_sentiments("nrc")
bing <- get_sentiments("bing")
```

## PARTYWISE CLASSIFICATION

To better compare patterns, I assigned tweets to major political parties
based on keyword mentions (like “Modi” for BJP or “Rahul Gandhi” for
INC). This classification enabled me to break down sentiment and
emotional tones by party. I filtered out tweets without clear party
references to maintain clarity in comparison.

``` r
party_patterns <- list(
  BJP   = c("@BJP4India", "\\bBJP\\b", "Modi", "Narendra Modi"),
  INC   = c("@INCIndia", "\\bCongress\\b", "Rahul Gandhi", "@RahulGandhi"),
  AAP   = c("@AamAadmiParty", "\\bAAP\\b", "Arvind Kejriwal"),
  TMC   = c("@AITCofficial", "\\bTMC\\b", "Mamata Banerjee"),
  CPIM  = c("@cpimspeak", "@cpofindia", "\\bCPI\\b", "\\bCPIM\\b"),
  SP    = c("@samajwadiparty", "Akhilesh Yadav", "\\bSP\\b")
)  #these are some party specific keywords or twitter handles that I thought would be relevant given my knowledge of Indian Politics

detect_party <- function(tweet) { for (party in names(party_patterns)) { if (any(str_detect(tweet, regex(party_patterns[[party]], ignore_case = TRUE)))) { return(party) } }
  return("Other")
}
                #this is a party helper function that goes and loops through each tweet and checks whether any of the party specific patterns that I mentioned above show in the then

tweets <- tweets %>% mutate(party = map_chr(text, detect_party)) %>%filter(party != "Other")   #function to each tweet and add a new parrty column, filter out tweets that have no parties mentioned especially the ones I have mentioned in the code above

tidy_tweets <- tweets %>%
  unnest_tokens(word, text) %>%                     #break tweet into indiv words and then remove filler words and numbers, hashtags and punctuations
  anti_join(stop_words, by = "word") %>%
  filter(str_detect(word, "^[a-z]+$"))              #filter out the hashtags, numerics which are not used in our study or analysis
```

## SENTIMENT BY PARTY

The party wise sentiment analysis showed interesting contrasts. Tweets
about the BJP contained slightly more positive sentiment compared to
other, reflecting successful outreach or high engagement from
supporters. On the other hand, tweets about the Congress and AAP parties
had a more critical tone, with higher levels of negative sentiment. I
interpreted this as evidence of an opposition driven strategy that
emphasized critique rather than celebration.

``` r
sentiment_bing <- tidy_tweets %>%
  select(word, party) %>%          
  inner_join(bing, by = "word") %>%
  count(party, sentiment, sort = TRUE)        #joining tidytweets dataset with bing sentient lexicon, labelling words as positive or negative 
ggplot(sentiment_bing, aes(x = reorder(party, -n), y = n, fill = sentiment)) + geom_col(position = "dodge") +
 labs(title = "Sentiment Word Usage by Party", x = "Party", y = "Word Count", fill = "Sentiment") +
  theme_grey()      #https://ggplot2.tidyverse.org/reference/ggtheme.html
```

![](README_files/figure-gfm/sentiment%20analysis%20using%20bing-1.png)<!-- -->

## EMOTIONAL BREAKDOWN

Using the NRC lexicon, I categorized emotion related words (e.g., joy,
anger, fear, trust). BJP related tweets leaned more heavily into
emotions like “joy” and “trust,” while INC and TMC were more frequently
associated with “anger” and “fear.” This variation may reflect differing
campaign tactics—BJP projecting confidence and momentum, while the
opposition appealed to dissatisfaction and a sense of crisis.

``` r
emotion_df <- tidy_tweets %>% inner_join(nrc, by = "word") %>% count(party, sentiment) %>% filter(sentiment %in% c("joy", "anger", "fear", "trust", "sadness", "disgust"))                                  #similar to the bing sentiment I create the emotion_df dataframe using tidytweetsa and inner joining by words but using the nrc now https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
ggplot(emotion_df, aes(x = reorder(party, -n), y = n, fill = sentiment)) +
  geom_col(position = "stack") +                                                    #here i have used stack because I wanted the emotions to be stacked upon one another to understand and visualise each sentiment party wise better
  labs(title = "Emotion Distribution Across Parties (NRC Lexicon)",
       x = "Party", y = "Word Count", fill = "Emotion") + scale_y_continuous(labels = comma) +
  theme_grey()                  #https://ggplot2.tidyverse.org/reference/ggtheme.html
```

![](README_files/figure-gfm/emotion%20breakdwon%20using%20nrc-1.png)<!-- -->

## CONCLUSION

Through this text analysis, I gained valuable insights into how
different political parties shaped public discourse during the 2019
Indian General Elections. Although all parties used both positive and
negative rhetoric, the tone, vocabulary, and emotional triggers varied.
BJP appeared to lead in projecting positive sentiment and trust,while
also their sample size in the twitter data was huge as compared to the
other political parties suggesting widespread use of social media
campaigning during the 2019 election campaign. While Congress and AAP
emphasized critique and policy issues. These patterns reflect both top
down campaign messaging and bottom up voter engagement. If I had more
time, I would compare this dataset to offline speech transcripts or
newspaper headlines to test whether the online discourse matched real
world narratives.

## HELP LOGS

AI Help: I tried to get help from AI regarding generating a different
shape of my word cloud but that did not help my case and I thought
continuing with the basic shape would be the best In order to complete
this project I was stuck with the problem of having way too many filler
words and while attending the penultimate lecture I figureds stopwords
out.  
I tried using AI to help me with generating a different shape of the
word cloud but I didnt implement it in this project. Kaggle was of great
help as was stackoverflow. <br>

1.  <https://www.kaggle.com/datasets/yogesh239/twitter-data-about-2019-indian-general-election>
    <br>
2.  <https://stackoverflow.com/questions/72498240/removing-stop-words-from-text-in-r>
    <br>
3.  <https://www.tidytextmining.com/sentiment> <br>
4.  <https://chatgpt.com/share/68782ef1-f428-8009-9eee-77ba34907e53>
    <br>
5.  <https://ggplot2.tidyverse.org/reference/ggtheme.html> <br>
6.  <https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm> <br>
7.  <https://en.wikipedia.org/wiki/2019_Balakot_airstrike> <br>
8.  <https://en.wikipedia.org/wiki/2019_Pulwama_attack> <br>
