import re
import logging
logger = logging.getLogger(__name__)

import ftfy
import emoji

class RegexFilters:
    URL_RE = re.compile(r"https?://\S+|www\.\S+")
    MENTION_RE  = re.compile(r"@\w+")
    HASHTAG_RE  = re.compile(r"#(\w+)")
    REPEAT_RE   = re.compile(r"(.)\1{2,}")
    SPACE_RE    = re.compile(r"\s+")

class TextCleaner:
    def __init__(self,lowercase = True):
        self.lowercase = lowercase

    def clean(self,text):
        if not text or not isinstance(text,str):
            return ""
        
        text = ftfy.fix_text(text)
        text = emoji.demojize(text,delimiters=("[","]"))
        text = RegexFilters.URL_RE.sub("[URL]",text)
        text = RegexFilters.MENTION_RE.sub("[USER]",text)
        text = RegexFilters.HASHTAG_RE.sub(r"\1",text)
        if self.lowercase:
            text = text.lower()
        text = RegexFilters.REPEAT_RE.sub(r"\1\1",text)
        text = RegexFilters.SPACE_RE.sub(" ",text).strip()

        return text

    def clean_df(self,df,text_col = "text",min_tokens = 3):
        df = df.copy()
        df[text_col] = df[text_col].apply(self.clean)

        before = len(df)
        df = df[df[text_col].str.split().str.len()>= min_tokens].reset_index(drop=True)
        dropped = before - len(df)
        if dropped > 0:
            logger.info(f"Dropped {dropped:,} rows with < {min_tokens} tokens")
        return df