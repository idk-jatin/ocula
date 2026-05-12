import logging
import collections
from pathlib import Path
import pandas as pd
from src.ocula.data.bridge_map import map_label,ID2LABEL

logger = logging.getLogger(__name__)

class DatasetLoader:

    """
    This is a global dataset loader class
    to load all the datasets required for ocula
    """

    def __init__(self,raw_dir:str | Path="data/raw",):
        self.raw_dir = Path(raw_dir)

    def _row(self,text,raw_label,dataset,lang):
        label_int = map_label(dataset,raw_label)
        return {
            "text":str(text).strip(),
            "label_int":label_int,
            "label_str":ID2LABEL[label_int],
            "source":dataset,
            "lang":lang
        }
    
    def _check(self,path):
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}\n")

    def hatexplain(self):
        import json
        import urllib.request
        logger.info("Loading HateXplain Dataset...")
        url = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json"
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
        rows = []
        for _,ex in data.items():
            text = " ".join(ex["post_tokens"])
            votes = [a["label"] for a in ex["annotators"]]
            majority = collections.Counter(votes).most_common(1)[0][0]
            rows.append(self._row(text, majority, "hatexplain", "en"))
        df = pd.DataFrame(rows)
        logger.info(f"HateXplain Dataset: {len(df):,} rows loaded!")
        return df
    
    def davidson(self):
        from datasets import load_dataset

        logger.info("Loading Davidson Dataset...")
        ds = load_dataset("tdavidson/hate_speech_offensive")
        rows = []
        for ex in ds["train"]:
            rows.append(self._row(ex["tweet"],str(ex["class"]),"davidson","en"))
        df = pd.DataFrame(rows)
        logger.info(f"Davidson Dataset: {len(df):,} rows loaded!")
        return df
    
    def hasoc_english(self):
        path = self.raw_dir / "hasoc_2019_english.tsv"
        self._check(path)

        logger.info("Loading Hasoc 2019 English Dataset...")
        df_raw = pd.read_csv(path,sep='\t')
        rows=[]
        for _,row in df_raw.iterrows():
            rows.append(self._row(row["text"],row["task_2"],"hasoc_english","en"))
        df = pd.DataFrame(rows)
        logger.info(f"Hasoc English Dataset: {len(df):,} rows loaded!")
        return df
    
    def hasoc_hindi(self):
        path = self.raw_dir / "hasoc_2019_hindi.tsv"
        self._check(path)

        logger.info("Loading Hasoc 2019 Hindi Dataset...")
        df_raw = pd.read_csv(path,sep='\t')
        rows=[]
        for _,row in df_raw.iterrows():
            rows.append(self._row(row["text"],row["task_2"],"hasoc_hindi","hi"))
        df = pd.DataFrame(rows)
        logger.info(f"Hasoc Hindi Dataset: {len(df):,} rows loaded!")
        return df
    
    def indo_hate(self):
        path = self.raw_dir / "indo_hate.xlsx"
        self._check(path)

        logger.info("Loading IndoHateSpeech Dataset...")
        df_raw = pd.read_excel(path)
        df_raw = df_raw.dropna(subset=["Comment"])
        rows = []
        for _,row in df_raw.iterrows():
            rows.append(self._row(row["Comment"],row['Label'],"indo_hate","hi-Latn"))
        df = pd.DataFrame(rows)
        logger.info(f"IndoHateSpeech Dataset: {len(df):,} rows loaded!")
        return df

    def load_all(self):
        loaders = [self.hatexplain,self.davidson,self.hasoc_english,self.hasoc_hindi,self.indo_hate]
        dfs = []
        for fn in loaders:
            try:
                dfs.append(fn())
            except FileNotFoundError as e:
                logger.warning(f"Skipping: {e}")
            except Exception as e:
                logger.warning(f"Skipping {fn.__name__} | Found Error: {e}")

        combined = pd.concat(dfs,ignore_index=True)
        logger.info(f"Total combined: {len(combined):,} rows loaded!")
        return combined