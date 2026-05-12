import sys
import logging
import hashlib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocula.data.bridge_map import ID2LABEL
from src.ocula.data.loaders import DatasetLoader
from src.ocula.data.preprocess import TextCleaner

logging.basicConfig(level=logging.INFO,format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def dedup(df):
    before = len(df)
    df["_hash"] = df["text"].apply(lambda t: hashlib.sha256(t.encode("utf-8",errors="ignore")).hexdigest())
    df = df.drop_duplicates(subset=["_hash"]).drop(columns=["_hash"])
    logger.info(f"Dedup: {before:,} to {len(df):,} removed({before - len(df):,})")
    return df.reset_index(drop=True)

def print_dist(df,title):
    total = len(df)
    logger.info(f"\n{title}")
    logger.info("-" * 40)
    for i, name in ID2LABEL.items():
        count = (df["label_int"] == i).sum()
        logger.info(f"  {name:12s} = {count:6,}  ({count/total*100:.1f}%)")
    logger.info(f"  {'TOTAL':12s} = {total:6,}")

def split(df):
    train_val,test = train_test_split(df,test_size=0.10,stratify=df["label_int"],random_state=42)
    train,val = train_test_split(train_val,test_size=0.111,stratify=train_val["label_int"],random_state=42)
    return(train.reset_index(drop=True),val.reset_index(drop=True),test.reset_index(drop=True))

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Building Unified Corpus...")
    logger.info("=" * 50)

    loader = DatasetLoader()
    df = loader.load_all()
    print_dist(df, "Raw combined")

    logger.info("\nCleaning text ...")
    cleaner = TextCleaner()
    df = cleaner.clean_df(df)
    print_dist(df,"After cleaning")

    df = dedup(df)
    print_dist(df,"After dedup")

    train, val, test = split(df)
    print_dist(train,f"Train ({len(train):,})")
    print_dist(val,f"Val({len(val):,})")
    print_dist(test,f"Test({len(test):,})")

    Path("data/splits").mkdir(parents=True,exist_ok=True)
    train.to_csv("data/splits/train.csv",index=False)
    val.to_csv("data/splits/val.csv",index=False)
    test.to_csv("data/splits/test.csv",index=False)
    logger.info("\nSaved in data/splits/")

    logger.info("\nUnified dataset complete.")