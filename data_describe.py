import pandas as pd
import logging

logger = logging.getLogger(__name__) 

def load_data(csv_path):
    logger.info("Loading data from '%s'", csv_path)
    # If 'date' column doesn't exist, remove 'parse_dates' or update it
    df = pd.read_csv(csv_path, parse_dates=["date"], delimiter=";") 
    logger.info("Data loaded. Shape: %s", df.shape)
    return df

def summarize_data(df):
    logger.info("Summarizing data...")
    # Basic info
    logger.info("DataFrame info:\n%s", df.info(buf=None))  # .info() prints automatically
    # Basic stats
    desc = df.describe()
    logger.info("Statistical summary:\n%s", desc)

def clean_data(df):
    logger.info("Cleaning data...")
    # Example: drop rows with nulls
    original_shape = df.shape
    df = df.dropna()
    df = df.drop_duplicates()
    logger.info("Dropped missing/duplicate rows. Original shape=%s, New shape=%s",
                original_shape, df.shape)
    return df

def feature_target_split(df, target_column="BC"):
    logger.info("Splitting DataFrame into features and target='%s'", target_column)
    X = df.drop(columns=["date", target_column], errors="ignore")
    y = df[target_column]
    logger.info("Feature set shape=%s, Target shape=%s", X.shape, y.shape)
    return X, y

def correlation_matrix(df):
    logger.info("Computing correlation matrix...")
    corr = df.corr(numeric_only=True)
    logger.info("Correlation matrix computed.")
    return corr
