import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os

logger = logging.getLogger(__name__)

def plot_histograms(df, columns=None, bins=30, output_dir="plots"):
    logger.info("Plotting histograms for columns=%s with bins=%d", columns, bins)
    os.makedirs(output_dir, exist_ok=True)

    if columns is None:
        columns = df.select_dtypes(include=["number"]).columns

    for col in columns:
        logger.info("Plotting histogram for '%s'", col)
        plt.figure()
        df[col].hist(bins=bins)
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")

        # Save to file
        filename = os.path.join(output_dir, f"hist_{col}.png")
        plt.savefig(filename)
        logger.info("Saved histogram of '%s' to %s", col, filename)

        plt.close()  # Close figure to free memory

def plot_correlations(corr_matrix, output_dir="plots"):
    logger.info("Plotting correlation heatmap...")
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix Heatmap")

    filename = os.path.join(output_dir, "correlation_matrix.png")
    plt.savefig(filename)
    logger.info("Saved correlation matrix heatmap to %s", filename)

    plt.close()

def plot_timeseries(df, date_col="date", columns=None, output_dir="plots"):
    logger.info("Plotting time series for columns=%s", columns)
    os.makedirs(output_dir, exist_ok=True)

    if columns is None:
        columns = [c for c in df.columns if c != date_col]

    for col in columns:
        logger.info("Plotting time series for '%s'", col)
        plt.figure()
        plt.plot(df[date_col], df[col])
        plt.title(f"Time Series for {col}")
        plt.xlabel("Date")
        plt.ylabel(col)
        plt.xticks(rotation=45)
        plt.tight_layout()

        filename = os.path.join(output_dir, f"time_series_{col}.png")
        plt.savefig(filename)
        logger.info("Saved time series of '%s' to %s", col, filename)

        plt.close()
