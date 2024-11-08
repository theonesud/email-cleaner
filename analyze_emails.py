import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import unquote

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from tqdm import tqdm


def load_and_sample_data(sample_size=1000, random_seed=42):
    """Load data and create a stratified sample based on domains"""
    print("Loading full dataset...")
    df = pd.read_csv("all_emails_raw.csv")

    if len(df) == 0:
        print("Warning: Input dataset is empty")
        return df

    # Clean and parse dates
    def clean_date(date_str):
        if pd.isna(date_str):
            return date_str
        # Remove timezone in parentheses
        date_str = re.sub(r"\s*\([^)]*\)", "", str(date_str))
        # Remove timezone abbreviations
        date_str = re.sub(r"\s+[A-Z]{2,4}(?:\s+|$)", " ", date_str)
        # Remove GMT/UTC offsets
        date_str = re.sub(r"\s*GMT[+-]\d{2}:?\d{2}", "", date_str)
        # Clean up any double spaces
        date_str = re.sub(r"\s+", " ", date_str)
        return date_str.strip()

    print("Parsing dates...")
    df["date"] = pd.to_datetime(
        df["date"].apply(clean_date),
        format="mixed",
        utc=True,
        errors="coerce",
    )

    # Drop rows with invalid dates
    original_len = len(df)
    df = df.dropna(subset=["date"])
    if len(df) < original_len:
        print(f"Dropped {original_len - len(df)} rows with invalid dates")

    if len(df) == 0:
        print("Warning: No valid data after date parsing")
        return df

    # Extract domains for stratification
    def extract_domain(email_str):
        email_match = re.search(r"<(.+?)>", email_str)
        email = email_match.group(1) if email_match else email_str
        domain_match = re.search(r"@(.+)$", email)
        return domain_match.group(1).lower() if domain_match else "unknown"

    df["sender_domain"] = df["from"].apply(extract_domain)

    # Calculate sampling fractions for each domain
    domain_counts = df["sender_domain"].value_counts()
    sampling_fractions = (sample_size * domain_counts / len(df)).clip(
        upper=domain_counts
    )

    # Create stratified sample
    sampled_indices = []
    np.random.seed(random_seed)

    for domain, count in sampling_fractions.items():
        domain_indices = df[df["sender_domain"] == domain].index
        sampled_indices.extend(
            np.random.choice(domain_indices, size=int(count), replace=False)
        )

    sample_df = df.loc[sampled_indices].copy()

    # Save sample
    sample_df.to_csv("email_sample.csv", index=False)
    print(
        f"Created sample with {len(sample_df)} emails from {len(sample_df['sender_domain'].unique())} domains"
    )

    return sample_df


def basic_analysis(df):
    """Perform basic analysis on the dataset"""
    print("\n=== Basic Statistics ===")
    print(f"Total emails: {len(df):,}")

    # Handle date range safely
    if len(df) > 0:
        min_date = df["date"].min()
        max_date = df["date"].max()
        if pd.notna(min_date) and pd.notna(max_date):
            print(
                f"Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
            )
        else:
            print("Date range: No valid dates found")
    else:
        print("Date range: No data available")

    print(f"Unique senders: {df['from'].nunique():,}")
    print(f"Unique domains: {df['sender_domain'].nunique():,}")

    if len(df) > 0:
        # Size statistics
        print("\n=== Size Statistics ===")
        size_mb = df["size_estimate"] / (1024 * 1024)
        stats = size_mb.describe()
        print("Email sizes (MB):")
        print(f"Average: {stats['mean']:.2f}")
        print(f"Median: {stats['50%']:.2f}")
        print(f"Max: {stats['max']:.2f}")
        print(f"Total size: {size_mb.sum():.2f} MB")

        # Attachment statistics
        print("\n=== Attachment Statistics ===")
        print(
            f"Emails with attachments: {df['has_attachments'].sum():,} ({df['has_attachments'].mean()*100:.1f}%)"
        )

        if "attachments" in df.columns:
            attachments = df[df["attachments"].notna()]["attachments"].apply(eval)
            if len(attachments) > 0:
                all_attachments = [att for sublist in attachments for att in sublist]
                print(f"Total attachments: {len(all_attachments):,}")
                print("\nTop attachment types:")
                mime_types = (
                    pd.Series([att["mime_type"] for att in all_attachments])
                    .value_counts()
                    .head()
                )
                for mime_type, count in mime_types.items():
                    print(f"{mime_type}: {count:,}")

        # Unsubscribe statistics
        print("\n=== Unsubscribe Links ===")
        print(
            f"Emails with header unsubscribe: {df['list_unsubscribe'].notna().sum():,} ({df['list_unsubscribe'].notna().mean()*100:.1f}%)"
        )
        print(
            f"Emails with body unsubscribe: {df['body_unsubscribe_link'].notna().sum():,} ({df['body_unsubscribe_link'].notna().mean()*100:.1f}%)"
        )


def plot_email_patterns(df):
    """Create visualizations of email patterns"""
    if len(df) == 0:
        print("No data available for plotting")
        return

    # Set style
    sns.set_theme(style="whitegrid")

    # 1. Emails over time
    plt.figure(figsize=(15, 6))
    df["year_month"] = df["date"].dt.to_period("M")
    monthly_counts = df.groupby("year_month").size()

    if len(monthly_counts) > 0:
        ax = monthly_counts.plot(kind="bar")
        plt.title("Email Volume by Month")
        plt.xlabel("Month")
        plt.ylabel("Number of Emails")
        plt.xticks(rotation=45)

        # Add value labels
        for i, v in enumerate(monthly_counts):
            ax.text(i, v, str(v), ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig("email_volume_by_month.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Top domains
    plt.figure(figsize=(15, 8))
    top_domains = df["sender_domain"].value_counts().head(20)
    ax = sns.barplot(x=top_domains.values, y=top_domains.index)
    plt.title("Top 20 Email Domains")
    plt.xlabel("Number of Emails")

    # Add value labels
    for i, v in enumerate(top_domains.values):
        ax.text(v, i, f" {v}", va="center")

    plt.tight_layout()
    plt.savefig("top_domains.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Email sizes distribution
    plt.figure(figsize=(12, 6))
    size_mb = df["size_estimate"] / (1024 * 1024)
    sns.histplot(data=size_mb[size_mb < size_mb.quantile(0.95)], bins=50)
    plt.title("Distribution of Email Sizes (excluding top 5% largest)")
    plt.xlabel("Size (MB)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("email_sizes.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Daily email patterns
    plt.figure(figsize=(12, 6))
    df["hour"] = df["date"].dt.hour
    hourly_counts = df.groupby("hour").size()
    sns.lineplot(x=hourly_counts.index, y=hourly_counts.values)
    plt.title("Email Activity by Hour of Day")
    plt.xlabel("Hour (24-hour format)")
    plt.ylabel("Number of Emails")
    plt.xticks(range(0, 24, 2))
    plt.tight_layout()
    plt.savefig("email_activity_by_hour.png", dpi=300, bbox_inches="tight")
    plt.close()


def test_unsubscribe_link(url):
    """Test a single unsubscribe link and return its status"""
    try:
        # Don't actually click/POST, just check if the link is valid
        response = requests.head(unquote(url), allow_redirects=True, timeout=5)
        return {
            "url": url,
            "status_code": response.status_code,
            "final_url": response.url,
        }
    except Exception as e:
        return {"url": url, "status_code": None, "error": str(e)}


def analyze_unsubscribe_links(df):
    """Analyze all unsubscribe links in the dataset"""
    print("\n=== Analyzing Unsubscribe Links ===")

    # Get all valid unsubscribe links
    links = df["body_unsubscribe_link"].dropna().unique()
    print(f"Found {len(links)} unique unsubscribe links")

    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(test_unsubscribe_link, link) for link in links]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Testing links"
        ):
            results.append(future.result())

    # Analyze results
    success_count = sum(
        1 for r in results if r.get("status_code") in [200, 301, 302, 307, 308]
    )
    print("\nResults:")
    print(f"Working links: {success_count}")
    print(f"Failed links: {len(results) - success_count}")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("unsubscribe_links_analysis.csv", index=False)
    print("\nDetailed results saved to 'unsubscribe_links_analysis.csv'")


def main():
    # Create sample if it doesn't exist
    sample_path = Path("email_sample.csv")
    if not sample_path.exists():
        df = load_and_sample_data()
    else:
        print("Loading existing sample...")
        df = pd.read_csv(sample_path)
        df["date"] = pd.to_datetime(df["date"], format="mixed", utc=True)

    # Perform analysis
    # basic_analysis(df)
    # plot_email_patterns(df)
    analyze_unsubscribe_links(df)

    print("\nAnalysis complete. Check the generated CSV files and plots.")


if __name__ == "__main__":
    main()
