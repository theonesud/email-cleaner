import os
import pickle
import re

import matplotlib.pyplot as plt
import pandas as pd
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from thefuzz import fuzz

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


class GmailAnalyzer:
    def __init__(self):
        self.creds = None
        self.service = None

    def authenticate(self):
        """Handles Gmail authentication"""
        if os.path.exists("token.pickle"):
            with open("token.pickle", "rb") as token:
                self.creds = pickle.load(token)

        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "client_secret.json", SCOPES
                )
                self.creds = flow.run_local_server(port=0)

            with open("token.pickle", "wb") as token:
                pickle.dump(self.creds, token)

        self.service = build("gmail", "v1", credentials=self.creds)

    def get_emails(self, max_results=None):
        """Fetches emails from Gmail using batch requests of 100 emails each"""
        messages = []
        next_page_token = None

        # Fetch all message IDs using pagination
        while True:
            results = (
                self.service.users()
                .messages()
                .list(userId="me", maxResults=100, pageToken=next_page_token)
                .execute()
            )
            messages.extend(results.get("messages", []))
            next_page_token = results.get("nextPageToken")

            if not next_page_token:
                break

            print(f"{len(messages)} emails fetched...")

        print(f"Total messages found: {len(messages)}")
        emails_data = []

        # Process messages in batches of 100
        for i in range(0, len(messages), 100):
            batch_messages = messages[i : i + 100]
            batch = self.service.new_batch_http_request()

            def callback(request_id, response, exception):
                if exception is None:
                    headers = response["payload"]["headers"]
                    email_data = {
                        "from": next(
                            (h["value"] for h in headers if h["name"] == "From"), ""
                        ),
                        "date": next(
                            (h["value"] for h in headers if h["name"] == "Date"), ""
                        ),
                        "subject": next(
                            (h["value"] for h in headers if h["name"] == "Subject"), ""
                        ),
                    }
                    emails_data.append(email_data)

            # Add requests for this batch
            for message in batch_messages:
                batch.add(
                    self.service.users()
                    .messages()
                    .get(
                        userId="me",
                        id=message["id"],
                        format="metadata",
                        metadataHeaders=["From", "Date", "Subject"],
                    ),
                    callback=callback,
                )

            print(
                f"Fetching emails batch {(i//100) + 1} of {(len(messages) + 99)//100}..."
            )
            batch.execute()

        return emails_data

    def analyze_emails(self, emails_data):
        """Analyzes email patterns using fuzzy matching for sender domains"""
        df = pd.DataFrame(emails_data)

        def normalize_sender(email_str):
            """Keep the full email address instead of just domain"""
            # Extract email from the "From" field
            email_match = re.search(r"<(.+?)>", email_str)
            return email_match.group(1) if email_match else email_str

        def clean_date(date_str):
            """Clean date string by removing timezone information in parentheses"""
            # Remove anything in parentheses
            date_str = re.sub(r"\([^)]*\)", "", date_str)
            # Remove timezone abbreviations like IST, GMT, etc.
            date_str = re.sub(r"\s+[A-Z]{2,4}(?:\s+|$)", " ", date_str)
            return date_str.strip()

        df["normalized_sender"] = df["from"].apply(normalize_sender)

        # Clean and parse dates
        df["date"] = pd.to_datetime(
            df["date"].apply(clean_date),
            format="mixed",
            utc=True,
            errors="coerce",
        )

        # Drop rows with invalid dates
        df = df.dropna(subset=["date"])

        # Group similar senders using fuzzy matching with lower threshold
        unique_senders = df["normalized_sender"].unique()
        sender_groups = {}

        for sender in unique_senders:
            matched = False
            for existing_group in sender_groups:
                # Reduced threshold from 80 to 65 for more lenient matching
                if fuzz.ratio(sender.lower(), existing_group.lower()) > 65:
                    sender_groups[existing_group].append(sender)
                    matched = True
                    break
            if not matched:
                sender_groups[sender] = [sender]

        # Map each sender to its group
        def get_sender_group(sender):
            for group, members in sender_groups.items():
                if sender in members:
                    return group
            return sender

        df["sender_group"] = df["normalized_sender"].apply(get_sender_group)

        # Calculate statistics by sender group
        sender_stats = (
            df.groupby("sender_group")
            .agg({"date": ["count", "min", "max"]})
            .reset_index()
        )

        sender_stats.columns = ["sender", "total_emails", "first_email", "last_email"]

        # Calculate frequency (emails per day)
        sender_stats["days_span"] = (
            sender_stats["last_email"] - sender_stats["first_email"]
        ).apply(lambda x: x.total_seconds() / 86400)

        # Handle cases where emails were sent on the same day
        sender_stats["days_span"] = sender_stats["days_span"].apply(lambda x: max(x, 1))

        sender_stats["emails_per_day"] = (
            sender_stats["total_emails"] / sender_stats["days_span"]
        )

        # Fix infinite values
        sender_stats.loc[
            sender_stats["emails_per_day"].isin([float("inf"), float("-inf")]),
            "emails_per_day",
        ] = sender_stats[
            "total_emails"
        ]  # If all emails were on same day, use total as daily rate

        return sender_stats.sort_values("total_emails", ascending=False)

    def plot_top_senders(self, sender_stats, top_n=20):
        """Creates visualizations for email analysis"""
        # Plot top senders by volume
        plt.figure(figsize=(20, 10))  # Increased figure size for longer labels
        top_senders = sender_stats.nlargest(top_n, "total_emails")
        bars = plt.bar(range(len(top_senders)), top_senders["total_emails"])
        plt.xticks(
            range(len(top_senders)),
            top_senders["sender"],
            rotation=45,
            ha="right",
            fontsize=8,  # Smaller font size for longer labels
        )

        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
            )

        plt.title(f"Top {top_n} Email Senders by Volume")
        plt.xlabel("Sender")
        plt.ylabel("Number of Emails")
        plt.tight_layout()
        plt.savefig("top_senders.png", dpi=300)  # Higher DPI for better resolution
        plt.close()

        try:
            # Plot top senders by frequency (per day)
            plt.figure(figsize=(20, 10))  # Increased figure size
            top_frequent = sender_stats.nlargest(top_n, "emails_per_day")

            # Ensure we have finite values
            top_frequent = top_frequent[
                top_frequent["emails_per_day"].notna()
                & top_frequent["emails_per_day"].abs().lt(float("inf"))
            ]

            if len(top_frequent) > 0:
                bars = plt.bar(range(len(top_frequent)), top_frequent["emails_per_day"])
                plt.xticks(
                    range(len(top_frequent)),
                    top_frequent["sender"],
                    rotation=45,
                    ha="right",
                    fontsize=8,  # Smaller font size for longer labels
                )

                # Add value labels on top of each bar
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.2f}",
                        ha="center",
                        va="bottom",
                    )

                plt.title(
                    f"Top {len(top_frequent)} Email Senders by Frequency (emails per day)"
                )
                plt.xlabel("Sender")
                plt.ylabel("Emails per Day")
                plt.tight_layout()
                plt.savefig("top_frequent_senders.png", dpi=300)  # Higher DPI
            else:
                print("Warning: No valid frequency data to plot")
        except Exception as e:
            print(f"Error plotting frequency chart: {e}")
        finally:
            plt.close()


def main():
    if os.path.exists("all_emails_raw.csv"):
        print("Found existing email data, loading from all_emails_raw.csv...")
        df_raw = pd.read_csv("all_emails_raw.csv")
        emails_data = df_raw.to_dict("records")
    else:
        analyzer = GmailAnalyzer()
        analyzer.authenticate()

        print("Fetching all emails...")
        emails_data = analyzer.get_emails()

        # Save raw email data to CSV
        print("Saving raw email data...")
        df_raw = pd.DataFrame(emails_data)
        df_raw.to_csv("all_emails_raw.csv", index=False)

    print("Analyzing emails...")
    analyzer = GmailAnalyzer()  # Create analyzer instance if we loaded from CSV
    sender_stats = analyzer.analyze_emails(emails_data)

    # Save analyzed sender stats to CSV
    print("Saving sender statistics...")
    sender_stats.to_csv("sender_statistics.csv", index=False)

    print("\nTop 10 senders by total emails:")
    print(
        sender_stats.nlargest(10, "total_emails")[
            ["sender", "total_emails", "emails_per_day"]
        ]
    )

    print("\nGenerating visualizations...")
    analyzer.plot_top_senders(sender_stats)
    print("Visualizations saved as 'top_senders.png' and 'top_frequent_senders.png'")
    print("\nData saved to 'all_emails_raw.csv' and 'sender_statistics.csv'")


if __name__ == "__main__":
    main()
