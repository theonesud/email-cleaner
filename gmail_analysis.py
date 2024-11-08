import os
import pickle
import re
import time
from collections import deque

import matplotlib.pyplot as plt
import pandas as pd
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

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
        """Fetches emails from Gmail using batch requests with extended metadata"""
        messages = []
        current_token = None

        print("Fetching message IDs...")

        def fetch_page(page_token, retries=3):
            for attempt in range(retries):
                try:
                    results = (
                        self.service.users()
                        .messages()
                        .list(userId="me", maxResults=500, pageToken=page_token)
                        .execute()
                    )
                    return results.get("messages", []), results.get("nextPageToken")
                except Exception as e:
                    if attempt == retries - 1:
                        print(f"Failed to fetch page after {retries} attempts: {e}")
                        return [], None
                    print(f"Attempt {attempt + 1} failed, retrying... Error: {e}")
                    time.sleep(2**attempt)
                return [], None

        # First request to get initial data
        try:
            initial_results = (
                self.service.users()
                .messages()
                .list(userId="me", maxResults=500)
                .execute()
            )
            messages.extend(initial_results.get("messages", []))
            current_token = initial_results.get("nextPageToken")

            while current_token:
                page_messages, next_token = fetch_page(current_token)
                if page_messages:
                    messages.extend(page_messages)
                    print(f"{len(messages)} message IDs fetched...")
                current_token = next_token
                time.sleep(0.1)

            print(f"Total messages found: {len(messages)}")

        except Exception as e:
            print(f"Fatal error in message ID fetching: {e}")
            raise

        # Process messages in batches
        emails_data = []
        batch_size = 100
        batches = [
            messages[i : i + batch_size] for i in range(0, len(messages), batch_size)
        ]

        def process_batch(batch_messages):
            batch_data = []
            batch = self.service.new_batch_http_request()
            response_queue = deque()

            def callback(request_id, response, exception):
                if exception is None:
                    headers = response["payload"]["headers"]
                    email_data = {
                        "id": response["id"],
                        "thread_id": response.get("threadId", ""),
                        "label_ids": response.get("labelIds", []),
                        "snippet": response.get("snippet", ""),
                        "internal_date": response.get("internalDate", ""),
                        "size_estimate": response.get("sizeEstimate", 0),
                        "from": next(
                            (
                                h["value"]
                                for h in headers
                                if h["name"].lower() == "from"
                            ),
                            "",
                        ),
                        "to": next(
                            (h["value"] for h in headers if h["name"].lower() == "to"),
                            "",
                        ),
                        "cc": next(
                            (h["value"] for h in headers if h["name"].lower() == "cc"),
                            "",
                        ),
                        "bcc": next(
                            (h["value"] for h in headers if h["name"].lower() == "bcc"),
                            "",
                        ),
                        "date": next(
                            (
                                h["value"]
                                for h in headers
                                if h["name"].lower() == "date"
                            ),
                            "",
                        ),
                        "subject": next(
                            (
                                h["value"]
                                for h in headers
                                if h["name"].lower() == "subject"
                            ),
                            "",
                        ),
                        "message_id": next(
                            (
                                h["value"]
                                for h in headers
                                if h["name"].lower() == "message-id"
                            ),
                            "",
                        ),
                        "reply_to": next(
                            (
                                h["value"]
                                for h in headers
                                if h["name"].lower() == "reply-to"
                            ),
                            "",
                        ),
                        "references": next(
                            (
                                h["value"]
                                for h in headers
                                if h["name"].lower() == "references"
                            ),
                            "",
                        ),
                        "list_unsubscribe": next(
                            (
                                h["value"]
                                for h in headers
                                if h["name"].lower() == "list-unsubscribe"
                            ),
                            "",
                        ),
                        "list_unsubscribe_post": next(
                            (
                                h["value"]
                                for h in headers
                                if h["name"].lower() == "list-unsubscribe-post"
                            ),
                            "",
                        ),
                        "importance": next(
                            (
                                h["value"]
                                for h in headers
                                if h["name"].lower() == "importance"
                            ),
                            "",
                        ),
                        "content_type": next(
                            (
                                h["value"]
                                for h in headers
                                if h["name"].lower() == "content-type"
                            ),
                            "",
                        ),
                    }

                    # Extract unsubscribe URL from email body
                    def find_unsubscribe_link(payload):
                        unsubscribe_patterns = [
                            r'href=["\']?(.*?unsubscribe.*?)["\']?',
                            r'href=["\']?(.*?opt.?out.*?)["\']?',
                            r'href=["\']?(.*?remove.*?subscription.*?)["\']?',
                        ]

                        if "body" in payload and payload["body"].get("data"):
                            import base64

                            try:
                                body_data = base64.urlsafe_b64decode(
                                    payload["body"]["data"]
                                ).decode("utf-8")
                                for pattern in unsubscribe_patterns:
                                    matches = re.finditer(
                                        pattern, body_data, re.IGNORECASE
                                    )
                                    for match in matches:
                                        return match.group(1)
                            except Exception:
                                pass

                        if "parts" in payload:
                            for part in payload["parts"]:
                                if part.get("mimeType") == "text/html":
                                    result = find_unsubscribe_link(part)
                                    if result:
                                        return result
                        return ""

                    # Process payload and find unsubscribe link in content
                    if "payload" in response:
                        body_unsubscribe = find_unsubscribe_link(response["payload"])
                        email_data["body_unsubscribe_link"] = body_unsubscribe

                    # Extract email content and attachments
                    def get_content_and_attachments(
                        payload, attachments=None, content=None
                    ):
                        if attachments is None:
                            attachments = []
                        if content is None:
                            content = []

                        # Get body content
                        if "body" in payload:
                            if payload["body"].get("data"):
                                import base64

                                body_data = payload["body"]["data"]
                                try:
                                    decoded = base64.urlsafe_b64decode(
                                        body_data
                                    ).decode("utf-8")
                                    content.append(
                                        {
                                            "mime_type": payload.get("mimeType", ""),
                                            "content": decoded,
                                        }
                                    )
                                except Exception as e:
                                    content.append(
                                        {
                                            "mime_type": payload.get("mimeType", ""),
                                            "content": f"[Error decoding content: {str(e)}]",
                                        }
                                    )

                        # Handle attachments
                        if "filename" in payload and payload["filename"]:
                            attachments.append(
                                {
                                    "filename": payload["filename"],
                                    "mime_type": payload.get("mimeType", ""),
                                    "size": payload.get("body", {}).get("size", 0),
                                    "attachment_id": payload.get("body", {}).get(
                                        "attachmentId", ""
                                    ),
                                }
                            )

                        # Recursively process parts
                        if "parts" in payload:
                            for part in payload["parts"]:
                                get_content_and_attachments(part, attachments, content)

                        return content, attachments

                    # Process payload
                    if "payload" in response:
                        content, attachments = get_content_and_attachments(
                            response["payload"]
                        )
                        email_data["content"] = content
                        email_data["attachments"] = attachments
                        email_data["has_attachments"] = bool(attachments)

                    response_queue.append(email_data)

            # Add requests to batch
            for message in batch_messages:
                batch.add(
                    self.service.users()
                    .messages()
                    .get(
                        userId="me",
                        id=message["id"],
                        format="full",  # Keep as 'full' to get content and attachments
                    ),
                    callback=callback,
                )

            batch.execute()
            batch_data.extend(response_queue)
            return batch_data

        # Process batches sequentially
        print("Fetching email details...")
        for i, batch in enumerate(batches):
            try:
                batch_data = process_batch(batch)
                emails_data.extend(batch_data)
                print(f"Processed batch {i + 1} of {len(batches)}")
            except Exception as e:
                print(f"Batch {i} generated an exception: {e}")
                time.sleep(2)

        print(f"Successfully processed {len(emails_data)} emails")
        return emails_data

    def analyze_emails(self, emails_data):
        """Analyzes email patterns using domain-based grouping"""
        df = pd.DataFrame(emails_data)

        def extract_domain(email_str):
            """Extract domain from email address"""
            # First extract email from the "From" field if it's in format "Name <email>"
            email_match = re.search(r"<(.+?)>", email_str)
            email = email_match.group(1) if email_match else email_str

            # Then extract domain from email
            domain_match = re.search(r"@(.+)$", email)
            return domain_match.group(1).lower() if domain_match else "unknown"

        def clean_date(date_str):
            """Clean date string by removing timezone information in parentheses"""
            date_str = re.sub(r"\([^)]*\)", "", date_str)
            date_str = re.sub(r"\s+[A-Z]{2,4}(?:\s+|$)", " ", date_str)
            return date_str.strip()

        # Extract both full email and domain
        df["normalized_sender"] = df["from"].apply(
            lambda x: re.search(r"<(.+?)>", x).group(1)
            if re.search(r"<(.+?)>", x)
            else x
        )
        df["sender_domain"] = df["from"].apply(extract_domain)

        # Clean and parse dates
        df["date"] = pd.to_datetime(
            df["date"].apply(clean_date),
            format="mixed",
            utc=True,
            errors="coerce",
        )

        # Drop rows with invalid dates
        df = df.dropna(subset=["date"])

        # Calculate statistics by domain
        domain_stats = (
            df.groupby("sender_domain")
            .agg(
                {
                    "date": ["count", "min", "max"],
                    "normalized_sender": lambda x: len(
                        set(x)
                    ),  # unique senders per domain
                    "size_estimate": "sum",
                    "has_attachments": "sum",
                }
            )
            .reset_index()
        )

        # Rename columns
        domain_stats.columns = [
            "domain",
            "total_emails",
            "first_email",
            "last_email",
            "unique_senders",
            "total_size",
            "emails_with_attachments",
        ]

        # Calculate additional metrics
        domain_stats["days_span"] = (
            domain_stats["last_email"] - domain_stats["first_email"]
        ).dt.total_seconds() / 86400

        # Handle cases where emails were sent on the same day
        domain_stats["days_span"] = domain_stats["days_span"].apply(lambda x: max(x, 1))

        domain_stats["emails_per_day"] = (
            domain_stats["total_emails"] / domain_stats["days_span"]
        )
        domain_stats["attachment_ratio"] = (
            domain_stats["emails_with_attachments"] / domain_stats["total_emails"]
        )
        domain_stats["avg_size_per_email"] = (
            domain_stats["total_size"] / domain_stats["total_emails"]
        )

        # Add marketing email analysis
        def is_likely_marketing(domain):
            marketing_keywords = {
                "marketing",
                "newsletter",
                "mail",
                "info",
                "news",
                "updates",
                "notify",
                "noreply",
                "no-reply",
            }
            return any(keyword in domain.lower() for keyword in marketing_keywords)

        domain_stats["is_marketing_domain"] = domain_stats["domain"].apply(
            is_likely_marketing
        )

        # Sort by total emails
        domain_stats = domain_stats.sort_values("total_emails", ascending=False)

        # Calculate per-sender statistics within each domain
        sender_stats = (
            df.groupby(["sender_domain", "normalized_sender"])
            .agg(
                {
                    "date": ["count", "min", "max"],
                    "size_estimate": "mean",
                    "has_attachments": "mean",
                }
            )
            .reset_index()
        )

        sender_stats.columns = [
            "domain",
            "sender",
            "total_emails",
            "first_email",
            "last_email",
            "avg_size",
            "attachment_ratio",
        ]

        # Save detailed statistics to CSV files
        domain_stats.to_csv("domain_statistics.csv", index=False)
        sender_stats.to_csv("sender_statistics.csv", index=False)

        return domain_stats, sender_stats

    def plot_top_senders(self, domain_stats, sender_stats, top_n=20):
        """Creates visualizations for email analysis"""
        # Plot top domains by volume
        plt.figure(figsize=(20, 10))
        top_domains = domain_stats.nlargest(top_n, "total_emails")
        bars = plt.bar(range(len(top_domains)), top_domains["total_emails"])
        plt.xticks(
            range(len(top_domains)),
            top_domains["domain"],
            rotation=45,
            ha="right",
            fontsize=8,
        )

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
            )

        plt.title(f"Top {top_n} Email Domains by Volume")
        plt.xlabel("Domain")
        plt.ylabel("Number of Emails")
        plt.tight_layout()
        plt.savefig("top_domains.png", dpi=300)
        plt.close()

        # Plot domain activity (emails per day)
        plt.figure(figsize=(20, 10))
        top_active = domain_stats.nlargest(top_n, "emails_per_day")
        bars = plt.bar(range(len(top_active)), top_active["emails_per_day"])
        plt.xticks(
            range(len(top_active)),
            top_active["domain"],
            rotation=45,
            ha="right",
            fontsize=8,
        )

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
            )

        plt.title(f"Top {top_n} Active Domains (Emails per Day)")
        plt.xlabel("Domain")
        plt.ylabel("Average Emails per Day")
        plt.tight_layout()
        plt.savefig("top_active_domains.png", dpi=300)
        plt.close()

    def delete_emails_from_sender(self, sender, emails_df):
        """Delete all emails from a specific sender"""
        # Get message IDs for this sender
        sender_emails = emails_df[emails_df["normalized_sender"] == sender]

        # Get unsubscribe information from the first email
        if len(sender_emails) > 0:
            first_email = sender_emails.iloc[0]
            unsubscribe_links = []

            # Check header-based unsubscribe
            if first_email.get("list_unsubscribe"):
                unsubscribe_links.append(("Header", first_email["list_unsubscribe"]))

            # Check body-based unsubscribe
            if first_email.get("body_unsubscribe_link"):
                unsubscribe_links.append(("Body", first_email["body_unsubscribe_link"]))

        # Print deletion summary
        print(f"\nFound {len(sender_emails)} emails from {sender}")
        if unsubscribe_links:
            print("\nUnsubscribe options found:")
            for source, link in unsubscribe_links:
                print(f"{source}: {link}")

        # Ask for confirmation
        confirm = input(f"Delete all emails from {sender}? (y/n): ").lower()

        if confirm == "y":
            print(f"Deleting {len(sender_emails)} emails from {sender}...")

            # Create batch request for deletion
            batch = self.service.new_batch_http_request()
            for _, row in sender_emails.iterrows():
                batch.add(
                    self.service.users().messages().trash(userId="me", id=row["id"])
                )

            # Execute deletion
            batch.execute()
            print(f"Deleted {len(sender_emails)} emails from {sender}")

            if unsubscribe_links:
                print("\nPlease unsubscribe using one of these options:")
                for source, link in unsubscribe_links:
                    print(f"{source}: {link}")

            return True

        return False

    def process_unsubscribe_candidates(self, emails_data, domain_stats):
        """Process unsubscribe candidates interactively by domain"""
        # Convert emails_data to DataFrame if it's not already
        emails_df = (
            pd.DataFrame(emails_data)
            if not isinstance(emails_data, pd.DataFrame)
            else emails_data
        )

        # Extract domain from email addresses
        def extract_domain(email_str):
            """Extract domain from email address"""
            email_match = re.search(r"<(.+?)>", email_str)
            email = email_match.group(1) if email_match else email_str
            domain_match = re.search(r"@(.+)$", email)
            return domain_match.group(1).lower() if domain_match else "unknown"

        # Add normalized sender and domain columns
        emails_df["normalized_sender"] = emails_df["from"].apply(
            lambda x: re.search(r"<(.+?)>", x).group(1)
            if re.search(r"<(.+?)>", x)
            else x
        )
        emails_df["sender_domain"] = emails_df["from"].apply(extract_domain)

        # Sort domains by combination of volume and frequency
        candidates = domain_stats.copy()
        candidates["priority_score"] = (
            candidates["total_emails"]
            * candidates["emails_per_day"]
            * (candidates["is_marketing_domain"].astype(int) + 1)
        )
        candidates = candidates.sort_values("priority_score", ascending=False)

        # Process each domain
        processed_count = 0
        for _, row in candidates.iterrows():
            domain = row["domain"]
            print(f"\n{'-'*50}")
            print(f"Domain: {domain}")
            print(f"Total emails: {row['total_emails']}")
            print(f"Emails per day: {row['emails_per_day']:.2f}")
            print(f"Marketing domain: {'Yes' if row['is_marketing_domain'] else 'No'}")
            print(f"Unique senders: {row['unique_senders']}")

            # Show unique senders from this domain
            domain_senders = emails_df[emails_df["sender_domain"] == domain][
                "normalized_sender"
            ].unique()
            print("\nSenders from this domain:")
            for i, sender in enumerate(domain_senders, 1):
                print(f"{i}. {sender}")

            # Ask which sender to process
            while True:
                choice = input(
                    "\nEnter sender number to delete their emails (0 to skip, 'all' for entire domain): "
                )
                if choice.lower() == "all":
                    # Process all senders from domain
                    for sender in domain_senders:
                        if self.delete_emails_from_sender(sender, emails_df):
                            processed_count += 1
                    break
                elif choice == "0":
                    break
                else:
                    try:
                        sender_idx = int(choice) - 1
                        if 0 <= sender_idx < len(domain_senders):
                            if self.delete_emails_from_sender(
                                domain_senders[sender_idx], emails_df
                            ):
                                processed_count += 1
                            break
                        else:
                            print("Invalid sender number")
                    except ValueError:
                        print("Please enter a valid number or 'all'")

            # Ask if user wants to continue after every few domains
            if processed_count % 3 == 0:
                continue_process = input("\nContinue with next domain? (y/n): ").lower()
                if continue_process != "y":
                    break

    def analyze_unsubscribe_links(self, emails_data):
        """Analyzes unsubscribe links across email providers"""
        df = pd.DataFrame(emails_data)

        def extract_domain(email_str):
            email_match = re.search(r"<(.+?)>", email_str)
            email = email_match.group(1) if email_match else email_str
            domain_match = re.search(r"@(.+)$", email)
            return domain_match.group(1).lower() if domain_match else "unknown"

        # Extract sender domains
        df["sender_domain"] = df["from"].apply(extract_domain)

        # Collect unsubscribe data
        unsubscribe_data = []
        for domain in df["sender_domain"].unique():
            domain_emails = df[df["sender_domain"] == domain]

            # Get all unique unsubscribe links for this domain
            header_links = set(
                link
                for link in domain_emails["list_unsubscribe"].dropna()
                if link.strip()
            )
            body_links = set(
                link
                for link in domain_emails["body_unsubscribe_link"].dropna()
                if link.strip()
            )

            if header_links or body_links:
                unsubscribe_data.append(
                    {
                        "domain": domain,
                        "email_count": len(domain_emails),
                        "header_unsubscribe": "|".join(header_links),
                        "body_unsubscribe": "|".join(body_links),
                        "has_header_unsubscribe": bool(header_links),
                        "has_body_unsubscribe": bool(body_links),
                    }
                )

        # Convert to DataFrame and sort
        unsubscribe_df = pd.DataFrame(unsubscribe_data)
        unsubscribe_df = unsubscribe_df.sort_values("email_count", ascending=False)

        # Save to CSV
        unsubscribe_df.to_csv("unsubscribe_links.csv", index=False)

        # Print summary
        print("\nUnsubscribe Link Analysis:")
        print(f"Total domains with unsubscribe links: {len(unsubscribe_df)}")
        print(
            f"Domains with header unsubscribe: {unsubscribe_df['has_header_unsubscribe'].sum()}"
        )
        print(
            f"Domains with body unsubscribe: {unsubscribe_df['has_body_unsubscribe'].sum()}"
        )

        return unsubscribe_df


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

        print("Saving raw email data...")
        df_raw = pd.DataFrame(emails_data)
        df_raw.to_csv("all_emails_raw.csv", index=False)

    # print("Analyzing emails...")
    # analyzer = GmailAnalyzer()
    # analyzer.authenticate()  # Need to authenticate for deletion

    # # Analyze unsubscribe links first
    # print("\nAnalyzing unsubscribe links...")
    # unsubscribe_df = analyzer.analyze_unsubscribe_links(emails_data)

    # # Continue with regular analysis
    # domain_stats, sender_stats = analyzer.analyze_emails(emails_data)

    # print("\nStarting interactive unsubscribe and deletion process...")
    # analyzer.process_unsubscribe_candidates(emails_data, domain_stats)


if __name__ == "__main__":
    main()
