"""
Web scraping utilities for the website assistant.

This module retrieves webpages, parses their HTML content, extracts
visible text and page metadata, and collects links for use by the
website search system.
"""
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def scrape_website(url):
    """
    Scrapes the HTML content and links from a webpage.

    Retrieves the webpage at the provided URL, parses its HTML, extracts
    the page title and hyperlinks, removes script and style elements, and
    returns the remaining visible text for use by the search system.

    Args:
        url: The URL of the webpage to scrape.

    Returns:
        A dictionary containing the page URL, title, visible text, and
        hyperlinks. Returns None if the webpage cannot be retrieved.
    """

    # Add HTTPS when the user provides a URL without a protocol.
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        # Retrieve the webpage and reject unsuccessful HTTP responses.
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Parse the returned HTML into a searchable document structure.
        soup = BeautifulSoup(response.text, "html.parser")

        title = (
            soup.title.string.strip()
            if soup.title and soup.title.string
            else ""
        )

        # Extract hyperlinks before modifying the parsed HTML.
        links = []

        for link in soup.find_all("a", href=True):
            link_text = link.get_text(strip=True)
            link_url = urljoin(url, link["href"])

            links.append({
                "text": link_text,
                "url": link_url
            })

        # Remove non-content elements before extracting visible page text.
        for element in soup(["script", "style"]):
            element.decompose()

        text = soup.get_text(separator=" ", strip=True)

        return {
            "url": url,
            "title": title,
            "text": text,
            "links": links
        }

    except requests.RequestException as error:
        print(f"Error scraping website: {error}")
        return None