import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def scrape_website(url):
    # Add https:// if the user didn't include it
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        # Get the webpage
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Parse the HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # Get the page title
        title = soup.title.string.strip() if soup.title and soup.title.string else ""

        # Find all links on the page
        links = []

        for link in soup.find_all("a", href=True):
            link_text = link.get_text(strip=True)
            link_url = urljoin(url, link["href"])

            links.append({
                "text": link_text,
                "url": link_url
            })

        # Remove code that isn't useful website information
        for element in soup(["script", "style"]):
            element.decompose()

        # Extract visible text
        text = soup.get_text(separator=" ", strip=True)

        # Return everything the brain may need
        return {
            "url": url,
            "title": title,
            "text": text,
            "links": links
        }

    except requests.RequestException as error:
        print(f"Error scraping website: {error}")
        return None


if __name__ == "__main__":
    url = input("Enter website URL: ")

    page = scrape_website(url)

    if page:
        print("\n--- TITLE ---")
        print(page["title"])

        print("\n--- WEBSITE INFORMATION ---")
        print(page["text"])

        print("\n--- LINKS ---")
        for link in page["links"]:
            print(f'{link["text"]} -> {link["url"]}')