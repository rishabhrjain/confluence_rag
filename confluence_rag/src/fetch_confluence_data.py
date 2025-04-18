import requests
import json
from requests.auth import HTTPBasicAuth
from urllib.parse import quote
import time

from utils import save_json, extract_formatted_content_from_atlas_doc
from confluence_rag.config import CONFLUENCE_BASE_URL, USERNAME, API_TOKEN, CACHE_DIR, SPACES_DIR, PAGES_DIR, CONTENTS_DIR


class ConfluenceService:

    def __init__(self, username, api_token, base_url):

        self.username = username
        self.api_token = api_token
        self.base_url = base_url if base_url else CONFLUENCE_BASE_URL 

        self.headers = {
                "Accept": "application/json", 
                }
        self.auth = HTTPBasicAuth(USERNAME, API_TOKEN)

        # create all dir if doesn't exist
        CACHE_DIR.mkdir(exist_ok=True)
        SPACES_DIR.mkdir(exist_ok=True)
        PAGES_DIR.mkdir(exist_ok=True)
        CONTENTS_DIR.mkdir(exist_ok=True)

    def get_all_public_spaces(self, use_cache = True):
        """
        Get all pages that are marked as public. These are generally team spaces. We can also pull private spaces but 
        that didn't seem to have any valuable info
        """

        cache_file = SPACES_DIR / "spaces.json"
        if use_cache and cache_file.exists() :
            print("Fetching spaces data from cache..")
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)

            except json.JSONDecodeError:
                print("Cache file for spaces is corrupted. Fetching fresh data..")



        url = f"{self.base_url}/wiki/api/v2/spaces"
        spaces = []
        while True:
            response = requests.request(
                    "GET",
                    url,
                    headers=self.headers,
                    auth=self.auth
                    )
            response = json.loads(response.text)

            # keep space if its public

            for space in response['results']:
                if space['type'] == "global":
                    spaces.append(space)

            # get next URL
            if response.get("_links", {}).get("next"):
                url = self.base_url + response['_links']['next']
                time.sleep(1)

            else:
                break
        
        # save spaces data
        save_json(dir=SPACES_DIR,filename="spaces.json", data=spaces)

        return spaces
    

    def get_all_pages(self, spaces: list[dict],  use_cache=True):
        """
        For list of given spaces, fetch the all page_ids and its content. 
        we can pull data from all spaces including personal but generally that might not be of high quality. 

        Args:
            use_cache (bool, optional): Uses cache if data exists
            spaces: List of confluence spaces to pull data from
        """
        cache_file = PAGES_DIR / "raw_pages.json"
        if use_cache and cache_file.exists() :
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    print("Fetching page_id data from cache..")
                    return json.load(f)

            except json.JSONDecodeError:
                print("Cache file for spaces is corrupted. Fetching fresh data..")

        # fetches 250 page at max
        url = f"{self.base_url}/wiki/api/v2/pages?body-format=atlas_doc_format&limit=250&space-id="

        space_ids = []
        for space in spaces:
            space_ids.append(str(space['id']))
        
        # make it a comma seperated list
        space_ids = ','.join(space_ids)
        url += space_ids

        pages = []
        i = 0
        while True:
            print(f"page: {i}")
            i += 1
            response = requests.request(
                    "GET",
                    url,
                    headers=self.headers,
                    auth=self.auth
                    )
            response = json.loads(response.text)

            # keep space if its public
            pages.extend(response['results'])

            # get next URL
            if response.get("_links", {}).get("next"):
                url = self.base_url + response['_links']['next']
                time.sleep(2)

            else:
                break
        
        # save page_ids data
        save_json(dir=PAGES_DIR,filename="raw_pages.json", data=pages)

        return pages
    

    def process_page_content(self, raw_pages: list[dict]):
        """
        Process raw page data with HTML content to useful readable text. 

        Args:
            pages (list[dict]): List of pages
        """
        processed_pages = []
        for page in raw_pages:

            # When doc type is body-format=storage
            # page_html_content = page['body']['storage']['value']
            # page_cleaned_content = clean_html_text(page_html_content)

            # When doc type is body-format=atlas_doc_format
            atlas_doc_content = page['body']['atlas_doc_format']['value']
            page_cleaned_content = extract_formatted_content_from_atlas_doc(atlas_doc_content)

            ## if page has less than 50 words, remove it. this might need to be modified once we start considering page attachments. 
            if len(page_cleaned_content.split()) < 50:
                continue

            page_data = {'page_id': page['id'], 
                    'space_id': page['spaceId'],
                    'created_at' : page['createdAt'], 
                    'page_url' : self.base_url + "/wiki" + page['_links']['webui'],
                    'page_title' : page['title'],
                    'parent_id' : page['parentId'], 
                    'page_content': page_cleaned_content

                    }
            processed_pages.append(page_data)
            

        save_json(dir = PAGES_DIR, filename='cleaned_pages.json', data=processed_pages)
             


if __name__ == "__main__":

    
    confluence_service = ConfluenceService(USERNAME, API_TOKEN, CONFLUENCE_BASE_URL)
    spaces = confluence_service.get_all_public_spaces()

    raw_pages = confluence_service.get_all_pages(spaces, use_cache=True)

    cleaned_pages = confluence_service.process_page_content(raw_pages)

    

    print("fetched pages")




