import wikipediaapi
import pandas as pd
import concurrent.futures
from tqdm import tqdm


def wiki_scrape(start_page_name, verbose=True):
    """Method to scrape Wikipedia pages associated with/linked to a starting page
    
    Parameters
    ----------
    start_page_name : str
        Name of page to start scraping from
    verbose : bool, optional
        Flag for displaying progress bar and verbose output

    Returns
    -------
    sources : pd.DataFrame
        DataFrame containing all scraped Wikipedia articles linked to start_page_name,
        with entries ('page', 'text', 'link', 'categories')
        
    References
    ----------
    Modified from https://towardsdatascience.com/auto-generated-knowledge-graphs-92ca99a81121
    """
    def follow_link(link):
        """Helper function to follow links using Wikipedia API
        """
        try:
            page = wiki_api.page(link)
            if page.exists():
                d = {'page': link, 'text': page.text, 'link': page.fullurl,
                     'categories': list(page.categories.keys())}
                return d
            else:
                return None
        except:
            return None
    
    # Instantiate Wikipedia API
    wiki_api = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)
    
    # Scrape starting page
    page_name = wiki_api.page(start_page_name)
    if not page_name.exists():
        print('page does not exist')
        return
    
    # Initialize dict (to be converted to df)
    sources = [{
        'page': start_page_name, 
        'text': page_name.text, 
        'link': page_name.fullurl,
        'categories': list(page_name.categories.keys())
    }]
    
    page_links = set(page_name.links.keys())
    # Multiprocessing to parallely scrape from multiple pages
    progress = tqdm(desc='Links Scraped', unit='', total=len(page_links)) if verbose else None
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Follow links from list of links
        future_link = {executor.submit(follow_link, link): link for link in page_links}
        for future in concurrent.futures.as_completed(future_link):
            data = future.result()
            progress.update(1) if verbose else None
            if data:
                # Update sources dict
                sources.append(data)
    progress.close() if verbose else None
    
    # Convert dict to df
    sources = pd.DataFrame(sources)
    
    # Filter out generic Wikipedia pages
    blacklist = ('Template', 'Help:', 'Category:', 'Portal:', 'Wikipedia:', 'Talk:')
    sources = sources[(len(sources['text']) > 20)
                      & ~(sources['page'].str.startswith(blacklist))]
    sources['categories'] = sources.categories.apply(lambda x: [y[9:] for y in x])
    
    return sources


def build_category_whitelist(wiki_data, page_whitelist, cat_blacklist):
    """Helper function to build whitelist of page categories
    
    This method finds a set of page categories which we can use to 
    reduce the amount of pages we use for building knowledge graphs. 
    Typically, we want to build KGs about particular domains and 
    specific pages.
    
    Parameters
    ----------
    wiki_data : pd.DataFrame
        
    page_whitelist : list
        List of pages from whose categories we select a domain-specific subset
    cat_blacklist : list
        List of categories which we don't want to include in the whitelist
        
    Returns
    -------
    cat_whitelist : set/list
        List of categories which we want to build KGs about
    """
    cat_whitelist = []
    for page_name in page_whitelist:
        # Iterate over categies list for each page in the page whitelist
        categories = list(wiki_data[wiki_data.page==page_name].categories)[0]
        for cat in categories:
            relevant_cat = True
            for unwanted in cat_blacklist:
                # If given category is part of blacklisted categories, 
                # do not add it to whitelist
                if unwanted in cat:
                    relevant_cat = False
                    break
            
            # All non-blacklisted categories from the page whitelist 
            # are added to categories whitelist
            if relevant_cat:
                cat_whitelist.append(cat)
                
    return set(cat_whitelist)
