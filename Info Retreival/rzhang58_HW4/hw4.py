import logging
import re
import sys
from bs4 import BeautifulSoup
from queue import Queue,PriorityQueue
from urllib import parse, request

logging.basicConfig(level=logging.DEBUG, filename='output.log', filemode='w')
visitlog = logging.getLogger('visited')
extractlog = logging.getLogger('extracted')


def parse_links(root, html):
    soup = BeautifulSoup(html, 'html.parser')
    for link in soup.find_all('a'):
        href = link.get('href')
        if href:
            text = link.string
            if not text:
                text = ''
            text = re.sub('\s+', ' ', text).strip()
            yield (parse.urljoin(root, link.get('href')), text)


def parse_links_sorted(root, html,key_w):
    # TODO: implement
    soup = BeautifulSoup(html, 'html.parser')
    for link in soup.find_all('a'):
        href = link.get('href')
        if href:
            text = link.string
            if not text:
                text = ''
            text = re.sub('\s+', ' ', text).strip()
            url = parse.urljoin(root, link.get('href'))
            yield ((relevance_func(url,key_w),url),text)


def get_links(url):
    res = request.urlopen(url)
    return list(parse_links(url, res.read()))

def relevance_func(link,key_w):
    if len(key_w) == 0:
        return 0
    for w in key_w:
        if w in link:
            return 0
    else:
        return 1

def get_nonlocal_links(url):
    '''Get a list of links on the page specificed by the url,
    but only keep non-local links and non self-references.
    Return a list of (link, title) pairs, just like get_links()'''

    # TODO: implement
    links = get_links(url)
    filtered = []
    for link in links:
        if (parse.urlparse(link[0]).hostname != parse.urlparse(url).hostname):
            filtered.append(link)
    return filtered


def crawl(root, wanted_content=[], within_domain=True,num_link=10,key_w=[]):
    '''
    Crawl the url specified by `root`.
    `wanted_content` is a list of content types to crawl
    `within_domain` specifies whether the crawler should limit itself to the domain of `root`
    '''
    # TODO: implement

    queue = PriorityQueue()
    queue.put((relevance_func(root,key_w),root))

    visited = set()
    extracted = []

    while not queue.empty():
        if len(visited) > num_link:
            break
        rank,url = queue.get()
        try:
            req = request.urlopen(url)
            wanted_content = [x.lower() for x in wanted_content]
            content = req.headers['Content-Type'].lower() 
            if wanted_content and (content not in wanted_content):
                continue
            html = req.read()

            visited.add(url)
            visitlog.debug(url)

            for ex in extract_information(url, html):
                extracted.append(ex)
                extractlog.debug(ex)

            for (rank,link), title in parse_links_sorted(url, html,key_w):
                if (link in visited) or (parse.urlparse(link) == parse.urlparse(root)) or (within_domain and parse.urlparse(link).hostname != parse.urlparse(url).hostname):
                    continue
                queue.put((rank,link))

        except Exception as e:
            print(e, url)

    return visited, extracted



def extract_information(address, html):
    '''Extract contact information from html, returning a list of (url, category, content) pairs,
    where category is one of PHONE, ADDRESS, EMAIL'''

    # TODO: implement
    results = []
    for match in re.findall('\d\d\d-\d\d\d-\d\d\d\d', str(html)):
        results.append((address, 'PHONE', match))

    for match in re.findall(r"([a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z0-9.-]+)", str(html)):
        results.append((address, 'EMAIL', match))

    for match in re.findall(r'([A-Z]([a-z])+(\.?)(\x20[A-Z]([a-z])+){0,2},\s[A-Za-z]+\.{0,1}\s(?!0{5})\d{5})', str(html)):
        results.append((address, 'ADDRESS', match[0]))
    return results


def writelines(filename, data):
    with open(filename, 'w') as fout:
        for d in data:
            print(d, file=fout)


def main():
    site = sys.argv[1]
    
    num_link = input("Enter how many links you would like the crawler to visit: ")
    num_l = int(num_link)
    
    key_w = []
    while True:
        quit = input("Enter any keyword you would like to prioritize in link (or press Enter to quit): ")
        if not quit:
            break
        key_w.append(quit)

    
    links = get_links(site)
    writelines('links.txt', links)

    nonlocal_links = get_nonlocal_links(site)
    writelines('nonlocal.txt', nonlocal_links)

    visited, extracted = crawl(site,[],True,num_link = num_l,key_w = key_w)
    writelines('visited.txt', visited)
    writelines('extracted.txt', extracted)


if __name__ == '__main__':
    main()