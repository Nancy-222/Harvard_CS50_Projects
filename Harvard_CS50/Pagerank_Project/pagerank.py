import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    prob_distribution={}
    N=len(corpus)
    if page in corpus:
        for one_link in corpus:
            prob_distribution[one_link]=(1-damping_factor)/N
        if len(corpus[page])>0:
            for one_link in corpus[page]:
                prob_distribution[one_link]+=damping_factor/len(corpus[page])
    else:
        for one_link in corpus:
            prob_distribution[one_link]=1/N
    return prob_distribution
    

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_count={}
    PR={}
    for page in corpus:
        page_count[page]=0
    N=len(corpus)
    start_page=random.choice(list(corpus.keys()))
    for i in range(n):
        transition_prob=transition_model(corpus,start_page,damping_factor) 
        if not transition_prob:
            break
        next_page_to_visit=random.choices(list(transition_prob.keys()),weights=list(transition_prob.values()))[0]
        page_count[next_page_to_visit] += 1
        start_page=next_page_to_visit
    for page,count in page_count.items():
        PR[page]=count/n
    sum_PR = sum(PR.values())
    for page in PR:
        PR[page] /= sum_PR
    return PR


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    dict1={}
    dict2={}
    N=len(corpus)
    for page in corpus:
        dict1[page]=1/N
        dict2[page]=1/N
    while True:
        max_diff = 0
        for page in corpus:
            new_PR = 0
            for outgoing_links in corpus:
                if page in corpus[outgoing_links]:
                    link_PR = dict1[outgoing_links]
                    new_PR += link_PR / len(corpus[outgoing_links])
                if len(corpus[outgoing_links]) == 0:
                    new_PR += dict1[outgoing_links] / N
            new_PR = (1 - damping_factor) / N + damping_factor * new_PR
            diff = abs(dict1[page] - new_PR)
            if diff > max_diff:
                max_diff = diff
            dict2[page] = new_PR

        if max_diff <= 0.001:
            sum_PR = sum(dict2.values())
            for page in dict2:
                dict2[page] /= sum_PR
            return dict2

        dict1 = dict2.copy()


if __name__ == "__main__":
    main()
