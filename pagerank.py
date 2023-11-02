import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000
ACCURACY = 0.001


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
    distribution = dict()
    if len(corpus[page]) == 0:
        for p in corpus:
            distribution[p] = 1 / len(corpus)
    else:
        for p in corpus:
            distribution[p] = (1 - damping_factor) / len(corpus)

    for p in corpus[page]:
        distribution[p] += damping_factor / len(corpus[page])

    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    sample = list()
    current = random.choice(list(corpus.keys()))
    for i in range(n):
        sample.append(current)
        distribution = dict(transition_model(corpus, current, damping_factor))
        current = random.choices(list(distribution.keys()), list(distribution.values()))[0]

    page_rank = dict()
    for page in corpus.keys():
        page_rank[page] = sample.count(page) / n

    return page_rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    isDifference = True
    page_rank = dict()
    for page in corpus.keys():
        page_rank[page] = 1 / N
    while isDifference:
        isDifference = False
        for page in corpus.keys():
            psum = 0.0
            for other in corpus.keys():
                if page in corpus[other]:
                    psum += page_rank[other] / len(corpus[other])
            current_rank = page_rank[page]
            page_rank[page] = (1 - damping_factor) / N + damping_factor * psum
            if abs(page_rank[page] - current_rank) > ACCURACY:
                isDifference = True

    return page_rank


if __name__ == "__main__":
    main()
