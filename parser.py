import nltk
nltk.download('punkt')
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> S Conj S | NP VP NP | NP VP | NP VP NP Adv | NP VP Adv | S Conj VP | S Conj VP NP
NP -> Det A N | Det N | N | NP P NP | P Det A N | P Det N | P N | P NP P NP
A -> A Adj | Adj
VP -> Adv V | V | V Adv


"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """

    words = nltk.word_tokenize(sentence, language='English')
    processed = list()
    for w in words:
        containsAlpha = False
        for letter in w:
            if letter.isalpha():
                containsAlpha = True
                break
        if containsAlpha:
            processed.append(w.lower())
    print(processed)
    return processed


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    stack = list()
    chunks = list()
    stack = list(tree.subtrees())
    while stack:
        sub = stack.pop(len(stack) - 1)
        children = sub.subtrees()
        if sub.label() == 'NP':
            if 'NP' not in children:
                chunks.append(sub)
    return chunks

if __name__ == "__main__":
    main()
