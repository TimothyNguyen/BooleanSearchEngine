"""
Please use Python version 3.7+
"""

import csv
from typing import List, Tuple

class TweetIndex:
    # Starter code--please override
    def __init__(self):
        self.index = "inverted_index.txt"
        self.vocab_index = {}
        self.timestamp_to_tweets = {}

    # Starter code--please override
    def process_tweets(self, list_of_timestamps_and_tweets: List[Tuple[int, str]]) -> None:
        """
        process_tweets processes a list of tweets and initializes any data structures needed for
        searching over them.

        :param list_of_timestamps_and_tweets: A list of tuples consisting of a timestamp and a tweet.
        """
        inverted_index = {}

        for row in list_of_timestamps_and_tweets:
            timestamp = int(row[0])
            tweet = str(row[1])
            tweet_list = tweet.lower().split(" ")
            self.timestamp_to_tweets[timestamp] = tweet
            for word in tweet_list:
                if word not in inverted_index:
                    inverted_index[word] = {}
                # Add the posting list for that inverted index
                if timestamp not in inverted_index[word]:
                    inverted_index[word][timestamp] = 0
                inverted_index[word][timestamp] += 1

        # Store it in a file
        with open(self.index, mode='w') as f:
            for word, posting_list in inverted_index.items():
                self.vocab_index[word] = f.tell()
                f.write(word + " ")
                f.write(str(posting_list))
                f.write("\n")

        f.close()

    # Accessing Index Information
    def accessTermInfoIndex(self, term: str, index: str, vocabIndex: dict):
        # enter your code here
        posting_list = []
        if term in vocabIndex:
            offset = vocabIndex[term]
            with open(index, mode='r') as f:
                f.seek(offset)
                line = f.readline()
                term, posting_list = line.split(" ", 1)
                posting_list = eval(posting_list)
            f.close()
        return posting_list

    '''
    & (ampersand) means logical AND (both the word/expression to the left and right
    must exist in the resulting tweet

    | = (pipe) means logical OR (either the word/expression to the left or the right
    must be in the resulting tweet, or both)

    ! (exclamation point) means logical NOT (the returned tweets should NOT
    contain the following word/expression following this operator)
    Spaces in the query will just exist to separate out words and operators from one
    another
    '''
    # Modified shunting-yard algorithm
    def parse_query(self, query:str) -> List[str]:
        infix_tokens = query.lower().split(" ")
        precedence = {'!': 3, '&': 2, '|': 1, '(': 0, ')':0}
        op_stack = list()
        ans = []
        for token in infix_tokens:
            if token == '(': op_stack.append(token)
            elif token == ')':
                op = op_stack.pop()
                while op != '(':
                    ans.append(op)
                    op = op_stack.pop()
            elif token in precedence:
                if op_stack:
                    curr_op = op_stack[-1]
                    while precedence[curr_op] > precedence[token] and op_stack:
                        op = op_stack.pop()
                        ans.append(op)
                        if op_stack:
                            curr_op = op_stack[-1]
                op_stack.append(token)
            else:
                ans.append(token)

        while op_stack:
            ans.append(op_stack.pop())
        return ans


    # Starter code--please override
    def search(self, query: str) -> List[Tuple[str, int]]:
        """
        NOTE: Please update this docstring to reflect the updated specification of your search function
        search looks for the most recent tweet (highest timestamp) that contains all words in query.

        :param query: the given query string
        :return: a list of tuples of the form (tweet text, tweet timestamp), ordered by highest timestamp tweets first. 
        If no such tweet exists, returns empty list.
        """
        query = query.replace('(', '( ')
        query = query.replace(')', ' )')
        # query = query.replace(" ", " & ") -> Need to fix this if there's a space
        # between words and (, ), !
        token_list_postfix = self.parse_query(query)

        output_stack = []

        while token_list_postfix:
            token = token_list_postfix.pop(0)
            if token == '&':
                t2 = output_stack.pop()
                t1 = output_stack.pop()
                temp_res = list(set(t1).intersection(set(t2)))
            elif token == '|':
                t2 = output_stack.pop()
                t1 = output_stack.pop()
                temp_res = list(set(t1).union(set(t2)))
            elif token == '!':
                t = output_stack.pop()
                temp_res = list(set(self.timestamp_to_tweets.keys()) - set(t))
            else:
                # Check if word in inverted index
                temp_res = self.accessTermInfoIndex(token, self.index, self.vocab_index)
                if len(temp_res) > 0:
                    temp_res = temp_res.keys()
            output_stack.append(temp_res)

        if len(output_stack) == 1:
            ans = []
            five_recent_timestamps = list(output_stack.pop())[-5:]
            if len(five_recent_timestamps) == 0:
                return [('', -1)]
            for t in five_recent_timestamps:
                ans.append((self.timestamp_to_tweets[t], t))
            return ans
        return [('', -1)]
if __name__ == "__main__":
    # A full list of tweets is available in data/tweets.csv for your use.
    tweet_csv_filename = "../data/tweets.csv"
    list_of_tweets = []
    with open(tweet_csv_filename, "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        for i, row in enumerate(csv_reader):
            if i == 0:
                # header
                continue
            timestamp = int(row[0])
            tweet = str(row[1])
            list_of_tweets.append((timestamp, tweet))

    ti = TweetIndex()
    ti.process_tweets(list_of_tweets)
    print(ti.search("! two"))
    #assert ti.search("hello")[-1] == ('hello this is also neeva', 15)
    #assert ti.search("hello me")[-1] == ('hello not me', 14)
    #assert ti.search("hello bye")[-1] == ('hello bye', 3)
    #assert ti.search("hello this bob")[-1] == ('hello neeva this is bob', 11)
    #assert ti.search("notinanytweets")[-1] == ('', -1)
    # assert ti.search("Noovi & is & (fast | (very & quick))")
