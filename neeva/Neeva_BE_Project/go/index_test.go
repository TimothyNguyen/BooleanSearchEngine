package index_test

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"testing"
)

// Tweet holds the contents and the timestamp of a tweet.
type Tweet struct {
	contents  string
	timestamp int
}

// TweetIndex is the interface that we ask you to implement. We have provided a sample implementation.
type TweetIndex interface {
	ProcessTweets(tweets []Tweet)
	Search(query string) []Tweet
}

// SimpleTweetIndex is our implementation of a TweetIndex. Please create your own, more efficient, implementation of a TweetIndex.
type SimpleTweetIndex struct {
	tweets []Tweet
}

// NewSimpleTweetIndex creates a SimpleTweetIndex.
func NewSimpleTweetIndex() TweetIndex {
	return &SimpleTweetIndex{tweets: []Tweet{}}
}

// ProcessTweets processes a list of tweets and initializes any data structures needed for
// searching over them.
func (s *SimpleTweetIndex) ProcessTweets(tweets []Tweet) {
	s.tweets = append(s.tweets, tweets...)
}

// Search looks for the most recent tweet (highest timestamp) that contains all words in the query.
// If no such tweet exists, returns an empty slice of Tweets.
// NOTE: Please update this docstring to reflect the updated specification of your search function
func (s *SimpleTweetIndex) Search(query string) []Tweet {
	queryWords := strings.Split(query, " ")
	var resultTweet Tweet
	var resultTs = -1
	for _, t := range s.tweets {
		tweetContainsWord := true
		for _, word := range queryWords {
			if !strings.Contains(t.contents, word) {
				tweetContainsWord = false
			}
		}
		if tweetContainsWord && t.timestamp > resultTs {
			resultTweet, resultTs = t, t.timestamp
		}
	}
	if resultTs == -1 {
		return []Tweet{}
	}
	return []Tweet{resultTweet}
}

func TestSearch(t *testing.T) {
	ti := NewSimpleTweetIndex()
	// A longer list of tweets is available in data/tweets.csv for your use.
	ti.ProcessTweets(readTweets(t, "../data/small.csv"))
	cases := []struct {
		query    string
		expected []Tweet
	}{
		{query: "hello", expected: []Tweet{{contents: "hello this is also neeva", timestamp: 15}}},
		{query: "hello me", expected: []Tweet{{contents: "hello not me", timestamp: 14}}},
		{query: "hello bye", expected: []Tweet{{contents: "hello bye", timestamp: 3}}},
		{query: "hello this bob", expected: []Tweet{{contents: "hello neeva this is bob", timestamp: 11}}},
		{query: "notinanytweets", expected: []Tweet{}},
	}
	for testNum, testCase := range cases {
		results := ti.Search(testCase.query)
		if len(testCase.expected) == len(results) {
			for i := range results {
				if results[i] != testCase.expected[i] {
					t.Errorf("Found unequal tweet results in test case %v: %v, %v", testNum, results, testCase.expected)
				}
			}
		} else {
			t.Errorf("Found unequal tweet results: %v, %v", results, testCase.expected)
		}
	}
}

// Returns the array of tweets read from the provided tweet file.
func readTweets(t *testing.T, filename string) []Tweet {
	in, err := os.Open(filename)
	if err != nil {
		t.Error(err)
	}
	defer in.Close()
	records, err := csv.NewReader(in).ReadAll()
	if err != nil {
		t.Error(err)
	}
	result := make([]Tweet, len(records))
	for i, record := range records {
		if i == 0 {
			// Skip the header
			continue
		}
		timestamp, err := strconv.Atoi(record[0])
		if err != nil {
			t.Error(err)
		}
		result[i].contents = record[1]
		result[i].timestamp = timestamp
	}
	return result
}

// This can be run with: go test -bench=. index_test.go
func BenchmarkSearch(b *testing.B) {
	testWords := []string{"neeva", "happy", "search", "quality", "tweet", "ranking"}
	cases := []struct {
		tweets []Tweet
	}{
		{generateTweets(testWords, 100, 2)},
		{generateTweets(testWords, 10000, 3)},
	}
	for _, tc := range cases {
		s := NewSimpleTweetIndex()
		s.ProcessTweets(tc.tweets)
		b.Run("Search"+strconv.Itoa(len(tc.tweets)), func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				word1 := testWords[n%len(testWords)]
				word2 := testWords[(n+1)%len(testWords)]
				s.Search(fmt.Sprintf("%s %s", word1, word2))
			}
		})
	}
}

// Returns a slice of `n` randomly generated tweets consisting of `k` words each.
func generateTweets(words []string, n, k int) []Tweet {
	tweets := make([]Tweet, n)
	for i := 0; i < n; i++ {
		rand.Shuffle(len(words), func(i, j int) { words[i], words[j] = words[j], words[i] })
		tweets[i] = Tweet{strings.Join(words[:k], " "), i}
	}
	return tweets
}
