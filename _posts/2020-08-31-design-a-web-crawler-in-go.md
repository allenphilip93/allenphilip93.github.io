---
title: Design Web Crawler from Go Tutorial
date: 2020-08-31 13:05:00 +0530
categories: [Golang]
tags: [Course, Learning]
math: true
pin: false
---

## What is a Web Crawler?

A Web Crawler is an internet bot that systematically crawls through the World Wide Web typically for the purpose of web indexing.

A web crawler starts with a list of URLs, which it visits and proceeds to collect all the URLs in each page in the form of hyperlinks for example. The web crawler then adds these URLs to the list of web pages to visit and continues crawling though it's list of URLs recursively.

## Applications of Web Crawlers

**Google** and other search engines use web crawlers to** update their search indexes**. Each search engine that **has** its own index also **has** its own **web crawler**. If you want to see your **web** pages on **Google's** search result pages, Googlebot **has** to visit your pages first.

Crawlers can also be used for automating **maintenance tasks** on a Web site, such as checking links or validating HTML code.

## Decisions made by Web Crawlers

As the web crawler crawls through the web, it needs to have a **policy** in place to make certain decisions on the fly based on the purpose it serves:

- Which pages should it visit and which should it skip?
- When should it re-visit pages in case they're updated?
- How to decide when to stop since the web crawler might be overloading the website?
- How to parallelize and crawl through the URLs? 

## Defining our Web Crawler policy

Let's keep it simple and say that that the goal of the web crawler in our case is to just capture as many URLs as it can.

When the web crawler will visits a URL, it adds all the URLs on the page to a list. In the next iteration, it picks a URL from its list and repeats the process. This continues recursively until the list is empty.

Given how vast the World Wide Web is, we wouldn't want the web crawler to keep crawling forever. So let's set a **limit on the depth** of the crawl. For example, say we have URL1 -> URL2 -> URL3 and our depth is 2. We start with URL1 at depth 1 and add its child URL2 to our list. We then pick up URL2 but we won't add it's children to our list since it's at depth 2.

There is yet another problem, what if URL1 has a link to URL2 which in turn has a link to URL1. We will end up in a infinite loop! In order to simplify let's say that we will **never revisit** any URL that we have already crawled.

Unfortunately in our example, the web crawler is going to be inconsiderate and **not check** if it's going to **overload **a website or not.

Crawling through millions of URLs would not be feasible with a single threaded web crawler, but let's **start **with a **single threaded** crawler and build on it.

For context, I'd strongly advise to go check out the problem once starting at [https://tour.golang.org/concurrency/10](https://tour.golang.org/concurrency/10).

## Serial Web Crawler

This is a very simple implementation where we have a single threaded web crawler, which maintains a lookup dictionary to check if it has already visited a URL or not.

    //
    // Serial crawler
    //
    
    func Serial(url string, fetcher Fetcher, fetched map[string]bool) {
    	if fetched[url] {
    		return
    	}
    	fetched[url] = true
    	urls, err := fetcher.Fetch(url)
    	if err != nil {
    		return
    	}
    	for _, u := range urls {
    		Serial(u, fetcher, fetched)
    	}
    	return
    }
    

## Concurrent Web Crawler

Now let's try to modify the serial web crawler so that we can crawl the URLs concurrently.

The quickest way would be to recursively create goroutines while calling the Serial() method as follows.

    func IncorrectCrawler(url string, fetcher Fetcher, fetched map[string]bool) {
    	if fetched[url] {
    		return
    	}
    	fetched[url] = true
    	urls, err := fetcher.Fetch(url)
    	if err != nil {
    		return
    	}
    	for _, u := range urls {
            // Call as goroutines
    		go IncorrectCrawler(u, fetcher, fetched)
    	}
    	return
    }
    

If you try running the above code, you'll realize that not more than one URL gets parsed. This is because the main thread exits before the goroutines can complete their tasks.

So it's clear that the main thread must **wait** for the goroutines to complete before it exits. We can achieve the same using "sync.WaitGroup" as follows.

    type fetchState struct {
    	fetched map[string]bool
    }
    
    func IncorrectCrawler(url string, fetcher Fetcher, f *fetchState) {
    	already := f.fetched[url]
    	f.fetched[url] = true
    	
    	if already {
    		return
    	}
    
    	urls, err := fetcher.Fetch(url)
    	if err != nil {
    		return
    	}
    	var done sync.WaitGroup
    	for _, u := range urls {
    		done.Add(1)
            go func(urls []string) {
                defer done.Done()
                IncorrectCrawler(urls, fetcher, f)
            }(u)
    	}
    	done.Wait()
    	return
    }
    

"sync.WaitGroup" is nothing but a counter. If a URL has 5 children, we call Add(1) five times so the counter is at 5. Each time the "done.Done()" is called, the counter decreases by 1. "done.Wait()" blocks the thread until the counter reaches 0.

Oddly when we try running this as well, we notice that some URLs are missing. This is because we are passing "u" directly to the closure. "u" has a reference to a list of URLs which gets updated as we iterate through the "urls" object from the "for" loop. Hence the goroutines suddenly see different values for "u" in the middle of execution.

So let's fix this bug.

    type fetchState struct {
    	fetched map[string]bool
    }
    
    func IncorrectCrawler(url string, fetcher Fetcher, f *fetchState) {
    	already := f.fetched[url]
    	f.fetched[url] = true
    	
    	if already {
    		return
    	}
    
    	urls, err := fetcher.Fetch(url)
    	if err != nil {
    		return
    	}
    	var done sync.WaitGroup
    	for _, u := range urls {
    		done.Add(1)
            u_copy := u
            go func() {
                defer done.Done()
                IncorrectCrawler(u_copy, fetcher, f)
            }()
    	}
    	done.Wait()
    	return
    }
    

We copy the reference onto another variable so that it wouldn't be updated later by the "for" loop and pass that onto the closure.

If we try running the code now, ***most likely*** we will end up with the correct results. This is becomes more evident if we run:

    > go run -race crawler.go
    

"-race" flag checks for any race conditions in the code which might pop up at runtime. An errors pops up saying that a variable is being accessed right before it gets modified by another goroutine.

    already := f.fetched[url]
    f.fetched[url] = true
    

Race condition in the above code is that a goroutine must race to complete both these statements before the next goroutine starts executing this statement. Here "f" is an object shared by all the goroutines, so we should be careful while reading and updating it.

The code can be fixed as follows:

    //
    // Concurrent crawler with shared state and Mutex
    //
    
    type fetchState struct {
    	mu      sync.Mutex
    	fetched map[string]bool
    }
    
    func ConcurrentMutex(url string, fetcher Fetcher, f *fetchState) {
    	f.mu.Lock()
    	already := f.fetched[url]
    	f.fetched[url] = true
    	f.mu.Unlock()
    
    	if already {
    		return
    	}
    
    	urls, err := fetcher.Fetch(url)
    	if err != nil {
    		return
    	}
    	var done sync.WaitGroup
    	for _, u := range urls {
    		done.Add(1)
            u2 := u
    		go func() {
    			defer done.Done()
    			ConcurrentMutex(u2, fetcher, f)
    		}()
    	}
    	done.Wait()
    	return
    }
    

Having a lock on the mutex object ensures that the read and update operation on "f.fetched" is atomic in nature.

This code works but there is still one downside. There is no way to constraint the parallelism. The web crawler could easily end up creating over 1 million threads which might even break the server.

## Concurrent Web Crawler using channels

We can think of a distributed web crawler as a master node which adds URLs to a go channel and concurrently distributes the URLs to workers who crawl the URLs and fetch all the URLs from the page and put it back into the channel.

Go channels have are thread-safe and buffered channels have this cool property to block the receiver until the buffer is full or if there are no more values to send to the receiver. This was we can have a control on the degree of concurrency at all times.

    //
    // Concurrent crawler with channels
    //
    
    func worker(url string, ch chan []string, fetcher Fetcher) {
    	urls, err := fetcher.Fetch(url)
    	if err != nil {
    		ch <- []string{}
    	} else {
    		ch <- urls
    	}
    }
    
    func master(ch chan []string, fetcher Fetcher) {
    	// n is used to control the depth limit
        n := 1
    	fetched := make(map[string]bool)
        // ch blocks the for loop until values are added to it
    	for urls := range ch {
    		for _, u := range urls {
    			if fetched[u] == false {
    				fetched[u] = true
    				n += 1
    				go worker(u, ch, fetcher)
    			}
    		}
    		n -= 1
    		if n == 0 {
    			break
    		}
    	}
    }
    
    func ConcurrentChannel(url string, fetcher Fetcher) {
    	ch := make(chan []string, 2)
    	go func() {
    		ch <- []string{url}
    	}()
    	master(ch, fetcher)
    }
    

## Resources

- Code - [https://pdos.csail.mit.edu/6.824/notes/crawler.go](https://pdos.csail.mit.edu/6.824/notes/crawler.go)
- Lecture - [https://www.youtube.com/watch?v=gA4YXUJX7t8&feature=emb_logo](https://www.youtube.com/watch?v=gA4YXUJX7t8&amp;feature=emb_logo)
