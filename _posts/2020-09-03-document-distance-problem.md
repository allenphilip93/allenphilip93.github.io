---
title: Performance tuning the document distance problem
date: 2020-09-03 10:05:00 +0530
categories: [Learning, Algorithm]
tags: [Course, Python]
math: true
pin: false
image:
  path: https://kinsta.com/wp-content/uploads/2017/05/how-to-optimize-images-for-web-and-performance-1200x675.jpg
  lqip: data:image/webp;base64,UklGRpoAAABXRUJQVlA4WAoAAAAQAAAADwAABwAAQUxQSDIAAAARL0AmbZurmr57yyIiqE8oiG0bejIYEQTgqiDA9vqnsUSI6H+oAERp2HZ65qP/VIAWAFZQOCBCAAAA8AEAnQEqEAAIAAVAfCWkAALp8sF8rgRgAP7o9FDvMCkMde9PK7euH5M1m6VWoDXf2FkP3BqV0ZYbO6NA/VFIAAAA
  alt: Performance tuning the document distance problem
---

## Problem Statement

Given two documents $D1$ and $D2$, we would like to compute a sense of "distance" which would give us a measure of the degree of similarity between the two documents.

The documents could be thought of a huge string with a lot of "words". A "word" can be defined as a sequence of alpha-numeric characters separated by spaces.

## Defining the distance function

In order to compute the distance between the two documents we need a mathematical function to arrive at a number which gives us a measure of similarity.

One option is to count the number of occurrences of each word in both the documents. We can represent this word count map as a vector where the each unique word will be a dimension and the magnitude will be the number of occurences.

For example, $$D1 = "I~want~a~cat~or~a~dog"$$ $$D2 = "I~have~a~cat"$$

The vector form for both the documents would look like,

$$D1 = 1\widehat{i} + 1\widehat{want} + 2\widehat{a} + 1\widehat{cat} + 1\widehat{or} + 1\widehat{dog}$$

$$D2 = 1\widehat{i} + 1\widehat{have} + 1\widehat{a} + 1\widehat{cat}$$

### Dot Product

Now that we have two vectors, dot product is a good metric to give us a measure of similarity between two vectors. We can compute the distance function as dot product for the above example as follows, $$d'(D1,D2) = 5$$

Now say I have another document, $$D3 = "She~wants~a~dog"$$ and compute the distance from $D2$, $$d'(D2,D3)=1$$

From this we can infer that $D2$ is more similar to $D1$ as compared to $D3$.

But there is a downside which is the fact that dot product is scale invariant. For example, if I have two documents with 100 words out of which 99 words are similar, the distance would be 99. Consider two other documents with 1000 words out of which 200 words are similar, the distance would be 200. This leads to an incorrect inference that the second pair of documents are more similar with each other than the first pair.

Ofcourse this can be solved by dividing the dot product with the magnitude of both the document vectors.

$$ d'(D1,D2) = \frac{D1.D2}{|D1||D2|}$$

### Arcosine Distance Function

The above equation is nothing but the cosine of the angle between the two vectors. So it's fair to say that the angle between the two vectors would be a good scale invariant estimate of the distance between the two documents.

$$ d(D1,D2) = cos^{-1}{\frac{D1.D2}{|D1||D2|}}$$

## Pseudocode

- Read the files
- Split the data into words
- Count the number of occurrences of words in each document
- Compute the inner product
- Compute the angle between the two documents using the above formula

## Solution v1.0

### Reading the files

    def read_file(filename):
        """ 
        Read the text file with the given filename;
        return a list of the lines of text in the file.
        """
        try:
            f = open(filename, 'r')
            return f.readlines()
        except IOError:
            print("Error opening or reading input file: ",filename)
            sys.exit()
    

Read the file as a list of string with each element representing a line terminated by "\n" in the document.

### Split the data into words

    def get_words_from_line_list(L):
        """
        Parse the given list L of text lines into words.
        Return list of all words found.
        """
    
        word_list = []
        for line in L:
            words_in_line = get_words_from_string(line)
            word_list = word_list + words_in_line
        return word_list
     
    def get_words_from_string(line):
        """
        Return a list of the words in the given input string,
        converting each word to lower-case.
    
        Input:  line (a string)
        Output: a list of strings 
                  (each string is a sequence of alphanumeric characters)
        """
        word_list = []          # accumulates words in line
        character_list = []     # accumulates characters in word
        for c in line:
            if c.isalnum():
                character_list.append(c)
            elif len(character_list)>0:
                word = "".join(character_list)
                word = word.lower()
                word_list.append(word)
                character_list = []
        if len(character_list)>0:
            word = "".join(character_list)
            word = word.lower()
            word_list.append(word)
        return word_list
    

We iterate through the list of strings and split each line into words. If there are "$N1, N2$" words in documents respectively, the runtime for this stage would be $O(N1+N2)$.

### Count the number of occurrences of words

    def count_frequency(word_list):
        """
        Return a list giving pairs of form: (word,frequency)
        """
        L = []
        for new_word in word_list:
            for entry in L:
                if new_word == entry[0]:
                    entry[1] = entry[1] + 1
                    break
            else:
                L.append([new_word,1])
        return L
    

We will use a list of tuples of the form (word, frequency). The runtime complexity of this stage would be $O(N1^2 + N2^2)$.

### Compute the inner product

    def insertion_sort(A):
        """
        Sort list A into order, in place.
    
        From Cormen/Leiserson/Rivest/Stein,
        Introduction to Algorithms (second edition), page 17,
        modified to adjust for fact that Python arrays use 
        0-indexing.
        """
        for j in range(len(A)):
            key = A[j]
            # insert A[j] into sorted sequence A[0..j-1]
            i = j-1
            while i>-1 and A[i]>key:
                A[i+1] = A[i]
                i = i-1
            A[i+1] = key
        return A
    
    def inner_product(L1,L2):
        """
        Inner product between two vectors, where vectors
        are represented as lists of (word,freq) pairs.
    
        Example: inner_product([["and",3],["of",2],["the",5]],
                               [["and",4],["in",1],["of",1],["this",2]]) = 14.0 
        """
        sum = 0.0
        for word1, count1 in L1:
            for word2, count2 in L2:
                if word1 == word2:
                    sum += count1 * count2
        return sum
    

Since we have a list of tuples of the form (word, frequency), we would like to sort the list for the both documents first using insertion sort which would take $O(N1log(N1) + N2log(N2))$.

Once sorted, we iterate through both the lists and compute the dot product. This stage would run at $O(N1 * N2)$.

### Compute the angle

    def vector_angle(L1,L2):
        """
        The input is a list of (word,freq) pairs, sorted alphabetically.
    
        Return the angle between these two vectors.
        """
        numerator = inner_product(L1,L2)
        denominator = math.sqrt(inner_product(L1,L1)*inner_product(L2,L2))
        return math.acos(numerator/denominator)
    

We reuse the inner product function to compute the denominator as well. This stage has a runtime complexity of $O(N1 ^ 2 + N2 ^2)$.

### Performance

The text files used and the source files can be found at the GitHub repo shared below.

    File pg-grimm.txt : 9569 lines, 105324 words, 5172 distinct words
    File pg-huckleberry_finn.txt : 12361 lines, 120896 words, 6519 distinct words
    The distance between the documents is: 0.460007 (radians)
             3003939 function calls in 39.250 seconds
    

## Solution v2.0

### Split the data into words

    def get_words_from_line_list_ver2(L):
        """
        Parse the given list L of text lines into words.
        Return list of all words found.
        """
    
        word_list = []
        for line in L:
            words_in_line = get_words_from_string(line)
            # Using "extend" is much more efficient than concatenation here:
            word_list.extend(words_in_line)
        return word_list
    

Instead of concatenating the two lists directly, let's use the "extend" method.

### Performance

    File pg-grimm.txt : 9569 lines, 105324 words, 5172 distinct words
    File pg-huckleberry_finn.txt : 12361 lines, 120896 words, 6519 distinct words
    The distance between the documents is: 0.460007 (radians)
             3025869 function calls in 23.266 seconds
    

## Solution v3.0

### Compute the inner product

    def inner_product_ver3(L1,L2):
        """
        Inner product between two vectors, where vectors
        are represented as alphabetically sorted (word,freq) pairs.
    
        Example: inner_product([["and",3],["of",2],["the",5]],
                            3   [["and",4],["in",1],["of",1],["this",2]]) = 14.0 
        """
        sum = 0.0
        i = 0
        j = 0
        while i<len(L1) and j<len(L2):
            # L1[i:] and L2[j:] yet to be processed
            if L1[i][0] == L2[j][0]:
                # both vectors have this word
                sum += L1[i][1] * L2[j][1]
                i += 1
                j += 1
            elif L1[i][0] < L2[j][0]:
                # word L1[i][0] is in L1 but not L2
                i += 1
            else:
                # word L2[j][0] is in L2 but not L1
                j += 1
        return sum
    

In the previous algorithm for inner product, we have not leverages the fact that the two lists are sorted! Once we fix this we can reduce our runtime complexity from $O(N1*N2)$ to $O(min(N1,N2))$.

### Performance

    File pg-grimm.txt : 9569 lines, 105324 words, 5172 distinct words
    File pg-huckleberry_finn.txt : 12361 lines, 120896 words, 6519 distinct words
    The distance between the documents is: 0.460007 (radians)
             3066906 function calls in 16.672 seconds
    

## Solution v4.0

### Count the number of occurrences of words

    def count_frequency_ver4(word_list):
        """
        Return a list giving pairs of form: (word,frequency)
        """
        D = {}
        for new_word in word_list:
            if new_word in D:
                D[new_word] = D[new_word]+1
            else:
                D[new_word] = 1
        return list(D.items())
    

Instead of using a list, we can use dictionaries (HashMaps) to count the number of occurrences of words. This brings down the complexity from from $O(N1^2 + N2^2)$ to $O(N1+N2)$. 

### Performance

    File pg-grimm.txt : 9569 lines, 105324 words, 5172 distinct words
    File pg-huckleberry_finn.txt : 12361 lines, 120896 words, 6519 distinct words
    The distance between the documents is: 0.460007 (radians)
             3055217 function calls in 10.172 seconds
    

## Solution v5.0

### Split the data into words

    # global variables needed for fast parsing
    # translation table maps upper case to lower case and punctuation to spaces
    translation_table = str.maketrans(string.punctuation+string.ascii_uppercase,
                                         " "*len(string.punctuation)+string.ascii_lowercase)
    
    def get_words_from_string_ver5(line):
        """
        Return a list of the words in the given input string,
        converting each word to lower-case.
    
        Input:  line (a string)
        Output: a list of strings 
                  (each string is a sequence of alphanumeric characters)
        """
        line = line.translate(translation_table)
        word_list = line.split()
        return word_list
    

Instead of manually splitting the line into words we can use the python translation table feature and apply the translation table on each line to convert it into a word list.

### Performance

    File pg-grimm.txt : 9569 lines, 105324 words, 5172 distinct words
    File pg-huckleberry_finn.txt : 12361 lines, 120896 words, 6519 distinct words
    The distance between the documents is: 0.460007 (radians)
             129615 function calls in 4.484 seconds
    

## Solution v6.0

### Compute the inner product

    def merge_sort(A):
        """
        Sort list A into order, and return result.
        """
        n = len(A)
        if n==1: 
            return A
        mid = n//2     # floor division
        L = merge_sort(A[:mid])
        R = merge_sort(A[mid:])
        return merge(L,R)
    
    def merge(L,R):
        """
        Given two sorted sequences L and R, return their merge.
        """
        i = 0
        j = 0
        answer = []
        while i<len(L) and j<len(R):
            if L[i]<R[j]:
                answer.append(L[i])
                i += 1
            else:
                answer.append(R[j])
                j += 1
        if i<len(L):
            answer.extend(L[i:])
        if j<len(R):
            answer.extend(R[j:])
        return answer
    

If we use merge sort instead of insertion sort we can improve the runtime complexity from $O(N^2)$ to $O(NlogN)$ at the expense of addition space complexity at $O(N)$.

### Performance

    File pg-grimm.txt : 9569 lines, 105324 words, 5172 distinct words
    File pg-huckleberry_finn.txt : 12361 lines, 120896 words, 6519 distinct words
    The distance between the documents is: 0.460007 (radians)
             635398 function calls (612020 primitive calls) in 1.453 seconds
    

## Solution v7.0

Let's use more hashing to remove sorting altogether.

### Count the number of occurrences of words

    def count_frequency_ver7(word_list):
        """
        Return a dictionary mapping words to frequency.
        """
        D = {}
        for new_word in word_list:
            if new_word in D:
                D[new_word] = D[new_word]+1
            else:
                D[new_word] = 1
        return D
    

To start with, let's return the dictionary instead of converting it into a list of tuples. This immediately removes the overhead of converting a dictionary to a list of tuples.

### Compute the inner product

    def inner_product_ver7(D1,D2):
        """
        Inner product between two vectors, where vectors
        are represented as dictionaries of (word,freq) pairs.
    
        Example: inner_product({"and":3,"of":2,"the":5},
                               {"and":4,"in":1,"of":1,"this":2}) = 14.0 
        """
        sum = 0.0
        for key in D1:
            if key in D2:
                sum += D1[key] * D2[key]
        return sum
    

Now that we have a dictionary which maintains the number of occurrences of each word, we can use that to compute the inner product in $O(min(N1,N2))$ and we can remove sorting completely!

### Performance

    File pg-grimm.txt : 9569 lines, 105324 words, 5172 distinct words
    File pg-huckleberry_finn.txt : 12361 lines, 120896 words, 6519 distinct words
    The distance between the documents is: 0.460007 (radians)
             88564 function calls in 0.312 seconds
    

## Solution v8.0

### Reading the files

Why not just treat the whole file as one big line instead of maintaining a list of lines?! 

    def get_words_from_line_list_ver8(text):
        """
        Parse the given text into words.
        Return list of all words found.
        """
        text = text.translate(translation_table)
        word_list = text.split()
        return word_list
    

This removes the overhead of calling "extend()" to concatenate the word lists.

### Performance

    File pg-grimm.txt : 540174 lines, 105324 words, 5172 distinct words
    File pg-huckleberry_finn.txt : 594262 lines, 120896 words, 6519 distinct words
    The distance between the documents is: 0.460007 (radians)
             554 function calls in 0.094 seconds
    

## Resources

Lecture - [https://www.youtube.com/watch?v=Zc54gFhdpLA&ab_channel=MITOpenCourseWare](https://www.youtube.com/watch?v=Zc54gFhdpLA&amp;ab_channel=MITOpenCourseWare)
Code - [https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/lecture-videos/lec02_code.zip](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/lecture-videos/lec02_code.zip)
GitHub - [https://github.com/allenphilip93/mit-6.006-document-distance](https://github.com/allenphilip93/mit-6.006-document-distance)
