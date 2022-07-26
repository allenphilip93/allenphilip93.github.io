---
title: Understanding peak finding algorithm
date: 2020-08-27 15:55:00 +0530
categories: [Algorithm]
tags: [Python, Learning]
math: true
pin: false
---

## What is the problem statement?

Consider a 1-D number array, $A$ of size $N$. An element $A[i]$ where $i \in [0, N-1]$ is a peak if and only if $$A[i-1] \le A[i] \ge A[i+1]$$ Assume that the boundaries $A[-1] = A[N] = -\infty$.

Goal is to find **a peak** in the input array if it exists.

## Will there always be a peak?

Let's consider a few corner case examples starting with a constant value array like $A = [4,4,4,4]$. In this case all the 4 elements are value peaks since $A[i-1] = A[i] = A[i+1]$.

Now let's consider a monotonically increasing array like $A=[1,2,3,4]$. The last point is peak since $A[2] \le A[3]$.

Lastly let's looks at a monotonically decreasing array like $A=[4,3,2,1]$. The first point is peak since $A[0] \ge A[1]$.

We can conclude that yes there will always be a peak!

## Brute force approach

The most simplest approach would be to iterate through the array from left to right and check if each element is an peak. And when we encounter the peak we break and return that index.

    if A[0] <= A[1]:
      return 0
    if A[N-1] >= A[N-2]:
      return N-1
    for i in range(1,N-1):
      if A[i] >= A[i-1] and A[i] >= A[i+1]:
        return i
    

Another way to look at the problem would be to find the global maxima of the input array. But definition the global maxima would be greater than or equal to all other values in the array and hence a peak. But it's not efficient since it has to iterate through the whole array.

    m = 0
    for i in range(1,N):
      if A[i] > A[m]:
        m = i
    return m
    

## Divide & Conquer

We had stated earlier that given the boundary conditions there will always be a peak. Now if we slice the input array into two equal halves, when can we say that there will be peak in either of the halves?

Consider the left half, the boundary condition has changed wherein $A[0] = -\infty$ but $A[N/2]$ is finite. So we can be sure that there exists a peak if $A[N/2 - 1] \ge A[N/2]$.

Similarly for the right half, we can be certain that there exists a peak if $A[N/2 + 1] \ge A[N/2]$.

If $A[N/2] \ge A[N/2 - 1]$ and $A[N/2] \ge A[N/2 + 1]$, then $A[N/2]$ is a peak.

    def findpeak(A, start, end):
      mid = (start + end) / 2
      if A[mid] >= A[mid-1] and A[mid] >= A[mid+1]:
        return mid
      else if A[mid-1] >= A[mid]:
        return findpeak(A, 0, mid-1)
      else:
        return findpeak(A, mid+1, end)
    

The runtime complexity now reduces to $O(\log_2(N))$ which is exponentially better than the previous $O(N)$.

## Peak finding in 2-D

Now let's slightly extend the problem statement as follows.

Consider a 2-D number array $A$ of size $m \times n$. $A[i][j]$ is a peak if and only if $$A[i][j] \ge A[i-1][j]$$ $$A[i][j] \ge A[i][j-1]$$ $$A[i][j] \ge A[i+1][j]$$ $$A[i][j] \ge A[i][j+1]$$

As with the 1-D case, consider all boundaries to be $-\infty$ which guarentees that there will be a peak.

## Divide & Conquer

We can go for a brute force approach and find a peak with a runtime complexity of $O(m n)$ but we can do better with divide and conquer.

Let's split the array down the middle and consider two halves $A[:][0:n/2-1]$ and $A[:][n/2+1:n]$.

We will use the previous 1-D algorithm to find a peak along column $n/2$. If $A[x][n/2-1] \ge A[x][n/2]$, we could claim that left half $A[:][0:n/2-1]$ will have a peak. But does this claim hold true?

Consider this example,

$$\begin{bmatrix}

0 & 8 & 7 & 3 & 11\

2 & 9 & 5 & 13 & 12\

3 & 10 & 11 & 12 & 10

\end{bmatrix}$$

We will split along the third column, and the peak will be $A[0][2]$ which is $7$. Since $A[0][1] \ge A[0][2]$ we claim that there is a peak in the left half. But from the example it is clear that there is no peak in the left half, though there is a peak in the right half $A[1][3]$ of value $13$.

## Divide & Conquer - Correct Approach

The previous approach fails because the boundary condition for $A[2][1]$ is not upheld, since $A[2][1] \le A[2][2]$. We should select the left half only if there is atleast one element, say $A[x][y]$, on column $n/2-1$ which is greater than all the elements on column $n/2$.

By induction, if any element on column $n/2-1$ is larger than $A[x][y]$ then it's larger than all elements on column $n/2$.

Also if $A[x][y]$ is the maxima on column $n/2-1$, then we can be certain there is a peak on row $x$.

The final pseudocode would look something like:

    def findpeak(A, colStart, colEnd):
      midCol = (colStart + colEnd) / 2
      rowMax = findGlobalMaxima(A, midCol)
      if A[rowMax][midCol] >= A[rowMax][midCol-1] and A[rowMax][midCol] >= A[rowMax][midCol+1]:
        return rowMax,midCol
      else if A[rowMax][midCol-1] >= A[rowMax][midCol]:
        return findpeak(A, 0, midCol-1)
      else:
        return findpeak(A, midCol+1, colEnd)
    

The runtime complexity for the same would be $O(m \log_2(n))$ since findGlobalMaxima() would take $O(m)$ before each split.

## Alternative approaches

We could solve the above problem by either splitting row-wise or column-wise. So we could solve the same problem with a runtime complexity of $O(n \log_2 m)$ by splitting the array row-wise.

Another approach, albeit a less efficient one, is where we find the best possible neighbour from a starting point, say $(0,0)$. The best neighbour would be greater than the current element. We iteratively do this until we chance upon a case where we don't find a better neighbour in which case the current element is the peak. This has a runtime complexity of $O(m n)$.

## Resources

[YouTube MIT Lecture](https://www.youtube.com/watch?v=HtSuA80QTyo&amp;list=PLUl4u3cNGP61Oq3tWYp6V_F-5jb5L2iHb&amp;index=2&amp;t=0s)

[Assignments & Sample Code](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/assignments/)
