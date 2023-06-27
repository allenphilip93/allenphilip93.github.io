---
title: Newton's method for finding roots
date: 2020-08-27 16:51:00 +0530
categories: [Algorithm]
tags: [Course, Learning]
math: true
pin: false
---

## What is the problem statement?

Given the following equation $$f(x)=0$$ we want to find the roots of the equation, i.e for what values of $x$ the function $f(x)$ goes to $0$. We assume that $f(x)$ is continuous and differentiable and a root exists.

## Newton's method

In 1664, Issac Newton came up with an iterative method to solve this problem. However, this method is sometimes also known as the Raphson's method since Raphson invented the same algorithm a few years after Newton published it.

The input to Newton's method takes in the function $f(x)$ as well as initial approximate guess $x_i$ for the root.

Newton's method attempts to iteratively to improve the guess. According to the method, the next guess $x_{i+1}$ would be $$x_{i+1} = x_i - \frac{f(x_i)}{f'(x_i)} $$

We iteratively repeat this until $f(x_{i+1}) \approx 0$

The next guess $x_{i+1}$ is computed by drawing a tangent to the function $f(x)$ at ${x_i}$ and the point where the tangent intersects with the X-axis will be $x_{i+1}$.

## Mathematics behind Newton's method

We know that the equation of a line is $$ \begin{equation} y = mx + c \end{equation} $$

Let's say that our starting guess is $x_i$. The slope at $x_i$ on $f(x)$ will be the differential at $x_i$ which is $f'(x_i)$. Also we know that $(x_i, f(x_i)$ lies on the line, so we can substitute the point into $(1)$.

$$f(x_i) = x_i f'(x_i) + c $$ Rearranging the terms we get: $$ c = f(x_i) - x_i f'(x_i) $$ Equation $(1)$ now becomes: $$ \begin{equation} y = x f'(x_i) + f(x_i) - x_i f'(x_i) \end{equation} $$

Now we know that when this line intersects the X-axis, $y=0$ and let's say $x = x_{i+1}$ Substituting this into equation $(2)$ we get: $$ 0 = x_{i+1} f'(x_i) + f(x_i) - x_i f'(x_i) $$ $$ \begin{equation} x_{i+1} = x_i - \frac{f(x_i)}{f'(x_i)} \end{equation} $$

## Example using Newton's method

Consider a quadratic function $$ f(x) = x^2 - 4 $$ We know that $$ f'(x) = 2x $$ Substitute the values into equation (3) $$ \begin{equation} x_{i+1} = x_i - \frac{x_i^2 - 4}{2 x_i} \end{equation} $$ Let's say $x_0 = 1$. After the first iteration $$ x_1 = 2.5 $$ It's quite clear we have overshot the root at $x=2$. So let's see what will the guess after one more iteration but substituting $x_1 = 2.5$ into equation (4) $$ x_2 = 2.05 $$ Values for the next few iterations look like $$ x_3 = 2.000609756097561 $$ $$ x_4 = 2.0000000929222947 $$

## Does Newton's method always converge?

We can observe from the above example for quadratic functions, newton's method converges as long as a root exists since the second order derivative is constant. This means our approximation gets better with each iteration.

Wiki has a more formal proof on the [quadratic convergence of Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method#Proof_of_quadratic_convergence_for_Newton%27s_iterative_method).

But more higher order functions like say a cubic function, [convergence is not guarenteed](https://math.stackexchange.com/questions/2407659/why-does-the-newton-raphson-method-not-converge-for-some-functions/2407690#:~:text=8%20Answers&amp;text=Newton%27s%20method%20does%20not%20always,can%20have%20highly%20nontrivial%20dynamics.) since based on the starting point of the guess and the other local peaks, it might end up [oscillating indefinitely and never find a solution](https://www.wolframalpha.com/input/?i=Use+Newton%27s+method+to+solve+-0.74+%2B+0.78*x+%2B+1.1*x%5E2+-+3.55*x%5E3+%3D+0+with+x0+%3D+0.54).

## Applications of Newton's method

The more direct implementation is in finding the square root of a rational number within a certain error.

    double sqrt_newton(double n) {
      const double eps = 1E-15;
      double x = 1;
      for (;;) {
          double nx = (x + n / x) / 2;
          if (abs(x - nx) < eps)
              break;
          x = nx;
      }
      return x;
    }
    

Another variant is when we want to find the integer root of a number. Finding the large $x$ such that $x^2 \le n$.

    int isqrt_newton(int n) {
      int x = 1;
      bool decreased = false;
      for (;;) {
          int nx = (x + n / x) >> 1;
          if (x == nx || nx > x && decreased)
              break;
          decreased = nx < x;
          x = nx;
      }
      return x;
    }
    

For calculating square root of large numbers it is simple and effective to take the initial guess as $ 2 ^ {nbits/2} $ where $ nbits $ is the number of bits in the number $ N $.
