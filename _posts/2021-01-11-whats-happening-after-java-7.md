---
title: What's happening after Java 7
date: 2021-01-11 16:48:59 +0530
categories: [Learning, Java]
tags: [Notes, Java]
math: true
pin: false
image:
  path: https://w0.peakpx.com/wallpaper/281/257/HD-wallpaper-java-logo.jpg
  lqip: data:image/webp;base64,UklGRpoAAABXRUJQVlA4WAoAAAAQAAAADwAABwAAQUxQSDIAAAARL0AmbZurmr57yyIiqE8oiG0bejIYEQTgqiDA9vqnsUSI6H+oAERp2HZ65qP/VIAWAFZQOCBCAAAA8AEAnQEqEAAIAAVAfCWkAALp8sF8rgRgAP7o9FDvMCkMde9PK7euH5M1m6VWoDXf2FkP3BqV0ZYbO6NA/VFIAAAA
  alt: What's happening after Java 7
---

## Introduction

Java has changed so much over the past few years most profound so far being the Java 8 release. For example, remember when the code for sorting a collections based on a member variable used to look like this:
```java
    Collections.sort(inventory, new Comparator<Apple>() {
        public int compare (Apple a, Apple b) {
            return a.getWeight().compareTo(b.getWeight());
        }
    });
```
In Java 8 this could be rewritten as:
```java
    inventory.sort(comparing(Apple::getWeight));
```

## Java's claim to fame

Object orientation became popular in 1990s for two reasons:

- Encapsulation discipline resulted in fewer software engineering issues that those of C
- Write-Once Read-Anywhere (WORA) model of Java, thanks to the fact that the compiled bytecode can be run on any JVM including applets on browsers.

There was an initial resistance to the additional run cost of Java vs C/C++ but with time machines got faster and programmer time became more and more important.

## Evolution of Java

Java has a rich history and has constantly evolved over the years with the prime goal of making concurrency simpler to use and less error-prone. At the time Java 1.0 came out, it had threads and even a memory model. Java 5 added more abstracted concurrency tools like thread pools and concurrent collections. Java 7 built on top of that and added the fork/join framework making parallelism even more practical but difficult.

Over the past few years commodity CPUs have become multicore. Java as a language didn't leverage this power available hence the vast majority of the Java program use only one of these cores.

## Stream Processing

Java 8 provided a simpler way to look at parallelism. Java 8 provided a new API called Streams that supports many parallel operations to process and modify data in a way similar to database query languages.

Streams abstract away all the code that uses "synchronized" etc to exploit the multi-core capabilities of CPUs so that the code is less error-prone and easy to use. Using Streams API we don't have to use loops or code for parallelism since it is abstracted away within the library. Streams API takes care of partitioning and forks/joins are abstracted away. But it's important to note that the methods passed to the library methods **must not interact and have no shared mutable state**.

## Behavior Parameterization

Looking at the above Java 8 code for sorting, we are actually passing a function as a parameter to the method "comparing()". Java 8 introduced "*behavior parameterization*" wherein we can pass functions like variables in a concise way. This feature of passing functions like variables gave access to a range of addition techniques commonly referred to as *functional-style programming*.

## Methods as first-class citizens

In a way, the whole point of a programming language is to manipulate values. Values were previously listed as first-class Java citizens but methods and classes exemplify second-class citizens. For example, methods are used to defined classes which in turn may be instantiated to product values but neither are values themselves.

It turns out that being able to pass methods at runtime hence making them first-class citizens is useful in programming. We can make other second-class citizens like classes to be first-class citizens, for example JavaScript!

Analogous to using an *object reference*, in Java 8 when we write 
```java
    Apple::getWeight
```

we create a *method reference* which can be simply passed around.

## Lambdas: Anonymous Functions

A ***lambda function*** is a small anonymous ***function***. A ***lambda function*** can take any number of arguments, but can only have one expression. 
```java
    (int x) -> x + 1
```
For example the above code would mean "the function that when called with argument 'x', returns the value 'x+1' "

Though we could achieve the same by defining a function and use the method reference, but it's annoying to write definitions for short methods when they're used perhaps once or twice.

## Default methods for interfaces

Prior to Java 8 you can update an interface only if you update all the classes that implement it! This prevented any attempt for the Java libraries to evolve and be modified. Default methods prepared Java to deal with API evolution in an elegant way.

Also the default methods for interfaces and optional objects features also served as the foundation for the Java streaming API.

Previously we could sort a list only from the "Collections" utility class method since adding a sort method to List interface would mean defining the implementation for each and every class that implements it! With Java 8, a sort method has been added to List interface with a default implementation in List interface.
```java
    default void sort(Comparator<? super e> c) {
    	Collections.sort(this, c);
    }
```
Default interfaces also introduce a form of multiple inheritance in Java. Since a class can implement multiple interfaces we can inherit multiple default implementations.

## Other good ideas from functional programming

One good practice is avoiding the use of *null* in favor of more descriptive datatype, case in point, *Optional<T> *class that was introduced in Java 8. If and when used consistently, it can help avoid null-pointer exceptions altogether.

It's basically a container object which may or may not contain a value and includes methods to explicitly deal with the case where a value is absent.

Another thought is why should the *switch* statement be limited to primitive values and strings. Functional languages then to allow switch to be used on many more datatypes. including allowing pattern matching.

## References

- [Modern Java in Action by Raoul-Gabriel Urma, Alan Mycroft, Mario Fusco](https://www.oreilly.com/library/view/modern-java-in/9781617293566/)
