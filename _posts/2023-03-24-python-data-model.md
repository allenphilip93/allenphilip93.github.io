---
title: Python Data Model
date: 2023-03-24 16:20:00 +0530
categories: [Python]
tags: [Notes, Learning]
math: false
pin: false
---

# Reference

[Fluent Python - Chapter 1](https://learning.oreilly.com/library/view/fluent-python-2nd/9781492056348/ch01.html)

# Introduction

Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python’s elegant syntax and dynamic typing, together with its interpreted nature, make it an ideal language for scripting and rapid application development in many areas on most platforms.

The Python interpreter is easily extended with new functions and data types implemented in C or C++ (or other languages callable from C). Python is also suitable as an extension language for customizable applications.

**Why Python?** 
- Python also offers much more error checking than lower-level languages like C
- Python being a very-high-level language, it has high-level data types built in, such as flexible arrays and dictionaries
- Python allows you to split your program into modules that can be reused in other Python programs
- Python is an interpreted language, which can save you considerable time during program development because no compilation and linking is necessary
- Python is extensible: if you know how to program in C it is easy to add a new built-in function or module to the interpreter

Programs written in Python are typically much shorter than equivalent C, C++, or Java programs, for several reasons:
- The high-level data types allow you to express complex operations in a single statement;
- Statement grouping is done by indentation instead of beginning and ending brackets;
- No variable or argument declarations are necessary.

# A Pythonic Card Deck

Let's look at class that represents a deck of cards write in the most "pythonic" fashion.

```python
import collections

Card = collections.namedtuple('Card', ['rank', 'suit'])

class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()

    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits
                                        for rank in self.ranks]

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, position):
        return self._cards[position]
```

The first thing to note is the use of `collections.namedtuple` to construct a simple class to represent individual cards.

Next say we want to initialize a deck of cards and pick a card at random, we'd have to do the following:
```python
>>> deck = FrenchDeck()
>>> len(deck)
52

>>> deck[0]
Card(rank='2', suit='spades')
>>> deck[-1]
Card(rank='A', suit='hearts')

>>> from random import choice
>>> choice(deck)
Card(rank='3', suit='hearts')
```
Notice how we didn't have to create a new method for access the cards in the deck or selecting one at random. But it gets even better ..

```python
>>> deck[:3]
[Card(rank='2', suit='spades'), Card(rank='3', suit='spades'),
Card(rank='4', suit='spades')]
>>> deck[12::13]
[Card(rank='A', suit='spades'), Card(rank='A', suit='diamonds'),
Card(rank='A', suit='clubs'), Card(rank='A', suit='hearts')]

>>> for card in deck:  # doctest: +ELLIPSIS
...   print(card)

>>> for card in reversed(deck):  # doctest: +ELLIPSIS
...   print(card)

>>> Card('Q', 'hearts') in deck
True
```

Because our `__getitem__` delegates to the `[]` operator of `self._cards`, our deck automatically supports slicing. We can even select every 13th card starting from the card in the 12th index.

Iteration is often implicit. If a collection has no `__contains__` method, the `in` operator does a sequential scan. Case in point: `in` works with our `FrenchDeck` class because it is iterable.

Now saw we want to sort the deck of cards based on the card rank and suit. This would require an additional method but still can be done in a pythonic way.
```python
suit_values = dict(spades=3, hearts=2, diamonds=1, clubs=0)

def spades_high(card):
    rank_value = FrenchDeck.ranks.index(card.rank)
    return rank_value * len(suit_values) + suit_values[card.suit]

>>> for card in sorted(deck, key=spades_high):  # doctest: +ELLIPSIS
...      print(card)
Card(rank='2', suit='clubs')
Card(rank='2', suit='diamonds')
```

Although FrenchDeck implicitly inherits from the object class, most of its functionality is not inherited, but comes from leveraging the data model and composition. By implementing the special methods `__len__` and `__getitem__`, our FrenchDeck behaves like a standard Python sequence, allowing it to benefit from core language features (e.g., iteration and slicing) and from the standard library, as shown by the examples using `random.choice`, `reversed`, and `sorted`. Thanks to composition, the `__len__` and `__getitem__` implementations can delegate all the work to a `list` object, `self._cards`.

# How special methods are used
> The first thing to know about special methods is that they are meant to be called by the Python interpreter, and not by you. 

When we write `len(obj)` and `obj` is a user-defined class, then Python calls the `__len__` method you implemented. But the interpreter takes a shortcut when dealing for built-in types like list, str, bytearray, or extensions like the NumPy arrays. If `obj` is one of the built-ins then `len(obj)` retrieves the value of `ob_size` from the `PyVarObject` class instance which is used to represent the built-in types in C internally.

More often than not, the special method call is implicit. For example, the statement `for i in x:` actually causes the invocation of `iter(x)`, which in turn may call `x.__iter__()` if that is available, or use `x.__getitem__()`, as in the `FrenchDeck` example.

> You should be implementing special methods more often than invoking them explicitly
If you need to invoke a special method, it is usually better to call the related built-in function (e.g., `len`, `iter`, `str`, etc.). These built-ins call the corresponding special method.

## Emulation numeric types
Several special methods allow user objects to respond to operators such as `+`. To start with we will implement a class to represent two-dimensional vectors—that is, Euclidean vectors like those used in math and physics and perform operations like vector addition, scalar multiplication, absolute value of vector etc.

Below is the `Vector` class implementation with the above operations and implements several built-in functions.
```python
import math

class Vector:

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        return f'Vector({self.x!r}, {self.y!r})'

    def __abs__(self):
        return math.hypot(self.x, self.y)

    def __bool__(self):
        return bool(abs(self))

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
```

`Vector` class implements two operators: `+` and `*`, to show basic usage of `__add__` and `__mul__`

```python
>>> v1 = Vector(2, 4)
>>> v2 = Vector(2, 1)
>>> v1 + v2
Vector(4, 5)
```

## String Representation

The `__repr__` special method is called by the `repr` built-in to get the string representation of the object for inspection. Without a custom `__repr__`, Python’s console would display a `Vector` instance `<Vector object at 0x10e100070>`.

The string returned by `__repr__` should be unambiguous and, if possible, match the source code necessary to re-create the represented object. That is why our `Vector` representation looks like calling the constructor of the class (e.g., `Vector(3, 4)`)

The interactive console and debugger call `repr` on the results of the expressions evaluated, as does the `%r` placeholder in classic formatting with the `%` operator, and the `!r` conversion field in the f-strings.

## Boolean value of a custom type

> Although Python has a bool type, it accepts any object in a Boolean context
To determine whether a value x is _truthy_ or _falsy_, Python applies `bool(x)`, which returns either `True` or `False`.

By default, instances of user-defined classes are considered truthy, unless either `__bool__` or `__len__` is implemented. Basically, `bool(x)` calls `x.__bool__()` and uses the result. If `__bool__` is not implemented, Python tries to invoke `x.__len__()`, and if that returns zero, `bool` returns `False`. Otherwise `bool` returns `True`.

Our implementation of `__bool__` is conceptually simple: it returns `False` if the magnitude of the vector is zero, `True` otherwise.

## Collection API

Collection classes implement 3 essential APIs:
- Iterable to support for, unpacking, and other forms of iteration
- Sized to support the len built-in function
- Container to support the in operator

Three very important specializations of Collection are:
- Sequence, formalizing the interface of built-ins like list and str
- Mapping, implemented by dict, collections.defaultdict, etc.
- Set, the interface of the set and frozenset built-in types
