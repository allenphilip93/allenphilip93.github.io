---
title: Python Data Class Builders
date: 2024-05-23 09:46:00 +0530
categories: [Book Summary, Fluent Python]
tags: [Notes, Python]
math: false
pin: false
image:
  path: https://m.media-amazon.com/images/I/71RiBEY6mWL._AC_UF1000,1000_QL80_.jpg
  lqip: data:image/webp;base64,UklGRpoAAABXRUJQVlA4WAoAAAAQAAAADwAABwAAQUxQSDIAAAARL0AmbZurmr57yyIiqE8oiG0bejIYEQTgqiDA9vqnsUSI6H+oAERp2HZ65qP/VIAWAFZQOCBCAAAA8AEAnQEqEAAIAAVAfCWkAALp8sF8rgRgAP7o9FDvMCkMde9PK7euH5M1m6VWoDXf2FkP3BqV0ZYbO6NA/VFIAAAA
  alt: Fluent Python by Luciano Ramalho
---

## Reference

[Fluent Python Chapter-5](https://learning.oreilly.com/library/view/fluent-python-2nd/9781492056348/ch05.html)

## Overview

Python offers a few ways to build a simple class that is just a collection of fields, with little or no extra functionality. That pattern is known as a “data class”—and `dataclasses` is one of the packages that supports this pattern. This chapter covers three different class builders that you may use as shortcuts to write data classes:

* `collections.namedtuple`
	* The simplest way—available since Python 2.6.
- `typing.NamedTuple`
	- An alternative that requires type hints on the fields—since Python 3.5, with `class` syntax added in 3.6.
- `@dataclasses.dataclass`
	- A class decorator that allows more customization than previous alternatives, adding lots of options and potential complexity—since Python 3.7.

Consider a simple class to represent a geographic coordinate pair.

```python
class Coordinate:
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon
```

Though this works you'll quickly realize that this is not really great:
-  `__init__` option quickly becomes old pretty fast if number of args to constructor increases
- `__repr__` inherited from `object` class is not useful. If we print the object of this class, it'll just show the memory ref
- `==` is meaningless without implementing `_eq_` method inherited from `object` class

> **NOTE** 
> The data class builders mentioned earlier take care of all these caveats. Moreover, none of the class builders discussed here depend on inheritance to do their work. Both `collections.namedtuple` and `typing.NamedTuple` build classes that are `tuple` subclasses! Each of them uses different metaprogramming techniques to inject methods and data attributes into the class under construction.

The class `Coordinate` can be implemented using `collections.namedtuple` as follows:
```python
>>> from collections import namedtuple
>>> Coordinate = namedtuple('Coordinate', 'lat lon')
>>> issubclass(Coordinate, tuple)
True
>>> moscow = Coordinate(55.756, 37.617)
>>> moscow
Coordinate(lat=55.756, lon=37.617)  
>>> moscow == Coordinate(lat=55.756, lon=37.617)  
True
```

`typing.NamedTuple` extends the same with type annotation to each field.
```python
>>> import typing
>>> Coordinate = typing.NamedTuple('Coordinate',
...     [('lat', float), ('lon', float)])
>>> issubclass(Coordinate, tuple)
True
>>> typing.get_type_hints(Coordinate)
{'lat': <class 'float'>, 'lon': <class 'float'>}

>>> # Alternatively
>>> Coordinate = typing.NamedTuple('Coordinate', lat=float, lon=float)
```

Since Python 3.6, `typing.NamedTuple` can also be used in a `class` statement as follows:
```python
from typing import NamedTuple

class Coordinate(NamedTuple):
    lat: float
    lon: float
    reference: str = 'WGS84'
```

> **WARN**
> Although `NamedTuple` appears in the `class` statement as a superclass, it’s actually not. `typing.NamedTuple` uses the advanced functionality of a metaclass to customize the creation of the user’s class.

Lastly we can define the class using `@dataclass` decorator
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Coordinate:
    lat: float
    lon: float
    reference: str = 'WGS84'
```

The `@dataclass` decorator does not depend on inheritance or a metaclass, so it should not interfere with your own use of these mechanisms. And this case, the class `Coordinate` is a subclass of `object`!

Each option has it's own merits and limitations. For example, a key difference is that `collections.namedtuple` and `typing.NamedTuple` build `tuple` subclasses, therefore the instances are immutable. By default, `@dataclass` produces mutable classes which can be switched using the `frozen` attribute. 

Also if you need to build data classes on the fly, at runtime, you can use the default function call syntax of `collections.namedtuple`, which is likewise supported by `typing.NamedTuple`. The `dataclasses` module provides a `make_dataclass` function for the same purpose.

## Type Hints 101

Type hints—a.k.a. type annotations—are ways to declare the expected type of function arguments, return values, variables, and attributes. This adds the typing to a duck types language like Python to "kind of" have some ordinance of strong typing.

Why do we say "kind of"?! It's because **Python does not enforce type hints at runtime**! Type hints have no impact on runtime whatsoever. The example below runs without any errors:
```python
>>> import typing
>>> class Coordinate(typing.NamedTuple):
...     lat: float
...     lon: float
...
>>> trash = Coordinate('Ni!', None)
>>> print(trash)
Coordinate(lat='Ni!', lon=None)   # <----- Runs without any error!!
```

So what's the use of Type Hints then? The type hints are intended primarily to support third-party type checkers like `pypy` that check the Python source code **at rest**!

Also while it is true that type hints have no effect at runtime, but at import time—when a module is loaded—Python does read them to build the `__annotations__` dictionary that `typing.NamedTuple` and `@dataclass` then use to enhance the class.

Using `@dataclass`, if the type hints are specified they are treated as an instance variable but if they're not it'll be treated as a class variable.

```python
@dataclass
class DemoDataClass:
    a: int            # Instance variable
    b: float = 1.1    # Instance variable
    c = 'spam'        # Class variable (shared across all instances)
    
>>> DemoDataClass.__annotations__
{'a': <class 'int'>, 'b': <class 'float'>}
>>> DemoDataClass.__doc__
'DemoDataClass(a: int, b: float = 1.1)'
>>> DemoDataClass.a   # Exists only in the instance variable
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: type object 'DemoDataClass' has no attribute 'a'
>>> DemoDataClass.b   # Just stores the default of the variable
1.1
>>> DemoDataClass.c   # Stores the class variable itself
'spam'
```

The behaviour is the same using `collections.NamedTuple` as well.

```python
class DemoNTClass(typing.NamedTuple):
    a: int            # Instance variable
    b: float = 1.1    # Instance variable
    c = 'spam'        # Class variable (shared across all instances)

>>> from demo_nt import DemoNTClass
>>> DemoNTClass.__annotations__
{'a': <class 'int'>, 'b': <class 'float'>}
>>> DemoNTClass.a     # Stores how to access the instance variable
<_collections._tuplegetter object at 0x101f0f940>
>>> DemoNTClass.b     # Stores how to access the instance variable
<_collections._tuplegetter object at 0x101f0f8b0>
>>> DemoNTClass.c     # Class variable
'spam'
```

## More about `@dataclass`

We’ve only seen simple examples of `@dataclass` use so far. The decorator accepts several keyword arguments.
```python
@dataclass(*, init=True, repr=True, eq=True, order=False,
              unsafe_hash=False, frozen=False)
```

### Field Options

We’ve already seen the most basic field option: providing (or not) a default value with the type hint. Python does not allow parameters without defaults after parameters with defaults, therefore after you declare a field with a default value, all remaining fields must also have default values.

Mutable default values are a common source of bugs. In function definitions, a mutable default value is easily corrupted when one invocation of the function mutates the default, changing the behavior of further invocations. To prevent bugs, `@dataclass` rejects such class definition.
```python
@dataclass
class ClubMember:
    name: str
    guests: list = []

$ python example.py
Traceback `(`most recent call last`)`:
  File `"club_wrong.py"`, line `4`, `in` <module>
    class ClubMember:
  ...several lines omitted...
ValueError: mutable default <class `'list'`> `for` field guests is not allowed:
use default_factory
```

But this can be easily fixed as follows:
```python
from dataclasses import dataclass, field

@dataclass
class ClubMember:
    name: str
    guests: list = field(default_factory=list)
```

The `default_factory` parameter lets you provide a function, class, or any other callable, which will be invoked with zero arguments to build a default value each time an instance of the data class is created. This way, each instance of `ClubMember` will have its own `list`—instead of all instances sharing the same `list` from the class, which is rarely what we want and is often a bug.

### Post-init Processing

Sometimes you may need to do more than that to initialize the instance. If that’s the case, you can provide a `__post_init__` method. When that method exists, `@dataclass` will add code to the generated `__init__` to call `__post_init__` as the last step.

```python
from dataclasses import dataclass
from club import ClubMember

@dataclass
class HackerClubMember(ClubMember):                         
    all_handles = set()                                     
    handle: str = ''                                        

    def __post_init__(self):
	    # Insert post init operations like validation
        pass
```

### Typed Class Attributes

Now consider the class below which I'll be running with `mypy`

```python
@dataclass(frozen=True)
class Coordinate:
    lat: float         # Instance variable
    lon: float         # Instance variable
    regions = set()    # Class variable
```

Running the above code with `mypy` will throw an error since `regions` is not provided with a type hint. Now let's look at a "possible" fix for the same.

```python
from dataclasses import dataclass, field
from typing import Set

@dataclass(frozen=True)
class Coordinate:
    lat: float
    lon: float
    regions: Set[str] = field(default_factory=set)
```

Now the code will be executed but there is an issue. `regions` will be treated like an instance variable now!

This is where `ClassVar` comes in, which can be used as a type hint to define a class variable. `ClassVar` indicates that a particular attribute is a class variable, not an instance variable. These variables are shared among all instances of the class and are not included in the `__init__` method generated by the `@dataclass`.

```python
from dataclasses import dataclass
from typing import ClassVar, Set

@dataclass(frozen=True)
class Coordinate:
    lat: float
    lon: float
    regions: ClassVar[Set[str]] = set()
```

### Initialisation Variables

Sometimes you may need to pass arguments to `__init__` that are not instance fields. Such arguments are called _init-only variables_. To declare an argument like that, the `dataclasses` module provides the pseudotype `InitVar`, which uses the same syntax of `typing.ClassVar`. 

`InitVar` is used for attributes that are only part of the initialization process but are not stored as instance variables. They can be used to perform some operations during initialization but will not be part of the instance state.

```python
@dataclass
class C:
    i: int
    j: int = None
    database: InitVar[DatabaseType] = None

    def __post_init__(self, database):
        if self.j is None and database is not None:
            self.j = database.lookup('j')

c = C(10, database=my_database)
```

## Data Class as a Code Smell

Whether you implement a data class by writing all the code yourself or leveraging one of the class builders, be aware that it may signal a problem in your design.

> Martin Fowler
> "*These are classes that have fields, getting and setting methods for fields, and nothing else. Such classes are dumb data holders and are often being manipulated in far too much detail by other classes.*"

The main idea of object-oriented programming is to place behavior and data together in the same code unit: a class. If a class is widely used but has no significant behavior of its own, it’s possible that code dealing with its instances is scattered (and even duplicated) in methods and functions throughout the system—a recipe for maintenance headaches.

Taking that into account, there are a couple of common scenarios where it makes sense to have a data class with little or no behavior.

### Data Class as Scaffolding

In this scenario, the data class is an initial, simplistic implementation of a class to jump-start a new project or module. With time, the class should get its own methods, instead of relying on methods of other classes to operate on its instances. Scaffolding is temporary; eventually your custom class may become fully independent from the builder you used to start it.

### Data Class as Intermediate Representation

A data class can be useful to build records about to be exported to JSON or some other interchange format, or to hold data that was just imported, crossing some system boundary. Python’s data class builders all provide a method or function to convert an instance to a plain `dict`, and you can always invoke the constructor with a `dict` used as keyword arguments expanded with `**`. Such a `dict` is very close to a JSON record.

In this scenario, the data class instances should be handled as immutable objects—even if the fields are mutable, you should not change them while they are in this intermediate form. If you do, you’re losing the key benefit of having data and behavior close together.
