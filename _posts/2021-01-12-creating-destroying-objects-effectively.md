---
title: Effective Java | Creating & Destroying objects
date: 2020-12-21 15:05:00 +0530
categories: [Java]
tags: [Notes, Learning]
math: true
pin: false
---

# Core Principle

When writing any piece of code, clarity and simplicity are of paramount importance. Code should be reused rather than copied. The dependencies should be kept to a minimum. Errors should be detected as soon as they're made and preferably at compile time.

# Static factory methods over constructors

A class can provide a public static factory method, which is simply a static method that returns an instance of the class.

    public static Boolean valueOf(boolean b) {
    	return b ? Boolean.TRUE : Boolean.FALSE;
    }

## Advantages

Unlike constructors, static factory methods have names. For example, *BigInteger(int, int, Random)* was used to return a *BigInteger* that is probably prime. This would have been better represented as a static factory method name *BigInteger.probablePrime *which was introduced in Java 4.

Unlike constructors, they're not required to create a **new **object each time they're invoked. Look at the above piece of code, *valueOf(bool)* doesn't create a new object ever. This is useful in cases where we would like to reuse objects and reduce unnecessary duplicate object creations.

Unlike constructors, they can return an object of any subtype of their return type. API can return objects without making their classes public which leads to *interface-based frameworks.*

Another advantage is that the class of the returned object can vary from call to call as a function of the input parameters. Similarly the class of the returned object need not exist when the class containing the method is written (*bridge pattern*).

    public <? super List> getInstance(int type) {
    	if (type == 0)
        	return new ArrayList();
        else
        	return new LinkedList();
    }

## Limitations

The main limitation with providing only static factory methods is that classes without  public or protected constructors can't be subclasses!

A second shortcoming is that static methods are hard for programmers to spot since they don't stand out in the documentation.

## Naming Conventions

- *from - *A type-conversion method
```java
    Date d = Date.from(instant);
```
- *of - *An aggregation method
```java
    Set<Rank> faceCards = EnumSet.of(JACK, QUEEN, KING);
```
- *valueOf - *Verbore alternative to *from & of*
- *instance or getInstance - *Returns an instance as described
```java
    LightSaber luke = LightSaber.getInstance(options)
```
- *create or newInstance - *Similar to the above but **gurantees** a new instance
```java
    Object newArray = Array.newInstance(obj, length);
```
- *getType - *Similar to *getInstance* but used if factory method is in a different class
```java
    FileStore fs = Files.getFileStore(path);
```
- *newType - *Similar to create*Instance* but used if factory method is in a different class
```java
    BufferedReader br = Files.newBufferedReader(path);
```
- *type - *A concise alternative to the above two.
```java
    List<Complaint> library = Collections.list(oldBooks);
```
# Builder for creating objects

Static factories and constructors share a limitation, they do not scale well to large number of optional parameters. Normally this is solved in two ways:

- **Telescoping constructor pattern** - Where we keep overloading the constructors and call the one with fewer parameters from the one with the larger set and set the difference. Though this does the job, it offers very little readability and difficult to maintain.

    public Car(int numwheels) {
    	this.wheels = numwheels
    }
    
    public Car(int numwheels, boolean ismanual) {
    	this(numwheels);
        this.ismanual = ismanual;
    }
    
    public Car(int numwheels, boolean ismanual, String fuel) {
    	this(numwheels, ismanual, fuel);
        ...
    }

- **JavaBeans pattern** - We use the default constructor to create an instance and set the parameters using the setter methods. But this bears the risk of the object being in an inconsistent state partway through construction. Also this pattern prevents the class from being immutable.

## Builder Pattern

Instead of making the desired object directly, the client calls a constructor (or static factory) with all of the required parameters and gets a builder object. The client calls the setter-like method on the builder object to set each optional parameter of interest. Finally, the client call the parameterless *build()* method to create an instance of the object.

This combines the safety of the telescoping pattern with the readability of the JavaBeans pattern as illustrated in the example below.
```java
    public class Car {
    	private final int kms;
        private final int wheels;
        private final boolean ismanual;
        private final String manufacturer;
        
        public static class Builder {
        	// Required parameters
        	private final int wheels;
            // Optional parameters
            private int kms = 0;
            private boolean ismanual = true;
            private String manufacturer = "NA";
            
            public Builder(int wheels) {
            	this.wheels = wheels;
            }
            
            // setters for optional parameters which return the builder instance //
            
            public Car build() {
            	return new Car(this);
            }
        }
        
        private Car(Builder b) {
        	kms = b.kms;
            wheels = b.wheels;
            ismanual = b.ismanual;
            manufacturer = b.manufacturer;
        }
    }

    Car wrv = new Car.Builder(4).setIsmanual(true).setManufacturer("Honda").build();
    Car pickupTruck = new Car.Builder(6).setKms(12312).build();
```
Builder pattern is well suited to class hierarchies for example.
```java
    public abstract class Vehicle {
        final int numwheels;
        
        abstract static class Builder<T extends Builder<T>> {
        	int numwheels = 4;
            public T setNumwheels(int n) {
            	this.numwheels = n;
            }
            
            abstract Vehicle build();
        }
        
        Vehicle(Builder<?> builder) {
        	this.numwheels = builder.getNumwheels();
        }
    }

    public class Bike {
    	private boolean abs;
        private boolean topspeed;
        
        public static class Builder extends Vehicle.Builder<Builder> {
        	boolean abs = false;
            int topspeed;
            
            // getters & setters //
            
            public Builder(int topspeed) {
            	super(2);
            	this.topspeed = topspeed;
            }
            
            public Bike build() {
            	return new Bike(this);
            }
        }
        
        private Bike(Builder builder) {
        	super(builder);
            this.abs = builder.abs;
            this.topspeed = builder.topspeed;
        }
    }

    Bike handicapBike = new Bike.Builder(60).setNumwheels(4).build();
    Bike kawasaki = new Bike.Builder(340).setAbs(true).build();
```
It is important to note that builder pattern is more verbose than the telescoping pattern and should be used only if there are enough parameters to make it worthwhile.

# Enforcing the Singleton pattern

A singleton is a class that can be instantiated only once. At times singletons can prove to be difficult to write unit test cases for. If a singleton is called from another class it may be difficult to mock it in certain cases.

There are a few ways to enforce a singleton pattern:
```java
    public class Singleton {
    	public static final Singleton INSTANCE = new Singleton();
        private Singleton() {}
    }
```
The constructor is made private and is called only once to initialize the public INSTANCE member. This technique is also called eager initialization since the instance is made available at class loading. Now let's look at another way:
```java
    public class Singleton {
    	private static final Singleton INSTANCE = new Singleton();
        private static Singleton getInstance() {
        	return INSTANCE;
        }
        private Singleton() {}
    }
```
One advantage with using the static factory approach is that we can change the behavior of the class from being a Singleton without changing the API. This also offers more flexibility on how we want the Singleton to be implemented. Lastly a static factory can be used as a method reference to a Supplier.

In order to make a Singleton class serializable, it not sufficient to not just implement Serializable but also declare all instance fields transient and provide a readResolve method. Otherwise, each time a new instance will be created.
```java
    private Object readResolve() {
    	return INSTANCE;
    }
```
There is one more way to implement a Singleton which is the preferred approach:
```java
    public enum Singleton {
    	INSTANCE;
        public void doSomething() {..}
    }
```
This approach is more concise, provides Serialization protection for free and prevents any unwanted multiple instantiation even in the face of reflection! A single-element enum is often the best way to implement a singleton.

Another implementation take the approach of lazy initialization wherein we don't create the singleton instance until it's accessed for the first time. This is not recommended and it would complicate the implementation with no measurable performance improvement.
```java
    public class Singleton {
    	private static final Singleton INSTANCE;
        // Alternatively the method can be made synchronized as well
        public static Singleton getInstance() {
        	if (INSTANCE == null) {
            	synchronized (Singleton.class) {          
                    INSTANCE = new Singleton();
                }
            }
            return INSTANCE;
        }
        private Singleton() {}
    }
```
The preferred way to implement a Singleton where lazy initiation is absolutely required is as follows. Though this is still not safe from reflection/serialization attacks, its thread-safe without using synchronized since the inner class will be loaded into memory only when someone calls the *getInstance *method.
```java
    public class Singleton {
    	private Singleton() {}
        private static class SingletonHelper {
        	public static final Singleton INSTANCE = new Singleton();
        }
        public static Singleton getInstance() {
        	return SingletonHelper.INSTANCE;
        }
    }
```
# Noninstantiablity with a private constructor

Utility classes with only static methods are not designed to be instantiated. A default constructor is provided to every class by the compiler so a class can be made noninstantiable by including a private constructor.
```java
    public class UtilityClass {
    	private UtilityClass() {}
        // other static methods
    }
```
As a side effect, this prevents the class from being subclasses.

# Dependency Injection over hardwiring resources

Static utility classes and singletons are inappropriate for classes whose behavior is parameterized by an underlying resource. In such cases, a simple approach is to pass the resource into the constructor when creating the new instance. This is one form of *dependency injection*.

A useful variant of the pattern is to pass a *resource factory* to the constructor. A factory object is an object that can be called repeatedly to create instances of a type like in the Factory Method pattern.

Dependency injection greatly improved flexibility and testability but it can clutter up large projects. This clutter can be eliminated by using dependency injection frameworks like Spring, Guice or Dagger.

# Avoiding unnecessary object creation
```java
    String s = new String("BAD"); // Not good practice
```
The argument to the String constructor is a String in itself which is functionally identical. So why create a new instance and eat up memory. The correction option would be:
```java
    String s = "GOOD";
```
Often we can avoid creating unnecessary objects by using static factory methods like *Boolean.valueOf(String)* compared to the constructor* Boolean(String)* which is in fact deprecated in Java 9. The constructor must create a new object when it's called whereas the static factory methods have no such obligation.

Some object creations are much more expensive that others. So in cases where such objects are being created repeatedly, its advisable to cache the objects and reuse wherever possible.

Another unnecessary object creation that's easy to miss is autoboxing. Always prefer primitives to boxed primitives and be wary of unintentional autoboxing.
```java
    // Very slow because of autoboxing
    private static long sum() {
    	Long sum = 0L; // Should be declared as long
        for (long i=0; i <= Integer.MAX_VALUE; i++)
        	sum += i; // Autoboxing here - creates about 2^31 Long instances for i
        return sum;
    }
```
Lastly we wary of creating and maintaining your own object pools. Only implement such object pools when the objects are extremely heavyweight like in the case of database connections. Otherwise it just increases memory footprint and harms performance.

# Eliminate obsolete object references

Nulling out obsolete references ensures that if they're accessed by mistake it throws a null pointer exception. But the best way to eliminate an object reference is to let the variable that contained the reference fall out of scope. 

In general whenever a class maintains its own memory the programmer should be wary of memory leaks like the case of caches. Other common culprits include listeners and other callbacks.

# Avoid finalizers and cleaners

Finalizers are unpredictable, often dangerous and generally unnecessary. Cleaners are less dangerous than finalizers but still unpredictable, slow and generally unnecessary. Finalizers also open you up to finalizer attacks and is a security concern. In the case where you have to use it, you should never do anything time critical within them.

# Prefer try-with-resources to try-finally

Lost of resources like InputStream, Connection needs to be explicitly closed using the close method otherwise it leads to dire performance issues. Historically, try-finally was the best way to guarantee the same.
```java
    void readline(String path) {
    	BufferedReader br = new BufferedReader(new FileReader(path));
        try {
        	br.readline();
        } finally {
        	br.close();
        }
    } 
```
This approach becomes increasingly difficult when there are more resources. The best way to close resources as of Java 7 is as follows:
```java
    void readline(String path) {
        try (BufferedReader br = new BufferedReader(new FileReader(path));
        	 OutputStream os = new FileOutputStream(path);) {
         	String s = br.readline();
            out.write(s);
        }
    }
```
# References

- [Effective Java by Joshua Bloch](https://www.google.com/url?sa=t&amp;rct=j&amp;q=&amp;esrc=s&amp;source=web&amp;cd=&amp;cad=rja&amp;uact=8&amp;ved=2ahUKEwjol9i37d7tAhW0guYKHb3AD2UQFjABegQIARAC&amp;url=https%3A%2F%2Fwww.oreilly.com%2Flibrary%2Fview%2Feffective-java%2F9780134686097%2F&amp;usg=AOvVaw0_9_MCcKlk9FKgFO4yrZr2)
