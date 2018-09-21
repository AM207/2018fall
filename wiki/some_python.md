---
title: Some Useful Python
shorttitle: some_python
notebook: some_python.ipynb
noline: 1
summary: ""
layout: wiki
---
{% assign links = site.data.wikilinks %}

## Contents
{:.no_toc}
*  
{: toc}

### Python Classes and instance variables

Classes allow us to define our own *types* in the python type system. 



```python
class ComplexClass():
    
    def __init__(self, a, b):
        self.real = a
        self.imaginary = b
```




```python
c1 = ComplexClass(1,2)
print(c1, c1.real)
```


    <__main__.ComplexClass object at 0x10eac2f98> 1


### Inheritance and Polymorphism

**Inheritance** is the idea that a "Cat" is-a "Animal" and a "Dog" is-a "Animal". "Animal"s make sounds, but Cats Meow and Dogs Bark. Inheritance makes sure that *methods not defined in a child are found and used from a parent*.



```python
class Animal():
    
    def __init__(self, name):
        self.name=name
        print("Name is", self.name)

    def make_sound(self):
        raise NotImplementedError
        
class Mouse(Animal):
    def __init__(self, name):
        self.animaltype="prey"
        super().__init__(name)
        print("Created %s as %s" % (self.name, self.animaltype))
        
    def make_sound(self):
        return "Squeak"
    
class Cat(Animal):
    
    def make_sound(self):
        return "Meow"
```




```python
a0 = Animal("Rahul")
print(a0.name)
a0.make_sound()
```


    Name is Rahul
    Rahul



    ---------------------------------------------------------------------------

    NotImplementedError                       Traceback (most recent call last)

    <ipython-input-46-18721729352b> in <module>()
          1 a0 = Animal("Rahul")
          2 print(a0.name)
    ----> 3 a0.make_sound()
    

    <ipython-input-45-a732efef2388> in make_sound(self)
          6 
          7     def make_sound(self):
    ----> 8         raise NotImplementedError
          9 
         10 class Mouse(Animal):


    NotImplementedError: 




```python
a1 = Mouse("Tom")
a1.make_sound()
```


    Name is Tom
    Created Tom as prey





    'Squeak'





```python
a2 = Cat("Jerry")
a2.make_sound()
```


    Name is Jerry





    'Meow'





```python
animals = [a1, a2]
for a in animals:
    print(a.name)
    print(isinstance(a, Animal))
    print(a.make_sound())
    print('--------')
```


    Tom
    True
    Squeak
    --------
    Jerry
    True
    Meow
    --------


The above examples show inheritance and polymorphism. But notice that we didnt actually need to set up the inheritance. We could have just defined 2 different classes and have them both `make_sound`, the same code would work. In java and C++ this is done more formally through Interfaces and  Abstract Base Classes respectively plus inheritance, but in Python this agreement to define `make_sound` is called "duck typing"



```python
#both implement the "Animal" Protocol, which consists of the one make_sound function
class Dog():
    
    def make_sound(self):
        return "Bark"
    
class Cat():
    
    def make_sound(self):
        return "Meow"  
    
a1 = Dog()
a2 = Cat()
animals = [a1, a2]
for a in animals:
    print(isinstance(a, Animal))
    print(a.make_sound())
```


    False
    Bark
    False
    Meow


## The Python Data Model

Duck typing is used throught python. Indeed its what enables the "Python Data Model" 

- All python classes implicitly inherit from the root **object** class.
- The Pythonic way, is to just document your interface and implement it. 
- This usage of common **interfaces** is pervasive in *dunder* functions to comprise the python data model.

####   `__repr__`  

The way printing works is that Python wants classes to implement a `__repr__` and a `__str__` method. It will use inheritance to give the built-in `object`s methods when these are not defined...but any class can define these. When an *instance* of such a class is interrogated with the `repr` or `str` function, then these underlying methods are called.

We'll see `__repr__` here. If you define `__repr__` you have made an object sensibly printable...



```python
class Animal():
    
    def __init__(self, name):
        self.name=name
        
    def __repr__(self):
        class_name = type(self).__name__
        return "Da %s(name=%r)" % (class_name, self.name)
```




```python
r = Animal("Rahul")
r
```





    Da Animal(name='Rahul')



## Building out our class: instances and classmethods



```python
class ComplexClass():
    def __init__(self, a, b):
        self.real = a
        self.imaginary = b
        
    @classmethod
    def make_complex(cls, a, b):
        return cls(a, b)
        
    def __repr__(self):
        class_name = type(self).__name__
        return "%s(real=%r, imaginary=%r)" % (class_name, self.real, self.imaginary)
        
    def __eq__(self, other):
        return (self.real == other.real) and (self.imaginary == other.imaginary)
```




```python
c1 = ComplexClass(1,2)
c1
```





    ComplexClass(real=1, imaginary=2)





```python
c2 = ComplexClass.make_complex(1,2)
c2
```





    ComplexClass(real=1, imaginary=2)





```python
c1 == c2
```





    True



### Instance Variables shadow class variables



```python
class Demo():
    classvar=1
      
ademo = Demo()
print(Demo.classvar, ademo.classvar)
ademo.classvar=2 #different from the classvar above
print(Demo.classvar, ademo.classvar)
```


    1 1
    1 2


## Sequences and their Abstractions

#### What is a sequence?

Consider the notion of **Abstract Data Types**. 

The idea there is that one data type might be implemented in terms of another, or some underlying code, not even in python. 

As long as the interface and contract presented to the user is solid, we can change the implementation underlying it. 


The **dunder methods** in python are used towards this purpose. 

In python a sequence is something that follows the sequence protocol. An example of this is a python list. 

This entails defining the `__len__` and `__getitem__` methods. 



```python
alist=[1,2,3,4]
len(alist)#calls alist.__len__
```





    4





```python
alist[2]#calls alist.__getitem__(2)
```





    3



To see this lets create a dummy sequence which shows us what happens. This sequence does not create any storage, it just implements the protocol



```python
class DummySeq:
    
    def __len__(self):
        return 42
    
    def __getitem__(self, index):
        return index
```




```python
d = DummySeq()
len(d)
```





    42





```python
d[5]
```





    5



What about slicing?



```python
d[67:98]
```





    slice(67, 98, None)



Slicing creates a `slice object` for us of the form `slice(start, stop, step)` and then python calls `seq.__getitem__(slice(start, stop, step))`.

As sequence writers, our job is to interpret these in `__getitem__`. Here is a more realistic example.



```python
#taken from Fluent Python
import numbers, reprlib

class NotSoDummySeq:    
    def __init__(self, iterator):
        self._storage=list(iterator)
        
    def __repr__(self):
        components = reprlib.repr(self._storage)
        components = components[components.find('['):]
        return 'NotSoDummySeq({})'.format(components)
    
    def __len__(self):
        return len(self._storage)
    
    def __getitem__(self, index):
        cls = type(self)
        if isinstance(index, slice):
            return cls(self._storage[index])
        elif isinstance(index, numbers.Integral): 
            return self._storage[index]
        else:
            msg = '{cls.__name__} indices must be integers' 
            raise TypeError(msg.format(cls=cls))

```




```python
d2 = NotSoDummySeq(range(10))
len(d2)
```





    10





```python
d2
```





    NotSoDummySeq([0, 1, 2, 3, 4, 5, ...])





```python
d[4]
```





    4





```python
d2[2:4]
```





    NotSoDummySeq([2, 3])





```python
d2[1,4]
```



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-12-ae2b261447b5> in <module>()
    ----> 1 d2[1,4]
    

    <ipython-input-7-bae9aa90bd14> in __getitem__(self, index)
         22         else:
         23             msg = '{cls.__name__} indices must be integers'
    ---> 24             raise TypeError(msg.format(cls=cls))
    

    TypeError: NotSoDummySeq indices must be integers


## From positions in an array to Iterators

The salient points of this abstraction are:

- the notion of a `next` abstracting away the actual gymnastics of where to go next in a storage system
- the notion of a `first` to a `last` that `next` takes us on a journey from and to respectively

### Iterators and Iterables in python

Just as a sequence is something implementing `__getitem__` and `__len__`, an **Iterable** is something implementing `__iter__`. 

`__len__` is not needed and indeed may not make sense. 

The following example is taken from Fluent Python



```python
import reprlib
class Sentence:
    def __init__(self, text): 
        self.text = text
        self.words = text.split()
        
    def __getitem__(self, index):
        return self.words[index] 
    
    def __len__(self):
        #completes sequence protocol, but not needed for iterable
        return len(self.words) 
    
    def __repr__(self):
        return 'Sentence(%s)' % reprlib.repr(self.text)
```




```python
#sequence'
a= Sentence("Mary had a little lamb whose fleece was white as snow.")
len(a), a[3], a
```





    (11, 'little', Sentence('Mary had a l...hite as snow.'))





```python
a[3:5] # why did this make sense?
```





    ['little', 'lamb']





```python
min(a), max(a)
```





    ('Mary', 'whose')





```python
list(a)
```





    ['Mary',
     'had',
     'a',
     'little',
     'lamb',
     'whose',
     'fleece',
     'was',
     'white',
     'as',
     'snow.']



To iterate over an object x, python automatically calls `iter(x)`. An **iterable** is something which, when `iter` is called on it, returns an **iterator**.

(1) if `__iter__` is defined, calls that to implement an iterator.

(2) if not  `__getitem__` starting from index 0

(3) otherwise raise TypeError

Any Python sequence is iterable because they implement `__getitem__`. The standard sequences also implement `__iter__`; for future proofing you should too because  (2) might be deprecated in a future version of python.

This:



```python
for i in a:
    print(i)
```


    Mary
    had
    a
    little
    lamb
    whose
    fleece
    was
    white
    as
    snow.


is implemented something like this:



```python
it = iter(a)
while True:
    try:
        nextval = next(it)
        print(nextval)
    except StopIteration:
        del it
        break
```


    Mary
    had
    a
    little
    lamb
    whose
    fleece
    was
    white
    as
    snow.


`it` is an iterator. 

An iterator defines both `__iter__` and a `__next__` (the first one is only required to make sure an *iterator* IS an *iterable*). 

Calling `next` on an iterator will trigger the calling of `__next__`.



```python
it = iter(a)
next(it), next(it), next(it)
```





    ('Mary', 'had', 'a')



So now we can completely abstract away a sequence in favor an iterable (ie we dont need to support indexing anymore). From Fluent:



```python
class SentenceIterator:
    def __init__(self, words): 
        self.words = words 
        self.index = 0
        
    def __next__(self): 
        try:
            word = self.words[self.index] 
        except IndexError:
            raise StopIteration() 
        self.index += 1
        return word 

    def __iter__(self):
        return self
    
class Sentence:#an iterable
    def __init__(self, text): 
        self.text = text
        self.words = text.split()
        
    def __iter__(self):
        return SentenceIterator(self.words)
    
    def __repr__(self):
        return 'Sentence(%s)' % reprlib.repr(self.text)
```




```python
s2 = Sentence("While we could have implemented `__next__` in Sentence itself, making it an iterator, we will run into the problem of exhausting an iterator'.")
```




```python
len(s2)
```



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-23-4b382c72e9ab> in <module>()
    ----> 1 len(s2)
    

    TypeError: object of type 'Sentence' has no len()




```python
for i in s2:
    print(i)
```


    While
    we
    could
    have
    implemented
    `__next__`
    in
    Sentence
    itself,
    making
    it
    an
    iterator,
    we
    will
    run
    into
    the
    problem
    of
    exhausting
    an
    iterator'.




```python
s2it=iter(s2)
print(next(s2it))
s2it2=iter(s2)
next(s2it),next(s2it2)
```


    While





    ('we', 'While')



While we could have implemented `__next__` in Sentence itself, making it an iterator, we will run into the problem of "exhausting an iterator". 

The iterator above keeps state in `self.index` and we must be able to start anew by creating a new instance if we want to re-iterate. Thus the `__iter__` in the iterable, simply returns the `SentenceIterator`.



```python
min(s2), max(s2)
```





    ('Sentence', 'will')



Note that min and max will work even though we now DO NOT satisfy the sequence protocol, but rather the ITERABLE protocol, as its a pairwise comparison, which can be handled via iteration. The take home message is that in programming with these iterators, these generlization of pointers, we dont need either the length or indexing to work to implement many algorithms: we have abstracted these away.

## Generators

EVERY collection in Python is iterable.

Lets pause to let that sink in.

We have already seen iterators are used to make for loops. They are also used tomake other collections

to loop over a file line by line from disk
in the making of list, dict, and set comprehensions
in unpacking tuples
in parameter unpacking in function calls (*args syntax)
An iterator defines both __iter__ and a __next__ (the first one is only required to make sure an iterator IS an iterable).

SO FAR: Iterator: retrieves items from a collection. The collection must implement __iter__.

### Yield and generators

A generator function looks like a normal function, but instead of returning values, it yields them. The syntax is (unfortunately) the same.

Unfortunate, as a generator is a different beast. When the function runs, it creates a generator.

The generator is an iterator.. It gets an internal implementation of __next__ and __iter__, almost magically.



```python
def gen123():
    print("Hi")
    yield 1
    print("Little")
    yield 2
    print("Baby")
    yield 3
```




```python
print(gen123, type(gen123))
g = gen123()
type(g)
```


    <function gen123 at 0x10eabaf28> <class 'function'>





    generator





```python
#a generator is an iterator
g.__iter__
```





    <method-wrapper '__iter__' of generator object at 0x10eab2728>





```python
g.__next__
```





    <method-wrapper '__next__' of generator object at 0x10eab2728>





```python
next(g),next(g), next(g)
```


    Hi
    Little
    Baby





    (1, 2, 3)



When next is called on it, the function goes until the first yield. The function body is now suspended and the value in the yield is then passed to the calling scope as the outcome of the next.

When next is called again, it gets __next__ called again (implicitly) in the generator, and the next value is yielded..., and so on... ...until we reach the end of the function, the return of which creates a StopIteration in next.

Any Python function that has the yield keyword in its body is a generator function.



```python
for i in gen123():
    print(i)
```


    Hi
    1
    Little
    2
    Baby
    3


Use the language: "a generator yields or produces values"



```python
class Sentence:#an iterable
    def __init__(self, text): 
        self.text = text
        self.words = text.split()
        
    def __iter__(self):#one could also return iter(self.words)
        for w in self.words:#note this is implicitly making an iter from the list
            yield w
    
    def __repr__(self):
        return 'Sentence(%s)' % reprlib.repr(self.text)
a=Sentence("Mary had a little lamb whose fleece was white as snow.")
```




```python
for w in a:
    print(w)
```


    Mary
    had
    a
    little
    lamb
    whose
    fleece
    was
    white
    as
    snow.


## Lazy processing

Upto now, it might just seem that we have just represented existing sequences in a different fashion. But notice above, with the use of yield, that we do not have to define the entire sequence ahead of time. We see it in the generation of infinite sequences, where there is no data per se!

So, because of generators, we can go from fetching items from a collection to "generate"ing iteration over arbitrary, possibly infinite series...



```python
def fibonacci(): 
    i,j=0,1 
    while True: 
        yield j
        i,j=j,i+j
```




```python
f = fibonacci()
for i in range(10):
    print(next(f))
```


    1
    1
    2
    3
    5
    8
    13
    21
    34
    55


### Generator Expressions of data sequences.

There is an even simpler way: use a generator expression, which is just a lazy version of a list comprehension. (itrs really just sugar for a generator function, but its a nice bit of sugar)

Which syntax to choose?

Write a generator function if the code takes more than 2 lines.

Some syntax that might trip you up: double brackets are not necessary



```python
(i*i for i in range(5))
```





    <generator object <genexpr> at 0x10eab2bf8>





```python
list((i*i for i in range(5)))
```





    [0, 1, 4, 9, 16]





```python
list(i*i for i in range(5))
```





    [0, 1, 4, 9, 16]


