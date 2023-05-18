# What does it mean to learn?

## Generalization

Take for example a student which has to answer some questions on a test. Generalizing means that, if he saw some specific questions with the associated (corrected) answers, he is able to treat correctly new, unseen (related) questions.

But what does this mean in terms of Machine Learning?

## Induction

Take a recommender system. This system has to predict how much (on a scale) a student likes a course. The induction framework does this:

1. It looks at previous years' **examples** (course student pairs) taken from the so called **training set** and **induces** a function f that will map new examples to a **predicted** rating
2. It **evaluates** the induced function against the **test set**

![induction framework](../images/b9b18ab2b16b29486abb53e8c5258f47be06603bd42e1794851a237e531a5efe.png)

Step 1 is contained in the red box, aka the learning algorithm. 

![Learning algorithm](../images/05b31c1af1208989ecae2cc06d88a187151eaa00f02caef5dc16af2b94e15b5b.png) Hi  




Hi
