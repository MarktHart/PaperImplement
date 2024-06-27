This repo is a work in "progress" and probably has plenty of bugs.

# Goal
The goal of this repo is to implement some ML papers with minimal dependencies. Each implementation highly favors (my definition of) readability over traditional software extendability, trying to clearly expose all complex parts to readers.

## Why?
I think code is often clearer than documentation, it's the language that exactly defines what happens. Given the speed of ML breakthroughs, I do believe this type of code is easier to adopt for (experienced) users. If not, I hope it's easier to understand papers when written in this way.


## Coding guidelines/rules
* The rules are rules of thumb at best and meant to be broken.
* My definition of readability over speed or supporting all soft/hardware.
* No inheritance.
* No comments.
* No imports.
* No dependencies but torch.
* Seperate model net layout from other parts of the code (i.e. loss calculation or flops estimation in a seperate file).

# Other links
Shoutout to Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT), it inspired me to brush the dust of this repo.
