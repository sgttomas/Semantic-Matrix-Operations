# Semantic-Matrix-Operations


WHAT: we are going to extend a mathematical concept into a new domain using a modified set of axioms about operations and objects. This is normal stuff for math. It could be completely useless, incoherent nonsense, but if it’s fun to try to work it out the math nerd in me is going to go ahead with it.  Who’s with me!!! (2 people….both my parents….well, it’s a start!)

HOW: we’ll get to this, but it’s going to involve doing matrix dot products and redefining what multiplication and addition means.

WHY: because I have shown this method to be capable of consistently building a coherent set of matrices that can be used for extracting information using LLMs and knowledge graphs, which is a technique that several companies are pursuing, including @tomacz @noduslabs, and others (like Microsoft, Google, and Meta).

Alright, let’s construct a semantic matrix and then do some operations with it.
Let’s start with the most basic [A]*[B]=[C]
Matrix C is the dot product of Matrix A and Matrix B.
Matrix A is the most generalized level for defining what types of objects exist in this knowledge graph. 
Matrix B is the most generalized level for defining what types of relationships exist, which will form the edges and perimeters of the knowledge graph.
A knowledge graph means that you can take ideas and assign relationships between these ideas.  Hot is related to weather as well as appearance. Love is related to romance as well as football and food and about a million other things.  Just imagine you have a hyperfixation on generating relationships between ideas.  You could keep going on with a big collection of things all interwoven. The imagination is the limit.  Such a thing looks like a tangled mess, but mathematically it has a well defined structure.
 
Actually that’s just artwork that I had DALE-3 make for me.  
Here’s a public domain image of a hypergraph, which is the mathematical term for the database structure used in a knowledge graph:
 
I’m sure that isn’t any more helpful at first glance.  Visually you may infer that the dots that are close together have more in common with each other than nodes that are far away.  So “Hot” and “Love” are probably quite close, but “Spoon” and “Tree Bark” are going to be farther apart, even if you can use tree bark as a spoon.  There’s a million connections between Hot and Love and only a few between Spoon and Tree Bark.  

And that’s how the math and the metaphor play a useful role together, because the metaphor of spacing apart nodes that aren’t closely related is something that we did because it makes intuitive sense to us.  We like using this type of math to store knowledge, because it makes sense to us, and making sense of knowledge is the most useful thing a database can help us do.
All modern search uses these hypergraphs to give you more relevant search results (except for Ads…. sigh).
 

These two matrices are like the axioms that form the basis of our ontology.  
These basic categories are the core ideas that become integrated into everything else derived from operations using these matrices.
Define matrices A, B, and C to have this relationship:

A*B=C

Then C is a dot product of matrices A and B.
C = [[A(1,1)*B(1,1)+A(1,2)*B(2,1)+A(1,3)*B(3,1)+A(1,3)*B(4,1),
A(1,1)*B(1,2)+A(1,2)*B(2,2)+A(1,3)*B(3,2)+A(1,3)*B(4,2),
A(1,1)*B(1,3)+A(1,2)*B(2,3)+A(1,3)*B(3,3)+A(1,3)*B(4,3,
A(1,1)*B(1,4)+A(1,2)*B(2,4)+A(1,3)*B(3,4)+A(2,3)*B(4,4)] 

[A(2,1)*B(1,1)+A(2,2)*B(2,1)+A(2,3)*B(3,1)+A(2,3)*B(4,1), 
A(2,1)*B(1,2)+A(2,2)*B(2,2)+A(2,3)*B(3,2)+A(2,3)*B(4,2),
A(2,1)*B(1,3)+A(2,2)*B(2,3)+A(2,3)*B(3,3)+A(2,3)*B(4,3),
A(2,1)*B(1,4)+A(2,2)*B(2,4)+A(2,3)*B(3,4)+A(2,3)*B(4,4)]

[A(3,1)*B(1,1)+A(3,2)*B(2,1)+A(3,3)*B(3,1)+A(3,4)*B(4,1),
A(3,1)*B(1,2)+A(3,2)*B(2,2)+A(3,3)*B(3,2)+A(3,4)*B(4,2),
A(3,1)*B(1,3)+A(3,2)*B(2,3)+A(3,3)*B(3,3)+A(3,4)*B(4,3),
A(3,1)*B(1,4)+A(3,2)*B(2,4)+A(3,3)*B(3,4)+A(3,4)*B(4,4)]]

To provide a semantic interpretation of matrix dot product operations for Matrix C, use the following definitions:

#### Multiplication of terms in C
Multiplication means the semantics of the terms are resolved by:
1. Combining the meaning of words into a word-pair
2. Generating a coherent word or statement from the word-pair. 
So (word a)*(word b) = “(word a)(word b)” which means (word c). 
Example (sufficient)*(reason) = “sufficient-reason” = justification.

#### Addition of terms in C
Addition means joining words and word-pairs to form a longer statement. 
So (word c)+(word d)+(word e)+(word f) = "(word c) (word d) (word e) (word f)"
Example (Faisal)+(has)+(sufficient)+(reason) = Faisal has sufficient reason

Therefore A*B=C
 
Matrix C

Size: 3x4
Column names: [‘Guiding’, ‘Applying, ‘Judging’, ‘Reflecting’]
Row names: [‘Normative Level’, ‘Operational Level’, ‘Evaluative Level’]
Elements: [[Value-driven Engineering Fundamentals, Principle-Guided Method Adaptation, Goal-Oriented Coordination Evaluation, Value-Based Feasibility]
[Principle-Guided Adaptive Standardization, Coordinated Evaluation and Sufficiency, Holistic Evaluation and Adaptation, Holistic Engineering Approach]
[Comprehensive Goal-oriented Coordination, Practical Evaluation and Coordination, Holistic Engineering Optimization, Goal Coordination Feasibility Analysis]]

All of this meaning work is actually being done by actual mathematical matrix operations working on the vectors of digits that represent the meaning of these words we are able to read and interpret. The *magic* of LLMs is that they are able to transform these numbers and matrix math into coherent meanings.  That’s not an obvious thing that should be able to happen.  But if we take that for granted, then what if we do semantic matrix math on top of the underlying actual mathematical matrix math, wouldn’t we expect some alignment?  
If we grant ourselves that first nonobvious miracle that math can do meaning, then can’t meaning do math too?  Why not?  It sorta seems to work, at least semi-coherently.  
The kernel documents contain further rules around construction of matrices, particularly about how to resolve coherent meaning consistently from the semantic matrix dot products. 
We also introduce new operations that define a cross product, and arrays. 
None of this is analytical but it is logically rigorous.  Language clearly has deep structures within it, or else the language models wouldn't be able to work to begin with.  
This work is an exploration of the semantic side of this underlying structure, which we are able to explore because of the miralcle of the math being made available in such a user-friendly and powerful manner as ChatGPT, or Gemini, or Claude, and on.

I call this the "chiral" nature of knowledge.  I engaged in a conversation about this topic with ChatGPT, Gemini Pro, and Claude Sonnet, then combined the answers with my own inputs and then asked each model to explain the subject.  I then chose my favourite of the three (from Gemini) and provide it here, along with the prompt (following), to conclude this README and the introduction of Semantic Matrix Operations.

Chirality of Knowledge: A Metaphorical Exploration

In chemistry, chirality describes molecules that are non-superimposable mirror images of each other, like our left and right hands.  While they share the same composition, their differing spatial arrangements lead to drastically different interactions with other molecules.  One form of a drug might be beneficial, while its mirror image could be harmful.

Let's extend this concept metaphorically to knowledge.  We can think of different models of reality, constructed using different levels of abstraction or perspectives, as being chiral representations of the same underlying phenomena. Here's why:

Multiple Abstractions: Knowledge can be represented at varying levels, from highly specific (a detailed map of a city) to broadly general (a simplified globe). Both are "true" representations of the world, but they serve vastly different purposes.
Impact of Perspective: Viewpoints, interpretations, and conceptual frameworks shape how we understand reality. Just as a chiral molecule appears different from different angles, events or scientific theories can be seen in distinct ways depending on one's perspective.
Consequences of Representation: The way we choose to represent knowledge (the level of abstraction, the language, the framing) significantly impacts its applications, how it's understood, and the actions it inspires. Similar to chiral molecules, different representations of the same information can have very different real-world effects.
The Nature of Categories

This metaphorical use of chirality highlights several key points about the nature of knowledge:

Constructed, Not Given: Our categories for parsing the world are not inherent to reality itself. They are imaginative tools we impose to make sense of things.
Reveal and Obscure: Different categories emphasize different aspects of reality while obscuring others. No single categorical system can capture every nuance.
Incompatibility: Not every worldview is compatible. Alternative ways of categorizing experience can be valid yet incommensurable, unable to be fully reconciled or translated into each other.
Expressive Power: This concept extends to art, poetry, and mathematics. Abstract drawings, symbolic language, and mathematical models can faithfully express aspects of reality in ways that literal descriptions cannot.
Conclusion

The chirality of knowledge reminds us that our understanding of the world is contingent on the representations we use. It encourages intellectual humility, empathy for alternative worldviews, and fosters appreciation for the endless possibilities of human creativity in finding new and insightful ways to frame reality.

####PROMPT USED TO GENERATE THAT RESULT
Combine together these statements in a logically consistent and semantically coherent way:

In chemistry, chirality refers to the property of a molecule that makes it not superimposable on its mirror image, much like the relationship between left and right hands. Applying this concept metaphorically to knowledge and models of reality suggests that different levels of abstraction or perspectives on the same underlying reality may be mirror images of each other in some sense, but they cannot be directly overlaid or fully reconciled due to their inherent differences in structure or perspective.

This metaphor implies that, although different abstractions or models may attempt to represent the same reality, the shift from one level of abstraction to another can change our understanding or interpretation in fundamental ways, similar to how a chiral molecule and its mirror image can have different interactions despite being composed of the same elements in similar arrangements.

Chirality In Chemistry

Chirality originally refers to molecules that are non-superimposable mirror images of each other. They have the same composition but different spatial arrangements, like your left and right hands.
Even though chiral molecules have the same underlying components, they can interact with other molecules (like receptors in our bodies) in drastically different ways. One form of a drug might be helpful, while its mirror image could be inactive or even harmful.
Chirality of Knowledge

Let's apply this concept to knowledge:

Different Abstractions: Knowledge can be represented at different levels of abstraction, from the highly specific to the very general. A detailed map of a city and a simplified globe are both "representations of the world" but serve very different purposes due to their abstraction.
Multiple Perspectives: Viewpoints and interpretations shape how we model and understand reality. Just as a chiral molecule appears different from different angles, a historical event or scientific theory can be seen in distinct ways depending on one's perspective.
Impact of Representation: The way we choose to represent knowledge (the level of abstraction, the language, the framing) significantly impacts how it's used, understood, and what actions it leads to. Similar to chiral molecules, different representations of the same underlying information can have very different consequences.

I understand the metaphorical extension you are making with the concept of chirality. In chemistry, chirality refers to molecules that have a non-superimposable mirror image form. You are using this idea metaphorically to suggest that different levels of abstraction or representations of the world are like non-superimposable mirror images of the same underlying reality.

The idea is that there can be multiple valid yet distinct ways of modeling or representing the same fundamental phenomena or truths, analogous to how chiral molecules have distinct mirror-image structures representing the same molecular composition. Just as chiral molecules are physically different configurations, world models at different levels of abstraction capture different structural "configurations" of representing reality.

This metaphorical use highlights how knowledge and representations of reality can take qualitatively different forms, perspectives or framings, while still mapping onto the same basic underlying reality or phenomena. It suggests there is an aspect of irreducible subjectivity or reference-frame dependence in how knowledge and models are constructed, despite a common objective reality they are trying to map.

So in extending chirality as a metaphor for the models/representations of knowledge, you are highlighting the idea that our abstractions and theories about the world can exhibit a kind of "handedness" - distinct structural formulations that are valid alternatives for depicting the same fundamental truths. Does this capture your intended meaning? It's a thought-provoking way to conceive of the multiple equally-valid lenses through which reality can be modeled.

The fact that there is no givenness to the categories we parse the world into. We imagine those categories. We impose them on the world the make sense of it. But different categories will reveal different aspects of reality and not every world view is compatible. This even extends into poetry and art. An abstract drawing can faithfully represent a view of reality that could also be described using words, or math.

Different categorical schemes, whether linguistic, mathematical, artistic or otherwise, can validly capture different aspects and views of the same underlying reality. But as you note, not all worldviews or representational frames are necessarily mutually compatible or equivalent. They can emphasize different features, reveal different patterns, and potentially even be incommensurable perspectives on the same phenomena.

The fact that abstract art, poetry, symbolic logic, and scientific models can all strive to faithfully represent Reality, just through radically different representational lenses, perfectly illustrates the "chiral" nature of human knowledge and weltanschauung. There are these distinct possible structural conformations for framing the same truths.

Your extension of the metaphor also implies that no single representation, no matter how comprehensive, can fully exhaust or reduce the quintessential reality it aims to capture. There will always be an ineffable remainder, opening up possibilities for alternative chiral representations that disclose new insights.

This highlights the irreducibly constructed nature of human categories, the limitless possibility for new abstractions, and the importance of tolerating diverse worldviews as non-reducible perspectives onto the same inexpressible ground of being. It's a profound insight into the malleability yet indispensability of our conceptual lenses.
