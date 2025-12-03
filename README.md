# Lab 2 - Fine-Tuning a Large Language Model

## Students
**Niklas Dahlbom and Jesper Malmgren**

## URL to our Chatbot
https://huggingface.co/spaces/JeppCodeNik/ChatBot

---

## Overview

In this lab we fine-tuned the open-source base model `unsloth/Llama-3.2-3B-Instruct` on Maxime Labonne's FineTome-100k instruction dataset using LoRA (Low-Rank Adaptation) with QLoRA quantization via Unsloth and `trl.SFTTrainer`. The training used the following hyperparameters:

| Hyperparameter | Value | Explanation |
|----------------|-------|-------------|
| `per_device_train_batch_size` | 2 | Number of examples processed at once. Limited by GPU memory. |
| `gradient_accumulation_steps` | 4 | Accumulate gradients over 4 batches before updating weights. Effective batch size = 2 x 4 = 8. |
| `num_train_epochs` | 1 | One full pass through the training data. |
| `learning_rate` | 2e-4 | Step size for weight updates. Standard value for LoRA fine-tuning. |
| `warmup_steps` | 5 | Gradually increase learning rate for first 5 steps to prevent early instability. |
| `weight_decay` | 0.01 | L2 regularization to prevent overfitting. |
| `optimizer` | adamw_8bit | Adam optimizer with weight decay, using 8-bit precision to save memory. |

---

## Task 2: Improving Model Performance

This section describes approaches to improve model performance beyond our baseline fine-tuning.

### (a) Model-Centric Approach

A **model-centric approach** keeps the training data fixed and focuses on changing the model architecture, training configuration, or optimization procedure. The idea is that with the same data, we can achieve better results by finding better hyperparameters, using a more suitable model architecture, or optimizing the training process itself.

#### Hyperparameter Tuning

Hyperparameters control how the model learns. Finding optimal values can significantly impact performance:

**Learning Rate** is arguably the most important hyperparameter. It controls how much the model weights change with each update. If the learning rate is too high, the model may overshoot optimal values and become unstable. If too low, training becomes slow and may get stuck in suboptimal solutions. We used `2e-4`, which is a common default for LoRA, but sweeping over values like `1e-4`, `2e-4`, `5e-4` could find a better fit for our specific task.

**Number of Training Epochs** determines how many times the model sees the entire dataset. We only trained for 1 epoch due to time constraints. Training for 2-3 epochs could improve results, but risks overfitting (memorizing the training data rather than learning general patterns). Using early stopping based on validation loss would help find the sweet spot.

**Batch Size and Gradient Accumulation** affect training stability. Larger batches provide more stable gradient estimates but require more memory. Our effective batch size is 8 (2 examples x 4 accumulation steps). Increasing this within memory constraints could lead to smoother training.

**Warmup Steps** gradually increase the learning rate at the start of training. This prevents large, potentially harmful weight updates when the model hasn't yet learned the task structure. We used 5 steps, but 10-100 steps might improve stability.

**Weight Decay** is a regularization technique that penalizes large weights, encouraging the model to find simpler solutions. Testing values like `0.001`, `0.01`, `0.1` could help control overfitting.

#### LoRA Configuration

LoRA (Low-Rank Adaptation) works by injecting small trainable matrices into the model instead of updating all weights. Key parameters include:

**Rank (r)** controls the expressiveness of the adapter. Higher rank means the adapter can capture more complex patterns but uses more memory. Our default rank of 16 could be increased to 32 or 64 to allow more expressive updates.

**alpha** controls how much the adapter contributes to the final output. Adjusting this parameter affects the magnitude of changes the fine-tuning introduces.

**Target Modules** determine which parts of the model get LoRA adapters. Applying LoRA to attention layers only, MLP layers only, or both can yield different results depending on the task.

#### Model Architecture

**Base Model Selection** significantly impacts both quality and inference speed. We chose Llama-3.2-3B-Instruct as a balance between quality and the ability to run on CPU for inference. 

**Quantization** trades precision for memory efficiency. We used 4-bit quantization (QLoRA), which allows training on limited GPU memory. Comparing 4-bit vs 8-bit quantization would reveal the quality-speed tradeoff.

---

### (b) Data-Centric Approach

A **data-centric approach** keeps the model and training configuration mostly fixed and focuses on improving or extending the training data. The philosophy is that model performance is often limited by data quality rather than model capacity. Better data leads to better models without architectural changes.

#### Data Quality Improvements

The quality of training data directly impacts what the model learns:

**Deduplication** removes near-duplicate examples that can cause the model to overfit to specific patterns. If the same question-answer pair appears multiple times (perhaps with slight variations), the model may memorize it rather than learning generalizable skills.

**Balancing Task Types** adjusts the distribution of examples based on the target application. If our chatbot primarily needs to handle explanatory questions, we should up-sample explanation examples and down-sample less relevant categories like code generation or creative writing.

#### Additional Data Sources

Augmenting FineTome-100k with specialized datasets can improve performance on specific capabilities:

| Dataset | Focus Area | Potential Benefit |
|---------|------------|-------------------|
| OpenAssistant Conversations | Multi-turn dialogue | Better at handling follow-up questions and maintaining context |
| GSM8K / MetaMath | Math reasoning | Improved step-by-step mathematical problem solving |
| CodeAlpaca / Code-Feedback | Programming tasks | Better code generation and explanation |
| FLAN Collection | Diverse NLP tasks | Broader coverage of different task types |
| UltraChat | Long-form dialogue | Better handling of extended conversations |

Looking at our evaluation examples, the model sometimes struggles with complex mathematical reasoning. Adding GSM8K or MetaMath to the training data would likely improve this without hurting general performance.

#### Domain-Specific Fine-Tuning

**Curriculum Learning** structures training from simple to complex. Starting with general instructions and gradually shifting to more specialized or difficult examples can help the model build foundational skills before tackling harder tasks.

**Task-Specific Adapters** train separate LoRA adapters for different domains (math, code, creative writing). At inference time, we can select the appropriate adapter based on the user's query. This allows specialization without degrading general performance.

#### Data Alignment

**Matching Output Format** ensures training examples demonstrate the desired output style. If the chatbot should provide step-by-step reasoning, the training data should contain examples with that format.

**User Feedback Loop** uses real user interactions (when permitted and anonymized) to create a fine-tuning set that reflects actual usage patterns. This closes the gap between training data and real-world deployment.

---

## Results of Improvement

To demonstrate improvement, we evaluated both the base model and our fine-tuned model using ROUGE scores on held-out test data. This section explains our evaluation methodology and presents the results.

### What is ROUGE?

**ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) is a standard metric for evaluating text generation. It measures how much the generated text overlaps with a reference answer.

The core idea is simple: if a model's output contains the same words and phrases as a known good answer, it's probably a good output. ROUGE counts these overlaps and produces a score between 0 and 1, where higher means more similar to the reference.

#### Precision, Recall, and F-measure

ROUGE uses concepts from information retrieval:

**Precision** answers: "Of everything the model said, how much was correct?"
```
Precision = matching words / total words in generated text
```

**Recall** answers: "Of everything that should be said, how much did the model include?"
```
Recall = matching words / total words in reference text
```

**F-measure** balances both. A model that outputs the entire dictionary would have perfect recall (it contains all reference words) but terrible precision (lots of irrelevant words). F-measure penalizes both extremes.
```
F-measure = 2 × (Precision × Recall) / (Precision + Recall)
```

#### The Three ROUGE Metrics

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **ROUGE-1** | Single word (unigram) overlap | Captures vocabulary similarity - does the model use the right words? |
| **ROUGE-2** | Word pair (bigram) overlap | Captures phrase-level similarity - does the model use the right phrases? |
| **ROUGE-L** | Longest common subsequence | Captures sentence structure - are words in the right order? |

**ROUGE-2 is particularly informative** because it measures phrase-level patterns. A model might use correct individual words but combine them incorrectly. ROUGE-2 catches this.

#### Limitations of ROUGE

ROUGE only measures surface-level overlap, not semantic meaning:
- "The dog chased the cat" vs "The cat was pursued by the canine" = low ROUGE but same meaning
- Doesn't understand synonyms ("big" vs "large" = no match)
- Penalizes valid paraphrasing

Despite these limitations, ROUGE is useful for comparing two models on the same task. Relative improvement is meaningful even if absolute scores aren't perfect.

### Evaluation Methodology

We evaluated both models on **100 held-out examples** from a 15% test split of FineTome-100k. For each example, we used the conversation history as input and treated the last assistant turn as the reference answer.

**Generation Settings:**
- Deterministic generation (`do_sample=False`) ensures reproducible results
- Maximum 256 new tokens per response
- Same settings for both models to ensure fair comparison

### Results

The fine-tuned model outperformed the base model across all ROUGE metrics:

| Metric | Base Model | Fine-Tuned Model | Improvement |
|--------|------------|------------------|-------------|
| ROUGE-1 | 0.4732 | 0.5323 | **+12.5%** |
| ROUGE-2 | 0.2255 | 0.2849 | **+26.4%** |
| ROUGE-L | 0.2856 | 0.3521 | **+23.3%** |

**Key Findings:**

1. **All metrics improved significantly** after just one epoch of LoRA fine-tuning, demonstrating that the approach works.

2. **ROUGE-2 showed the largest improvement (+26.4%)**, indicating that the fine-tuned model learned to use similar phrases and word combinations as the training data. This is more valuable than just using correct individual words.

3. **ROUGE-L improved by 23.3%**, showing better sentence structure and fluency in the fine-tuned model's responses.

### Response Length Analysis

| Metric | Base Model | Fine-Tuned Model | Reference |
|--------|------------|------------------|-----------|
| Mean length (words) | 165.0 | 155.7 | 216.3 |
| Median length (words) | 176.0 | 165.5 | 199.0 |

The fine-tuned model produces **shorter responses** (156 vs 165 words) while achieving **higher ROUGE scores**. This suggests improved information density - the model says more relevant content with fewer words.

### Qualitative Observations

From examining the example outputs in the Appendix:

1. **Base model over-formats**: Often adds unnecessary headers, bold text, and numbered lists even when not requested. Example 1 shows the base model adding "**What is Photosynthesis?**" headers when asked for a simple explanation.

2. **Fine-tuned model matches reference style**: Produces more direct, conversational responses that align with the training data format.

3. **Both models struggle with complex math**: Neither model consistently solves multi-step mathematical problems correctly, suggesting math-specific data augmentation would help.

4. **Creative writing improved**: The fine-tuned model's stories follow similar structures to the reference (e.g., "Once upon a time..." openings in Example 6).

---

## Conclusion

Our fine-tuning pipeline demonstrates measurable improvements over the base model, with ROUGE scores increasing by 12-26% on a held-out test set. The largest gain was in ROUGE-2 (phrase-level similarity), indicating the model learned to produce responses that match the style and phrasing of the training data.

**Most promising next steps:**

1. **Hyperparameter sweep** on learning rate and number of epochs (model-centric)
2. **Increase LoRA rank** from 16 to 32 or 64 for more expressive updates (model-centric)
3. **Add math reasoning datasets** like GSM8K to improve mathematical problem-solving (data-centric)

---

## Appendix: Example Comparisons

Below are side-by-side comparisons of the base model and fine-tuned model responses on held-out test examples. These illustrate the qualitative differences between the models.

---

### Example 1: Photosynthesis Explanation

**User Prompt:**
> Explain the process of photosynthesis in simple terms and describe its importance for the ecosystem.

**Reference:**
> Photosynthesis is a process by which plants, algae, and some bacteria convert sunlight, water, and carbon dioxide into sugar and oxygen. This process occurs in the chloroplasts of these organisms. In simple terms, sunlight is absorbed, and its energy is used to break down water and carbon dioxide molecules, which are then reassembled into sugars and oxygen. The sugar provides energy for growth, while oxygen is released into the atmosphere. Photosynthesis is essential for the ecosystem because it...

**Base Model:**
> **What is Photosynthesis?**
>
> Photosynthesis is a process by which plants, algae, and some bacteria convert light energy from the sun into chemical energy in the form of glucose (a type of sugar). This process is essential for life on Earth, as it provides energy and organic compounds for plants to grow and thrive.
>
> **The Process of Photosynthesis:**
>
> 1. **Light absorption**: Plants absorb light energy from the sun through specialized pigments such as chlorophyll.
> 2. **Water absorption**: Plants ...

**Fine-Tuned Model:**
> Photosynthesis is a process used by plants, algae, and some bacteria to convert sunlight, water, and carbon dioxide into glucose and oxygen. In simple terms, plants use sunlight to make food (glucose) and release oxygen as a byproduct. This process is crucial for the ecosystem as it provides energy and organic compounds for plants to grow, which in turn supports the food chain and maintains the balance of the environment.

---

### Example 2: Mathematical Foundation of Equations

**User Prompt:**
> What is the mathematical foundation for adding or multiplying the same value to both sides of an equation, and why does it maintain the validity of the equation?

**Reference:**
> The principle of adding or multiplying the same value to both sides of an equation is a fundamental property of equality, known as the "reflexive property of equality." This property states that if a number is equal to another number, then any operation performed on both sides of the equality will preserve that equality.
>
> For instance, consider the equation $x = 2$, where $x$ represents an unknown value. If we add 3 to both sides, we get $x + 3 = 2 + 3$, which simplifies to $x = 5$. The equalit...

**Base Model:**
> The mathematical foundation for adding or multiplying the same value to both sides of an equation lies in the properties of equality and the distributive property.
>
> **Equality Property:**
>
> The equality property states that if two expressions are equal, then any operation performed on both expressions will result in the same outcome. Mathematically, this can be expressed as:
>
> a = b ⇔ (a + c) = (b + c)
>
> where a and b are the original expressions, and c is any value.
>
> **Distributive Property:**
>
> Th...

**Fine-Tuned Model:**
> The mathematical foundation for this operation lies in the properties of equality and the concept of equivalence classes. When you add or multiply the same value to both sides of an equation, you are essentially creating a new equation that is equivalent to the original one. This is because the operations you perform do not change the relationship between the variables in the equation.
>
> For example, consider the equation $x + 2 = 5$. If you add 3 to both sides of the equation, you get $x + 5 = 8...

---

### Example 3: Geometry Problem

**User Prompt:**
> A pyramid has a rhombus-shaped base with sides of length 9 units and an angle of 120 degrees at one of its corners. The pyramid's height is 7 units. What is the surface area of the pyramid?

**Reference:**
> The base of the pyramid is a rhombus with sides of length 9 units and an angle of 120 degrees at one of its corners. The height of the pyramid is 7 units.
>
> To find the surface area of the pyramid, we need to find the area of the base and the area of each of the four triangular faces.
>
> The area of the base is given by the formula:
>
> Area of base = (1/2) * d1 * d2 * sin(theta)
>
> where d1 and d2 are the lengths of the diagonals of the rhombus and theta is the angle between the diagonals.
>
> Since the d...

**Base Model:**
> To find the surface area of the pyramid, we need to calculate the area of the base and the area of the four triangular faces, then add them together.
>
> The base of the pyramid is a rhombus with sides of length 9 units. Since the angle at one of its corners is 120 degrees, we can use trigonometry to find the length of the diagonals.
>
> The diagonals of a rhombus bisect each other at right angles, so we can use the Pythagorean theorem to find the length of the diagonals:
>
> Diagonal 1 = 2 * sin(60) * 9...

**Fine-Tuned Model:**
> The surface area of the pyramid is 173.5 square units.
>
> Explanation:
> To find the surface area of the pyramid, we need to calculate the area of the base (which is a rhombus) and the area of the four triangular faces.
>
> 1. Area of the base (rhombus):
> The area of a rhombus can be found using the formula: Area = (1/2) * d1 * d2 * sin(θ), where d1 and d2 are the lengths of the diagonals, and θ is the angle between them.
>
> Given that the sides of the rhombus are 9 units and the angle at one corner is 12...

---

### Example 4: Gravitational Waves

**User Prompt:**
> How do gravitational waves affect the curvature of spacetime and what is the mathematical expression that describes it?

**Reference:**
> Gravitational waves are ripples in the fabric of spacetime caused by the acceleration of massive objects, such as merging black holes or neutron stars. These waves propagate through spacetime at the speed of light, carrying energy and information about the events that generated them. They affect the curvature of spacetime by causing it to stretch and compress as the waves pass through.
>
> The mathematical expression that describes the effect of gravitational waves on the curvature of spacetime is ...

**Base Model:**
> Gravitational waves are ripples in the fabric of spacetime that are produced by the acceleration of massive objects, such as black holes or neutron stars. These waves cause a disturbance in the curvature of spacetime, which can be described using the Einstein field equations.
>
> The Einstein field equations describe the relationship between the curvature of spacetime (represented by the Ricci tensor, Rμν) and the mass-energy density of the objects causing the curvature. The equations are:
>
> Rμν - 1...

**Fine-Tuned Model:**
> Gravitational waves are ripples in the fabric of spacetime caused by the acceleration of massive objects, such as merging black holes or neutron stars. These waves propagate through spacetime at the speed of light and carry energy away from the source. The curvature of spacetime is affected by the presence of these waves, which causes a disturbance in the geometry of spacetime.
>
> The mathematical expression that describes the curvature of spacetime due to gravitational waves is given by the Einst...

---

### Example 5: Python Coding Task

**User Prompt:**
> Write Python code to solve the task:
> We will call a string obtained by arranging the characters contained in a string a in some order, an anagram of a.
>
> For example, `greenbin` is an anagram of `beginner`. As seen here, when the same character occurs multiple times, that character must be used that number of times.
>
> Given are N strings s_1, s_2, \ldots, s_N. Each of these strings has a length of 10 and consists of lowercase English characters. Additionally, all of these strings are distinct. Fin...

**Reference:**
> Step 1: We need to find the number of pairs of strings that are anagrams of each other.
> Step 2: We can start by creating a function to check if two strings are anagrams.
> Step 3: Then we can create a loop to compare each string with every other string in the list.
> Step 4: We should also consider the number of pairs we need to check and how we can reduce this number by half since the problem is symmetrical.
> Step 5: We should also utilize a dictionary to count the frequency of each character a...

**Base Model:**
```python
def count_anagrams(N, strings):
    anagrams = {}
    for s in strings:
        sorted_s = ''.join(sorted(s))
        if sorted_s in anagrams:
            anagrams[sorted_s].append(s)
        else:
            anagrams[sorted_s] = [s]
    count = 0
    for v in anagrams.values():
        count += len(v) * (len(v) - 1) // 2
    return count

N = int(input())
strings = [input() for _ in range(N)]
print(count_anagrams(N, strings))
```

**Fine-Tuned Model:**
> Step 1: The problem requires us to find the number of pairs of integers i, j (1 ≤ i < j ≤ N) such that s_i is an anagram of s_j.
> Step 2: We can start by creating a function to check if two strings are anagrams of each other.
> Step 3: Then, we can iterate through the input strings and count the number of pairs that satisfy the condition.
> Step 4: We also need to consider the constraints and the input format.
> Step 5: We can use a dictionary to store the count of each string and then calculate t...

---

### Example 6: Creative Writing

**User Prompt:**
> Write a short story about a computer that can predict the future.

**Reference:**
> Once upon a time, there was a computer named Oracle. Oracle was no ordinary computer – it had the unique ability to predict the future. It was the invention of a brilliant computer scientist, who had spent years creating and perfecting its software. The scientist had a vision to make the world a better place by preventing misfortunes and tragedies before they occurred.
>
> At first, people were skeptical of Oracle's abilities. But soon, the computer proved its worth. It predicted natural disasters ...

**Base Model:**
> **The Oracle of Code**
>
> In a small, cluttered laboratory, a team of brilliant engineers had been working on a top-secret project for years. Their goal was to create a computer that could predict the future. The team had tried countless approaches, from complex algorithms to advanced machine learning techniques, but nothing seemed to yield the desired results.
>
> That was until they stumbled upon an unusual approach. Instead of focusing on predicting specific events, they decided to create a comput...

**Fine-Tuned Model:**
> Once upon a time, in a small town nestled in the heart of a dense forest, there was a computer that possessed a unique gift. It could predict the future.
>
> The computer, named "Chrono," was created by a brilliant scientist who had spent years studying the mysteries of time. He had designed Chrono to be able to analyze vast amounts of data and make predictions based on that analysis.
>
> At first, Chrono's predictions were nothing more than guesses. But as time went on, the computer's accuracy began ...

---

### Example 7: Environmental Policy

**User Prompt:**
> What measures can we implement to reduce the negative impact of frequent air travel on the environment, particularly in relation to carbon emissions, atmospheric pollution, and depletion of natural resources? How can we maintain the benefits and convenience of air travel while also ensuring its sustainability in the long term? Additionally, how can we consider the impact of air travel on global temperature, biodiversity loss, and social equity when developing sustainable air travel practices?

**Reference:**
> There are several measures that can be implemented to reduce the negative impact of frequent air travel on the environment:
> 1. Promote the use of alternative modes of transportation, such as trains and buses, for shorter distances.
> 2. Encourage the use of more fuel-efficient aircraft and engines, as well as the use of alternative fuels like biofuels.
> 3. Implement carbon offsetting programs, where airlines invest in projects that reduce carbon emissions, such as renewable energy or reforestation....

**Base Model:**
> To reduce the negative impact of frequent air travel on the environment, several measures can be implemented:
>
> 1. **Carbon offsetting**: Calculate and offset carbon emissions from flights by investing in projects that reduce greenhouse gas emissions, such as reforestation or renewable energy projects.
> 2. **Fuel efficiency**: Improve aircraft fuel efficiency through more efficient engines, aerodynamic designs, and alternative fuels, such as biofuels or electric propulsion.
> 3. **Sustainable aviati...

**Fine-Tuned Model:**
> There are several measures that can be implemented to reduce the negative impact of frequent air travel on the environment. Here are some of them:
> 1. Carbon offsetting: This involves investing in projects that reduce greenhouse gas emissions, such as reforestation or renewable energy projects. Airlines can offer carbon offsetting options to passengers, and passengers can choose to offset their carbon emissions from flights.
> 2. Fuel efficiency: Airlines can improve fuel efficiency by using more e...

---

### Example 8: Earth Science

**User Prompt:**
> How does the Earth's structure impact our lives and the environment?

**Reference:**
> The Earth's structure plays a vital role in supporting life and influencing various natural phenomena. Here are three key ways in which the Earth's interior affects us:
>
> 1) The Earth's large metallic core generates a strong magnetic field, which serves as a shield against harmful solar radiation and high-energy particles. This protection is crucial for the survival of living organisms on the planet.
>
> 2) The core's heat, resulting from radioactive decay, drives convection currents in the mantle. ...

**Base Model:**
> The Earth's structure has a profound impact on our lives and the environment. Here are some ways in which the Earth's structure affects us:
>
> 1. **Climate and Weather Patterns**: The Earth's structure, including its rotation, orbit, and atmospheric circulation, influences the climate and weather patterns. This, in turn, affects the distribution of heat, moisture, and precipitation around the globe, impacting agriculture, water resources, and human settlements.
> 2. **Geological Activity**: The Eart...

**Fine-Tuned Model:**
> The Earth's structure plays a significant role in shaping our lives and the environment. Here are some ways in which it impacts us:
>
> 1. Climate: The Earth's structure, including its atmosphere, oceans, and landmasses, determines the climate we experience. The atmosphere traps heat, which in turn affects the temperature of the Earth's surface. The oceans and landmasses also play a role in regulating the climate.
>
> 2. Natural Resources: The Earth's structure is responsible for the availability of n...

---

### Example 9: Word Problem

**User Prompt:**
> There are 20 hands in Peter's class, not including his. Assume every student in the class has 2 arms and x hands. There are 11 students in Peter's class including him. What is the value of unknown variable x? What is the value of unknown variable x?

**Reference:**
> To solve this problem, we need to determine the value of x, which represents the number of hands each student in Peter's class has.
> We know that there are 11 students in Peter's class, including him. Therefore, there are 11 - 1 = 10 other students in the class.
> We also know that each student has 2 arms, which means they have 2 hands each.
> So, the total number of hands in the class, not including Peter's, is 10 students * 2 hands per student = 20 hands.
> Since each student has x hands, we can set ...

**Base Model:**
> To find the value of x, we need to know the total number of hands in the class. Since there are 20 hands not including Peter's, and Peter also has hands, we need to add Peter's hands to the total.
>
> There are 11 students in the class, including Peter. Since every student has 2 arms and x hands, the total number of hands in the class is:
>
> 11 students * 2 arms/student * x hands/arm = 22x hands
>
> However, we know that there are 20 hands not including Peter's. Since Peter also has hands, we can set up...

**Fine-Tuned Model:**
> 1. The problem states that there are 20 hands in Peter's class, not including his. This means that the total number of hands in the class, including Peter's, is 20 + 1 = 21.
> 2. The problem also states that every student in the class has 2 arms and x hands. This means that the total number of hands in the class, including Peter's, is 2 * (number of students) + x * (number of students).
> 3. We know that there are 11 students in Peter's class, including him. This means that the total number of hands...

---

### Example 10: Combinatorics

**User Prompt:**
> Solve the following math problem step-by-step.
> Simplify your answer as much as possible. Present your final answer as \boxed{Your Answer}.
> A coin is flipped eight times, and the sequence of heads and tails occurring is recorded. How many distinct sequences are possible?

**Reference:**
> Step 1: To calculate the number of distinct sequences possible, we need to determine the number of options for each flip of the coin and multiply them together.
> Step 2: Since there are two possible outcomes for each flip (heads or tails), there are 2 options for each flip.
> Step 3: Since there are 8 flips in total, we need to multiply the number of options for each flip together to find the total number of distinct sequences.
> Step 4: So, the solution is 2^8 = 256 distinct sequences.
>
> Answer:
> \box...

**Base Model:**
> To solve this problem, we need to consider the number of possible outcomes for each coin flip. Since a coin can land either heads or tails, there are 2 possible outcomes for each flip.
>
> Since the coin is flipped 8 times, we need to multiply the number of possible outcomes for each flip together to get the total number of distinct sequences.
>
> Here's the step-by-step calculation:
>
> 1. Number of possible outcomes for the first flip: 2
> 2. Number of possible outcomes for the second flip: 2
> 3. Number o...

**Fine-Tuned Model:**
> Step 1: To solve this problem, we need to understand the concept of permutations and how it applies to the given scenario.
> Step 2: The coin is flipped eight times, and the sequence of heads and tails occurring is recorded. This means that for each flip, there are two possible outcomes - heads or tails.
> Step 3: Since the coin is flipped eight times, we have a total of 8 flips, and each flip has 2 possible outcomes. This leads us to the total number of distinct sequences possible.
> Step 4: The ...

---

*More examples can be viewed by running the evaluation notebook (`EvaluateModels.ipynb`) and adjusting the `num_examples` variable.*
