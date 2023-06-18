# Genetic-Algorithm-Text-Classification
## Introduction
A genetic algorithm is an algorithm based on the mechanism of evolution (survival of the fittest). Its goal is to maximize/minimize the fitness function optimization. It starts with a set of K randomly generated states. The states are referred to as chromosomes and the set of states is called a population. Selection, reproduction, and mutation operators are used to generate new chromosomes and states.

**Selection**: It involves selecting the chromosomes that will produce the next generation.

**Reproduction**: Crosses over chromosomes to generate new solutions.

**Mutation**: Randomly changes some values of a chromosome.

![Genetic Algorithm](https://github.com/kursatkomurcu/Genetic-Algorithm-Text-Classification/blob/main/images/genetic_algorithm.png)

## Factors Affecting the Performance of Genetic Algorithms
**Population size/Number of chromosomes**: Increasing the number of chromosomes increases the runtime, while reducing it eliminates chromosome diversity.

**Mutation rate**: When chromosomes start to resemble each other but are still far from the solution points, mutation is the only way for the genetic algorithm to escape from the stuck point (all chromosomes in the same plateau). However, assigning a high value to mutation will hinder the genetic algorithm from reaching a stable point.

**Number of Crossover Points**: Although crossover is usually performed at a single point, research has shown that multi-point crossover can be very beneficial in some problems.

**Evaluation of offspring resulting from crossover**: Determines whether both offspring will be used or not.

**How the state encoding is done?:** Encoding a parameter linearly or logarithmically can make a significant difference in the performance of the genetic algorithm.

**How success is evaluated?:** Poorly designed evaluation function can extend the runtime and may never lead to a solution.

## Data Preprocessing
Firstly, the sentences in the dataset are tokenized into words. Then, unnecessary words (stopwords) and punctuation marks are removed to create a word pool. These words are used to create individuals that will form the population.

```python
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english')) # write turkish using turkish dataset
word_pool = x
word_pool = word_pool.ravel()

word_pool_tokens = []
for i in word_pool:
    token = nltk.word_tokenize(i)
    for word in token:
        if (word not in stop_words) and (word.isalnum()):
            word_pool_tokens.append(word)

print(word_pool_tokens) # word pool
```
## Text Classification with Genetic Algorithm
In this study, binary text classification is performed on five different datasets using a genetic algorithm, and hyperparameters are examined. No additional classification algorithm is used during the classification process. The method used is as follows:

**1.** Random words are assigned to the generated individual.

**2.** It is assumed that the first half of the individual holds the words for positive data, while the second half holds the words for negative data. For example, an individual with 100 genes is expected to contain words like "good, great, excellent" in the first half, and words like "terrible, bad, dislike" in the second half.

**3.** At each step, the genetic algorithm's selection, reproduction, and mutation features are used to reach the target individual.

## Fitness Function
When creating the fitness function, the individual is first divided into two halves. Then, the sentences in the dataset are tokenized into words. If the separated words appear only in the first half of the individual, they are considered positive, and if they appear only in the second half, they are considered negative. The function then adds the count of positive and negative occurrences and returns the total count.

```python
def fitness_function(individual, x):
    """
    individual: The individual used for classification
    x: Dataset consisting only of sentences
    toplam: it means sum
    
    When creating the fitness function, the individual is first divided into two parts.
    Then, the sentences in the dataset are split into words.
    The words that appear only in the first half of the individual are considered positive, and the words that appear only in the second half are considered negative.
    Then, the function adds the larger count between positive and negative counts to the total variable and returns the sum.
    """
    toplam = 0
    pos_arr = individual[:len(individual) // 2]
    neg_arr = individual[len(individual) // 2:]
    
    for i in range(len(x)):
        count_pos = 0
        count_neg = 0
        for j in x[i]:
            tokens = nltk.word_tokenize(j)

            for token in tokens:
                if (token in pos_arr) and (token not in neg_arr):
                    count_pos += 1
                if (token in neg_arr) and (token not in pos_arr):
                    count_neg += 1

            if count_pos > count_neg:
                toplam += count_pos

            elif count_neg > count_pos:
                toplam += count_neg

    return toplam
```

![Rows: Population size Columns: Mutation rate](https://github.com/kursatkomurcu/Genetic-Algorithm-Text-Classification/blob/main/images/table.png)
Table --> Rows: Population size Columns: Mutation rate

## Evaluation and Comments
As the mutation level increases, the value of the fitness function decreases. This is because when too many words change in the corresponding individual, the crossover operation becomes less significant. If the mutation rate is too low, the fitness function may remain at the same level when the words change very little. In the above table, experiments were conducted with our first dataset, the SMS Spam Collection Dataset, and it was observed that the highest fitness function was achieved at a mutation rate of 0.05.
In the experiments, the highest fitness function value was reached when the population size was 100. When the population size is low, the diversity may be reduced, leading to a decrease in success rate. Additionally, increasing the iteration count, increasing the length of the lists within individuals, using better data preprocessing methods, or writing the fitness function using a machine learning algorithm and improving the accuracy of this machine learning algorithm with genetic algorithm can be done to increase the success rate.

