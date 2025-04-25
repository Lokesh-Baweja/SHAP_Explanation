# SHAP Calculation Example

## How SHAP works under the hood?

SHAP (SHapley Additive exPlanations) calculates model predictions by assigning each feature a Shapley value,which represents the contribution of that feature to the prediction. The Shapley values are derived from cooperative game theory and are based on the idea of fairly distributing the payout (the model prediction) among the features.

This document provides a simple example of how SHAP calculates the model prediction using its formula.

# Model Setup

Suppose we have a model with three features: $X_1$, $X_2$, and $X_3$, and a prediction $f(X)$ for an instance $X = (X_1, X_2, X_3)$. We want to calculate the contribution of each feature to the prediction using SHAP.


# Shapley Value Formula
The Shapley value ϕᵢ for a feature Xᵢ is calculated as the weighted average of its marginal contribution across all possible orderings of the features:



$$
\phi_i(f) = \sum_{S \subseteq N \setminus \{i\}} 
\frac{|S|! (|N| - |S| - 1)!}{|N|!} 
\left( f(S \cup \{i\}) - f(S) \right)
$$


## Where:

N is the set of all features (in this case, N = {X₁, X₂, X₃}),

S is a subset of features excluding Xᵢ,

f(S) is the model prediction when using only the features in subset S,

f(S ∪ {i}) is the model prediction when using the features in subset S plus the feature Xᵢ.



# Step-by-Step Explanation
Consider all subsets S of features that do not include Xᵢ.
For example, if we are calculating the Shapley value for X₁, consider subsets of {X₂, X₃}.

Calculate the model prediction for each subset S (denoted as f(S))
and for the subset S ∪ {i} (denoted as f(S ∪ {i})).

Calculate the marginal contribution:
f(S ∪ {i}) - f(S)
which represents how much the feature Xᵢ adds to the prediction when included in the subset.

Weight the contributions by the factor:

(|S|! * (|N| - |S| - 1)!) / |N|!

This weight reflects how likely it is for the feature Xᵢ to appear in a particular position in an ordering of features.

Sum the weighted contributions across all subsets S.



# Simple Example
Let's consider a simple example with three features, and assume the following predictions:

$f(\emptyset) = 5$ (prediction with no features)

$f({X_1}) = 6$ (prediction with just $X_1$)

$f({X_2}) = 7$ (prediction with just $X_2$)

$f({X_3}) = 8$ (prediction with just $X_3$)

$f({X_1, X_2}) = 8$

$f({X_1, X_3}) = 9$

$f({X_2, X_3}) = 10$

$f({X_1, X_2, X_3}) = 11$

Now, we want to calculate the Shapley value for $X_1$:

$$
\phi_1(f) = \frac{1}{3!} \left( (f(\{X_1, X_2\}) - f(\{X_2\})) + (f(\{X_1, X_3\}) - f(\{X_3\})) + (f(\{X_1, X_2, X_3\}) - f(\{X_2, X_3\})) \right)
$$

Breaking it down:

$f({X_1, X_2}) - f({X_2}) = 8 - 7 = 1$

$f({X_1, X_3}) - f({X_3}) = 9 - 8 = 1$

$f({X_1, X_2, X_3}) - f({X_2, X_3}) = 11 - 10 = 1$

So:
$$
\phi_1(f) = \frac{1}{3!} (1 + 1 + 1) = \frac{3}{6} = 0.5
$$


# Final Prediction
The model's prediction for an instance $X = (X_1, X_2, X_3)$ can be expressed as the sum of the base value and the contributions from each feature:

$$
f(X) = f(\emptyset) + \phi_1(f) + \phi_2(f) + \phi_3(f)
$$
Each feature's Shapley value $\phi_i$ represents its contribution to the prediction.


# What is the Base Value
The base value (denoted as $$f(\emptyset)$$ is the expected value of the model's output when no features are known. In simple terms, it's what the model would predict if it had no information about the input.

Think of it like the model's baseline guess—its average output before seeing any features.



```python

```
