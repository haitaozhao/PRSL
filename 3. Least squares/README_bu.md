### Prediction on the Diabetes dataset by least squares



- Direct solution by equation

$$
\Beta = (X^TX)^{-1}X^TY\  or\  \ \Beta=X^{+}Y
$$

- solution with gradient descent

  1. Initialize $\Beta_1$, $i=1$, learning rate $\alpha$, and threshold value $\varepsilon$ 
  2. Compute $\Beta_{i+1}=\Beta_i-\alpha\grad J(\Beta_i)$ 
  3. If $\|J(\Beta_{i+1}-J(\Beta_i))\|^2\leq \varepsilon $, go Step 4; else $i=i+1$, go Step 2
  4. Output  $\Beta_{i+1}$

  where 
  $$
  J(\Beta)=\frac{1}{n}(Y-X\Beta)^T(Y-X\Beta)
  $$

  $$
  \grad J(B) = \frac{2}{n}(-X^TY+X^TX\Beta)
  $$



In the code, RMSProp is also used to compute $\Beta_{i+1}$

