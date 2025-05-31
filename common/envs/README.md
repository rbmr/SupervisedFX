## Derivation for `max_shares_to_sell`

Our goal is to find the maximum number of shares we can sell $S_{\text{sell}}$ such that 
$$
\frac{\text{position_value}}{\text{equity}} \geq -1,
$$
is always satisfied. We solve for the limit case where leverage is exactly -1.

### 1. The Limit Condition:

$$
\frac{\text{position_value}}{\text{equity}} = -1
$$

$$
\text{position_value} = -\text{equity}
$$

$$
\text{position_value} = -(\text{updated_cash} + \text{position_value})
$$

$$
2 \cdot \text{position_value} = -\text{updated_cash}
$$

### 2. Define Final State Variables in terms of \( S_{\text{sell}} \):

Let:

- $S_{\text{curr}} = \text{current_shares}$  
- $C_{\text{curr}} = \text{current_cash}$  
- $P_{\text{ask}} = \text{ask_price}$  
- $P_{\text{bid_eff}} = P_{\text{bid}} \cdot (1 - \text{transaction_cost_pct})$

Then:

- $\text{updated_shares} = S_{\text{curr}} - S_{\text{sell}}$
- $\text{updated_cash} = C_{\text{curr}} + S_{\text{sell}} \cdot P_{\text{bid_eff}}$
- $\text{position_value} = (S_{\text{curr}} - S_{\text{sell}}) \cdot P_{\text{ask}}$

### 3. Substitute into the Limit Condition and Solve for \( S_{\text{sell}} \):

Start with:

$$
2 \cdot (S_{\text{curr}} - S_{\text{sell}}) \cdot P_{\text{ask}} = -\left(C_{\text{curr}} + S_{\text{sell}} \cdot P_{\text{bid_eff}}\right)
$$

Distribute:

$$
2 S_{\text{curr}} P_{\text{ask}} - 2 S_{\text{sell}} P_{\text{ask}} = -C_{\text{curr}} - S_{\text{sell}} P_{\text{bid_eff}}
$$

Group $S_{\text{sell}}$ terms:

$$
S_{\text{sell}} \cdot (P_{\text{bid_eff}} - 2 P_{\text{ask}}) = -(C_{\text{curr}} + 2 S_{\text{curr}} P_{\text{ask}})
$$

Solve for $S_{\text{sell}}$:

$$
S_{\text{sell}} = \frac{-(C_{\text{curr}} + 2 S_{\text{curr}} P_{\text{ask}})}{P_{\text{bid_eff}} - 2 P_{\text{ask}}}
$$

### 4. Simplify the Formula (multiply numerator and denominator by -1):

$$
S_{\text{sell}} = \frac{C_{\text{curr}} + 2 S_{\text{curr}} P_{\text{ask}}}{2 P_{\text{ask}} - P_{\text{bid_eff}}}
$$

This formula gives us the precise maximum shares to sell to hit -100% leverage.
