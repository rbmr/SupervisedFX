
### Notation

For all mathematical descriptions of the Forex trading environment we use the following notation:

* $t$: the time period.
* $P_t^{\text{bid,open}}, P_t^{\text{bid,high}}, P_t^{\text{bid,low}}, P_t^{\text{bid,close}}$: the bid prices measured at open/high/low/close of period $t$.
* $P_t^{\text{ask,open}}, P_t^{\text{ask,high}}, P_t^{\text{ask,low}}, P_t^{\text{ask,close}}$: the ask prices measured at open/high/low/close of period $t$.
* $\kappa$: transaction cost percentage (e.g. $\kappa=0.001$ for 0.1%)
* $\epsilon_t \in [-1,1]$: current exposure of the agent at time $t$ (fraction of equity invested).
* $a_t$: action taken by agent at start of period $t$
* $x_t\in[-1,1]$: target exposure chosen by the agent at start of period $t$.
* $p_t$: position value at period $t$
* $C_{t}$, $S_{t}$: cash and number of shares held during period $t$
* $E_t^{\text{open}}, E_t^{\text{high}}, E_t^{\text{low}}, E_t^{\text{close}}$: equity measured at open/high/low/close of period $t$

If open, high, low or close isn't specified (e.g. $E_t$ or $P_t^{bid}$), that means it can be any.

Each simulation starts with some cash-only $\texttt{initial_capital}$. This means: $E_0 = C_0 = \texttt{initial_capital}$, and $S_0 = 0$. However, since we assume Zero Market Impact, this initial capital does not have any significant impact on the simulation.

### Portfolio value (equity) $E_t$ and position value $p_t$

For the determination of position value we uphold Mark-to-Market (MTM), this means valuing open positions based on current market prices rather than historical purchase costs. If shares $S_t$ is positive, the market price is the bid price (what others are willing to pay). If the shares is negative, the ask price is used (the price to buy back the asset). We write this as:

* $P_t^* = \begin{cases}
P_t^{\text{bid}}, & S_{t}\ge0,\\
P_t^{\text{ask}}, & S_{t}\lt0.
\end{cases}$ 
* $p_t = S_t \cdot P_t^*$

This impacts the calculation of equity and exposure directly: 

* $E_t = C_t + p_t$
* $\displaystyle \epsilon_t = \frac{p_t}{E_t}$

### Trade execution

At the end of every time period $t$ the agent picks an action $a_t$ (end-of-day trading). 

1. $t-1$ Close (execute trade)
2. [Gap]
3. $t$ Open
4. $t$ High/Low
5. $t$ Close (calculate reward)
6. Increment $t$, repeat 1-5

Each selected action is executed through the following steps.

#### 1. Action $a_t$ → Target Exposure $x_t$

1. Continuous action space ($\texttt{n_actions}=0$):
    
   $$
   x_t = a_t,\quad a_t\in[-1,1].
   $$

2. Discrete action space ($\texttt{n_actions}>0$):
   $$
   x_t = \frac{a_t - \texttt{n_actions}}{\texttt{n_actions}},
   \quad a_t\in\{0,1,\dots,2 \cdot \texttt{n_actions}\}.
   $$

#### 2. Jitter Mitigation

If the requested change in exposure is negligible,

$$
\bigl|\,x_t - \varepsilon_{t-1}\bigr| < 10^{-5},
$$

then no trade is executed:
$C_t = C_{t-1},\ S_t = S_{t-1}$.

This is mainly implemented for performance reasons, as agents tend to hold their positions for long periods of time. Re-computation in such cases would take unnecessary resources, as the computation would yield (almost) the same resulting state as before.

#### 3. Determining Trade Size

1. Target position value

   $$
   V_t^\star = x_t \cdot E_{t}^{\text{open}}.
   $$

2. Target number of shares

   $$
   S_t^\star = 
   \begin{cases}
   \displaystyle\frac{V_t^\star}{P_t^{\text{ask,open}}}, & x_t>0\quad(\text{long}),\\[1em]
   \displaystyle\frac{V_t^\star}{P_t^{\text{bid,open}}}, & x_t\le0\quad(\text{short}).
   \end{cases}
   $$

3. Shares to trade

   $$
   \Delta S_t = S_t^\star - S_{t-1}.
   $$

#### 4. Executing Buy vs. Sell

Let $\lambda = 1 + \kappa$, $\mu = 1 - \kappa$.

1. If $\Delta S_t>0$ (Buy):

   * Cost per share: $\lambda \cdot P_t^{\text{ask}}$
   * Maximum shares to buy (to avoid >100% long):
     $$
     \displaystyle S_t^{\max} = \frac{C_{t-1}}{\lambda\,P_t^{\text{ask}}}
     $$
   * Actual Shares Bought: $S_b = \min(\Delta S_t,\,S_t^{\max})$
   * Update:

     $$
     C_t = C_{t-1} - S_b \cdot \lambda \cdot P_t^{\text{ask}}, 
     \quad
     S_t = S_{t-1} + S_b.
     $$

2. If $\Delta S_t<0$ (Sell):

   * Proceeds per share: $\mu \cdot P_t^{\text{bid}}$
   * Maximum shares to sell (to avoid >100% short):

     $$
     S_t^{\max,-}
     = \frac{\,C_{t-1} + 2\,S_{t-1}\,P_t^{\text{ask}}\,}{\,2\,P_t^{\text{ask}} - \mu\,P_t^{\text{bid}}\,}
     $$
   * Actual Shares Sold: $S_s = \min(-\Delta S_t,\,S_t^{\max,-})$
   * Update:

     $$
     C_t = C_{t-1} + S_s \cdot \mu \cdot P_t^{\text{bid}}, 
     \quad
     S_t = S_{t-1} - S_s.
     $$

#### 5. Reward

Default per-step reward is equity change:

$$
r_t = E_t^{\text{close}} - E_{t-1}^{\text{close}}.
$$

A custom reward $r_t = f(\text{env})$ may override this.



*This completes the mathematical specification of the trading steps used in `ForexEnv`.*

### Derivation for `max_shares_to_buy` in `buy_shares`

Our goal is to find the maximum number of shares we can buy $S_{\text{buy}}$ such that
$$
\frac{\text{Position Value}}{\text{Equity}} \leq 1,
$$
is always satisfied.  We solve for the limit case where exposure is exactly 1.

#### 1. The Limit Condition:

Start with
$$
\frac{\text{Position Value}}{\text{Equity}} = 1
$$
Multiply both sides by $\text{Equity}$:
$$
\text{Position Value} = \text{Equity}
$$
But
$$
\text{Equity} = \text{Updated Cash} + \text{Position Value}
$$
So
$$
\text{Position Value} = \text{Updated Cash} + \text{Position Value}
$$
which implies
$$
\text{Updated Cash} = 0.
$$

#### 2. Define Final State Variables in terms of $S_{\text{buy}}$:

Let:
- $S_{\text{curr}} = \text{Current Shares}$  
- $C_{\text{curr}} = \text{Current Cash}$  
- $P_{\text{ask}} = \text{Ask Price}$  
- $P_{\text{ask_eff}} = P_{\text{ask}} \cdot \bigl(1 + \text{transaction cost pct}\bigr)$

Then:
- $\text{Updated Shares} = S_{\text{curr}} + S_{\text{buy}}$
- $\text{Updated Cash} = C_{\text{curr}} - S_{\text{buy}} \cdot P_{\text{ask_eff}}$

#### 3. Substitute into the Limit Condition and Solve for $S_{\text{buy}}$:

Since $\text{Updated Cash} = 0$:
$$
C_{\text{curr}} - S_{\text{buy}} \cdot P_{\text{ask_eff}} = 0
$$
Re‐arrange:
$$
S_{\text{buy}} \cdot P_{\text{ask_eff}} = C_{\text{curr}}
$$
Therefore
$$
S_{\text{buy}} = \frac{C_{\text{curr}}}{P_{\text{ask}} \cdot \bigl(1 + \text{transaction cost pct}\bigr)}.
$$

### Derivation for `max_shares_to_sell` in `sell_shares`

Our goal is to find the maximum number of shares we can sell $S_{\text{sell}}$ such that
$$
\frac{\text{Position Value}}{\text{Equity}} \geq -1,
$$
is always satisfied.  We solve for the limit case where exposure is exactly $-1$.

#### 1. The Limit Condition:

Start with
$$
\frac{\text{Position Value}}{\text{Equity}} = -1
$$
Multiply both sides by $\text{Equity}$:
$$
\text{Position Value} = -\,\text{Equity}
$$
But
$$
\text{Equity} = \text{Updated Cash} + \text{Position Value}
$$
So
$$
\text{Position Value} = -\bigl(\text{Updated Cash} + \text{Position Value}\bigr)
$$
Re‐arrange:
$$
2 \cdot \text{Position Value} = -\,\text{Updated Cash}.
$$

#### 2. Define Final State Variables in terms of $S_{\text{sell}}$:

Let:
- $S_{\text{curr}} = \text{Current Shares}$  
- $C_{\text{curr}} = \text{Current Cash}$  
- $P_{\text{ask}} = \text{Ask Price}$  
- $P_{\text{bid_eff}} = P_{\text{bid}} \cdot \bigl(1 - \text{transaction cost pct}\bigr)$

Then:
- $\text{Updated Shares} = S_{\text{curr}} - S_{\text{sell}}$
- $\text{Updated Cash} = C_{\text{curr}} + S_{\text{sell}} \cdot P_{\text{bid_eff}}$
- $\text{Position Value} = \bigl(S_{\text{curr}} - S_{\text{sell}}\bigr) \cdot P_{\text{ask}}$

#### 3. Substitute into the Limit Condition and Solve for $S_{\text{sell}}$:

Start with

$$
2 \cdot \bigl(S_{\text{curr}} - S_{\text{sell}}\bigr) \cdot P_{\text{ask}} = -\bigl(C_{\text{curr}} + S_{\text{sell}} \cdot P_{\text{bid_eff}}\bigr)
$$

Distribute:

$$
2\,S_{\text{curr}}\,P_{\text{ask}} -2\,S_{\text{sell}}\,P_{\text{ask}}
=
-\,C_{\text{curr}}-\,S_{\text{sell}}\,P_{\text{bid_eff}}
$$

Group the $S_{\text{sell}}$ terms on one side:
$$
S_{\text{sell}}\;\bigl(P_{\text{bid_eff}} - 2\,P_{\text{ask}}\bigr)
=
-\bigl(C_{\text{curr}} + 2\,S_{\text{curr}}\,P_{\text{ask}}\bigr)
$$
Solve for $S_{\text{sell}}$:
$$
S_{\text{sell}}
=
\frac{-(C_{\text{curr}} + 2\,S_{\text{curr}}\,P_{\text{ask}})}
     {P_{\text{bid_eff}} - 2\,P_{\text{ask}}}
=
\frac{C_{\text{curr}} + 2\,S_{\text{curr}}\,P_{\text{ask}}}
     {2\,P_{\text{ask}} - P_{\text{bid_eff}}}.
$$

This formula gives us the precise number of shares to sell to hit $-100\%$ leverage.
