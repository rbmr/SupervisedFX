import matplotlib.pyplot as plt
import numpy as np

# Setup latex for matplotlib

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']

# Graph parameters

np.random.seed(1)
tick_size = 0.1
timeframe_size = 1
eps = 0.02
t_start, t_end = 0.75, 2.25
p_low, p_high = -1.5, 2.8
timeframe_index = 1 # the index of the timeframe to label as "t", timeframe with index 0 will be t-1, and index 2 will be t+1, etc.
delay = 0.1

def ticks():
  """Generates random time ticks, avoiding round numbers."""
  t = np.sort(np.random.uniform(t_start, t_end, 256))
  return t[np.abs(t - np.round(t)) > eps]

def price(t):
  """Calculates the theoretical price at a given time."""
  return np.sin(2 * np.pi * t) + 0.5 * t - np.sin(20 * np.pi * t) * 0.25

def spread(t, min_spread = 2 * tick_size, max_spread = 9 * tick_size):
  """Calculates a variable spread size at a given time."""
  return (np.sin(np.sqrt(2) * 32 * t) * np.sin(64 * t)) * (max_spread - min_spread) * 0.5 + (max_spread + min_spread) * 0.5

# Calculate the bid and the ask prices.

def discretize(p, tick_size):
  """Rounds a price to the nearest tick size."""
  return np.round(p / tick_size) * tick_size

def ask(t):
  """Calculates the ask price at a given time."""
  return discretize(price(t) + spread(t) * 0.5, tick_size)

def bid(t):
  """Calculates the bid price at a given time."""
  return discretize(price(t) - spread(t) * 0.5, tick_size)

t_bid = ticks()
p_bid = bid(t_bid)

t_ask = ticks()
p_ask = ask(t_ask)

t = np.linspace(t_start, t_end, 256)
p = price(t)

# Ensure bid never crosses the ask.

ask_indices_for_bids = np.searchsorted(t_ask, t_bid, side='right') - 1
valid_mask = ask_indices_for_bids >= 0
overlap_mask = np.full_like(p_bid, False, dtype=bool)
overlap_mask[valid_mask] = (p_bid[valid_mask] >= p_ask[ask_indices_for_bids[valid_mask]])
p_bid[overlap_mask] = p_ask[ask_indices_for_bids[overlap_mask]] - tick_size

# Create the price plots.

plt.figure(figsize=(15, 10))
plt.step(t_ask, p_ask, where='post', label='Ask', color='blue', zorder=99)
plt.step(t_bid, p_bid, where='post', label='Bid', color='red', zorder=98)
plt.plot(t, p, label='Price', color='gray', linewidth=0.5, zorder=1)
plt.xlim([t_start, t_end])
plt.ylim([p_low, p_high])

# Create the horizontal and vertical lines, and hide the ticks.

def div_range(start, end, step):
  """Returns all numbers between start and end (inclusive) that are divisible by step."""
  assert step > 0
  assert start <= end
  step_start = np.ceil(start / step) * step
  step_end = np.floor(end / step) * step
  return np.arange(step_start, step_end + step, step)

for y in div_range(p_low, p_high + tick_size, tick_size):
  plt.axhline(y, linewidth=0.5, color='lightgray', zorder=0)
for x in div_range(t_start, t_end, timeframe_size):
  plt.axvline(x, linewidth=2, color="black", zorder=100)
plt.yticks([], [])
plt.xticks([], [])

# Add indications for the open, high, low, close, execution prices.

def create_label_and_point(t, p, ha, va, subscript = None, superscript = None):
  subscript_str = "" if subscript is None else f"_{{{subscript}}}"
  superscript_str = "" if superscript is None else f"^{{{superscript}}}"
  label = f"$P{subscript_str}{superscript_str}$"
  plt.scatter(t, p, color='black', zorder=101, s=20)
  t_text, p_text = t, p
  t_pad = 0.03
  p_pad = 0.08
  if ha == 'right':
    t_text -= t_pad
  elif ha == 'left':
    t_text += t_pad
  if va == 'bottom':
    p_text += p_pad
  elif va == 'top':
    p_text -= p_pad
  plt.text(t_text, p_text, label, ha=ha, va=va, zorder=102, fontsize=24,
           bbox=dict(facecolor='white', alpha=1.0, edgecolor='gray', boxstyle='round,pad=0.3', linewidth=0.5))

def timeframe_label(t):
  tf_i = t // timeframe_size
  rel_index = int(tf_i - timeframe_index)
  if rel_index == 0:
    time_label = 't'
  elif rel_index > 0:
    time_label = f't+{rel_index}'
  else:  # rel_index < 0
    time_label = f't{rel_index}'
  return time_label

def add_ohlc(t_series, p_series, tf_start, tf_end, superscript: str | None = None):
  """Adds the OHLC points including proper labels"""
  assert t_series.size == p_series.size

  start_idx = np.searchsorted(t_series, tf_start)
  end_idx = np.searchsorted(t_series, tf_end)

  if start_idx >= end_idx:
    return # No points, skip
  t_subset = t_series[start_idx:end_idx]
  p_subset = p_series[start_idx:end_idx]

  # Find OHLC values and their corresponding times
  t_open = t_subset[0]
  p_open = p_subset[0]
  t_close = t_subset[-1]
  p_close = p_subset[-1]
  high_idx = np.argmax(p_subset)
  p_high = p_subset[high_idx]
  t_high = t_subset[high_idx]
  low_idx = np.argmin(p_subset)
  p_low = p_subset[low_idx]
  t_low = t_subset[low_idx]

  # Determine the relative label for the timeframe
  time_label = timeframe_label(tf_start)

  # Create the price points
  proper_start = np.isclose(tf_start % timeframe_size, 0)
  proper_end = np.isclose(tf_end % timeframe_size, 0)
  if proper_start:
    create_label_and_point(t_open, p_open, 'left', 'center', f"{time_label},o", superscript)
  if proper_end:
    create_label_and_point(t_close, p_close, 'right', 'center', f"{time_label},c", superscript)
  if proper_start and proper_end:
    create_label_and_point(t_high, p_high, 'center', 'bottom', f"{time_label},h", superscript)
    create_label_and_point(t_low, p_low, 'center', 'top', f"{time_label},l", superscript)

def add_exec(t_series, p_series, t_exec, superscript: str | None):
  exec_idx = np.searchsorted(t_series, t_exec) - 1
  if exec_idx < 0 or exec_idx > t_series.size:
    return
  p_exec = p_series[exec_idx]
  time_label = timeframe_label(t_exec)
  create_label_and_point(t_exec, p_exec, 'left', 'center', f"{time_label}", superscript)

timeframe_boundaries = div_range(t_start, t_end, timeframe_size)
all_boundaries = np.unique(np.concatenate(([t_start], timeframe_boundaries, [t_end])))

for tf_start, tf_end in zip(all_boundaries[:-1], all_boundaries[1:]):
  add_ohlc(t_bid, p_bid, tf_start, tf_end, "b")
  add_ohlc(t_ask, p_ask, tf_start, tf_end, "a")

  t_exec = tf_start+delay
  if np.isclose(tf_start % timeframe_size, 0) and t_start < t_exec < t_end:
    add_exec(t_bid, p_bid, t_exec, "b")
    add_exec(t_ask, p_ask, t_exec, "a")
    plt.axvline(t_exec, linewidth=1, color="darkgray", zorder=100)

# Show the plot
plt.legend(loc="lower right")
plt.savefig('price_graph.svg', format='svg', bbox_inches='tight')