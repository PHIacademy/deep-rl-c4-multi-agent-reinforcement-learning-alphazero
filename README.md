# AlphaZero for Tic-Tac-Toe — Educational Walkthrough

This document walks through [`alphazero-TicTacToe.ipynb`](./alphazero-TicTacToe.ipynb), [`ConnectN.py`](./ConnectN.py), [`Play.py`](./Play.py), and [`MCTS.py`](./MCTS.py) — covering how to set up, play, train, and test an AlphaZero agent, as well as how the game engine, rendering, and Monte Carlo Tree Search (MCTS) are implemented under the hood.

The goal is to understand not just *what* the code does, but *why* it's written the way it is. The material is split into two parts:

- **Part 1 — Using the Notebook**: follows [`alphazero-TicTacToe.ipynb`](./alphazero-TicTacToe.ipynb) top to bottom — setting up a game, playing it, building the neural network, training an agent, and testing it. Treats the MCTS search as a "black box" that picks good moves.
- **Part 2 — Under the Hood**: opens up [`ConnectN.py`](./ConnectN.py), [`Play.py`](./Play.py), and especially [`MCTS.py`](./MCTS.py) to explain exactly how that black box is implemented — useful if you want to modify, extend, or debug the code.

**A quick note on framing:** this whole system is also a compact example of **multi-agent reinforcement learning (MARL)** — a setting where more than one learning agent acts in a shared environment. Here, two agents (the two players) compete in a zero-sum game, and — as you'll see in the code — both are actually driven by one shared "actor-critic" neural network playing against itself. Keep this in mind as you read; it's explained in full in the synthesis section at the end.

---

## Key Features

- **[PUCT-based MCTS search](#puct-search)** balancing exploitation vs. exploration
- **[Board symmetry augmentation](#board-symmetry-augmentation)** for sample-efficient training
- **[Temperature-controlled move sampling](#temperature-sampling)** for exploration vs. playing strength
- **[Policy-distillation self-play training loop](#policy-distillation)**
- **[Generalized Connect-N engine](#generalized-connect-n)**, not hardcoded to 3×3 Tic-Tac-Toe

---

## 1. Project Files Overview

| File | Purpose |
|---|---|
| [`ConnectN.py`](./ConnectN.py) | The **game environment**. A general "Connect-N in a row" game (Tic-Tac-Toe is just a 3×3, N=3 special case). Tracks board state, current player, and score. |
| [`MCTS.py`](./MCTS.py) | The **Monte Carlo Tree Search** engine used by AlphaZero. Builds a search tree, explores it using a neural network's guidance, and picks moves. |
| [`Play.py`](./Play.py) | An **interactive/visual game runner** built on `matplotlib`. Lets a human click moves, or watches AI vs AI games animate. |
| [`alphazero-TicTacToe.ipynb`](./alphazero-TicTacToe.ipynb) | The **notebook** that ties everything together: defining the neural network ("Policy"), training loop, and playing against the trained agent. |

---

# Part 1 — Using the Notebook

This part follows the notebook's own flow: set up a game, play it by hand, build the neural network, wrap it in a search-based player, train it through self-play, then test what it learned.

## 2. Setting Up and Playing a Game

The notebook starts by creating a game:

```python
from ConnectN import ConnectN

game_setting = {'size': (3, 3), 'N': 3}
game = ConnectN(**game_setting)
```

- `size` = board dimensions (3×3 for Tic-Tac-Toe).
- `N` = number in a row needed to win (3 for Tic-Tac-Toe).
- `ConnectN` is deliberately generalized — it can support much larger boards (e.g. Connect-4-style games) simply by changing these two parameters.

### Game state basics

A `ConnectN` object exposes:
- **`game.state`** — a matrix (3×3 here) representing the board. Empty = `0`.
- **`game.player`** — whose turn it is. First player is `+1`, second player is `-1`, and turns alternate automatically after every move.
- **`game.score`** — `None` while the game is undecided, `0` for a draw, `+1` if player 1 wins, `-1` if player 2 wins.

```python
game.move((0, 1))
print(game.state)
print(game.player)   # switches automatically after a valid move
print(game.score)    # still None — game not over
```

That's all you need to know to *use* `ConnectN` from the notebook. For how scoring, the pie rule, and fast copying are actually implemented internally, see **§8** in Part 2.

## 3. Playing Interactively

```python
from Play import Play
gameplay = Play(ConnectN(**game_setting), player1=None, player2=None)
```

- Setting a player to `None` means **that player is a human**, controlled by mouse clicks on the rendered board.
- Setting a player to a function (like `Policy_Player_MCTS`, introduced below) means **that player is AI**.
- You can mix and match: human vs human, human vs AI, or AI vs AI.

This is enough to play a game against yourself and get a feel for the board — for how the rendering and click-handling are implemented under the hood, see **§9** in Part 2.

## 4. Building the Neural Network ("Policy")

AlphaZero's core idea is to replace a hand-crafted evaluation function with a **neural network that outputs two things at once**:

1. A **policy** — a probability distribution over which move to play next.
2. A **critic / value** — a single number estimating who's likely to win from the current position (from -1 to +1).

```python
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=2, stride=1, bias=False)
        self.size = 2*2*16
        self.fc = nn.Linear(self.size, 32)

        # policy head
        self.fc_action1 = nn.Linear(32, 16)
        self.fc_action2 = nn.Linear(16, 9)

        # critic/value head
        self.fc_value1 = nn.Linear(32, 8)
        self.fc_value2 = nn.Linear(8, 1)
        self.tanh_value = nn.Tanh()

    def forward(self, x):
        y = F.relu(self.conv(x))
        y = y.view(-1, self.size)
        y = F.relu(self.fc(y))

        # --- policy head ---
        a = F.relu(self.fc_action1(y))
        a = self.fc_action2(a)

        # mask out illegal moves (cells already filled)
        avail = (torch.abs(x.squeeze()) != 1).type(torch.FloatTensor)
        avail = avail.view(-1, 9)

        maxa = torch.max(a)
        exp = avail * torch.exp(a - maxa)   # zero out illegal moves BEFORE normalizing
        prob = exp / torch.sum(exp)

        # --- value/critic head ---
        value = F.relu(self.fc_value1(y))
        value = self.tanh_value(self.fc_value2(value))

        return prob.view(3, 3), value
```

### Why not just use a plain Softmax?

A standard 9-way softmax would happily assign non-zero probability to squares that are already occupied — those moves are illegal. The fix:

1. Build an **availability mask**: `1` where a cell is empty, `0` where it's occupied.
2. Multiply the *exponentials* (not the raw logits) by this mask before summing, so illegal moves get exactly zero probability.
3. Normalize as usual.

This one trick means the rest of the code (MCTS, training) never has to worry about the network suggesting illegal moves.

### Why a tanh on the value head?

Game outcomes are naturally bounded: `+1` (win), `-1` (loss), `0` (draw). `tanh` squashes the output into `(-1, 1)`, matching that scale.

**Exercise prompt in the notebook:** the `Policy` class ships with the body commented out — you're meant to fill in the convolution → fully-connected → policy/critic heads yourself, which is why the solution above is shown as the reference answer.

## 5. Making a Move with MCTS (as a Black Box)

```python
import MCTS
from copy import copy
import random

def Policy_Player_MCTS(game):
    mytree = MCTS.Node(copy(game))
    for _ in range(50):
        mytree.explore(policy)

    mytreenext, (v, nn_v, p, nn_p) = mytree.next(temperature=0.1)
    return mytreenext.game.last_move

def Random_Player(game):
    return random.choice(game.available_moves())
```

At this level, you don't need to know exactly how `MCTS.Node` works internally — just what it does:

- `MCTS.Node(copy(game))` builds a fresh search tree rooted at the current position.
- `mytree.explore(policy)` runs one round of tree search, using the `Policy` network to guide which branches look promising. Calling it 50 times means 50 simulated look-aheads before committing to a move.
- `mytree.next(temperature=0.1)` picks the move that search found strongest (low temperature ≈ "always pick the best move found").
- The result is fed straight into `Play(..., player2=Policy_Player_MCTS)` as an AI opponent.

Before any training, the untrained `Policy` network's outputs are close to random, so `Policy_Player_MCTS` mostly just plays plausible-but-not-perfect moves — MCTS alone (with an untrained network) can still *block* obvious threats because the tree search finds forced wins/losses directly, but it won't yet know deeper positional strategy (e.g. why the center is stronger than an edge).

For exactly how `explore()` and `next()` build and search the tree internally, see **§10** in Part 2.

<a id="policy-distillation"></a>
## 6. Training Loop

```python
game = ConnectN(**game_setting)
policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=.01, weight_decay=1.e-4)

episodes = 400
outcomes, losses = [], []

for e in range(episodes):
    mytree = MCTS.Node(ConnectN(**game_setting))
    vterm, logterm = [], []

    while mytree.outcome is None:
        for _ in range(50):
            mytree.explore(policy)

        current_player = mytree.game.player
        mytree, (v, nn_v, p, nn_p) = mytree.next()     # temperature=1 (default) during training
        mytree.detach_mother()

        # policy cross-entropy term, with a zero-baseline constant
        loglist = torch.log(nn_p) * p
        constant = torch.where(p > 0, p * torch.log(p), torch.tensor(0.))
        logterm.append(-torch.sum(loglist - constant))

        vterm.append(nn_v * current_player)

    outcome = mytree.outcome
    outcomes.append(outcome)

    loss = torch.sum((torch.stack(vterm) - outcome)**2 + torch.stack(logterm))
    optimizer.zero_grad()
    loss.backward()
    losses.append(float(loss))
    optimizer.step()
    del loss
```

### What's actually happening every episode:

1. **Self-play**: the agent plays an entire game against *itself*, using MCTS (guided by the current network) to choose every move, all the way until the game ends (`mytree.outcome is not None`).
2. At each move, `mytree.next()` returns a 4-tuple `(v, nn_v, p, nn_p)`:
   - **`v`** — MCTS's own running value estimate for the position (`self.V`, negated). It's returned for completeness/logging but is **not actually used in the loss** — only the network's own value guess (`nn_v`) is trained.
   - **`nn_v`** — the network's raw critic output at that position (before search) — **this is what actually gets trained** to match the game outcome.
   - **`p`** — the *MCTS-refined* move probabilities (from visit counts). Because these come from real look-ahead search, they're treated as the "ground truth" **target** the policy head should learn to imitate.
   - **`nn_p`** — the network's *raw* policy output at that same position (before search) — this is what actually gets trained to match `p`.
3. **Building the loss, term by term:**

```python
loglist = torch.log(nn_p) * p                                  # p * log(π_θ) — cross-entropy piece
constant = torch.where(p > 0, p * torch.log(p), torch.tensor(0.))  # p * log(p) — the zero-baseline constant
logterm.append(-torch.sum(loglist - constant))                 # policy loss for this move

vterm.append(nn_v * current_player)                             # critic loss ingredient for this move
```
   These two lists accumulate one entry **per move** of the game. Only once the game ends do we know the final outcome `z` (`mytree.outcome`), so the value term can only be completed at the very end:

```python
loss = torch.sum((torch.stack(vterm) - outcome)**2 + torch.stack(logterm))
```

   This matches the notebook's LaTeX formula:

$$L = \sum_t \Big\{ (v_\theta^{(t)} - z)^2 - \sum_a p_a^{(t)} \log \pi_\theta(a \mid s_t) \Big\} + \text{constant}$$

   - **`(nn_v · current_player − outcome)²`** ↔ the first term `(v_θ − z)²`: pushes the network's own value prediction toward the actual game outcome.
   - **`-Σ(loglist − constant)`** ↔ the second term: cross-entropy between the MCTS-derived probabilities `p` (target) and the network's own policy `nn_p`/`π_θ` (prediction), training the network to imitate what full tree search discovered — this is where the network "distills" the benefits of search back into itself.
   - Summing `torch.stack(vterm)` and `torch.stack(logterm)` over the whole game, then taking one `torch.sum`, means **one game = one gradient step** — every move's contribution is added together before `loss.backward()` is called once per episode.
- **Why the `current_player` multiplication (`nn_v * current_player`)?** `nn_v` (as returned by `mytree.next()`) is the network's raw critic estimate of the position **from the perspective of whichever player is about to move** at that point in the game — it's a "how good is this for me, the mover" score. But the final outcome `z` (`mytree.outcome`) is always recorded on an **absolute** scale, relative to *player 1* (`+1` = player 1 won, `-1` = player 2 won). Multiplying by `current_player` (`+1` or `-1`) re-expresses that move's critic estimate onto the same absolute, player-1-relative scale as `z`.

  This alignment has to be applied to **every single entry appended to `vterm`** over the course of the game — not just the final one — because `vterm` collects one value per move, made alternately by both players, and all of it gets compared against the *single* scalar `outcome` at once via broadcasting: `(torch.stack(vterm) - outcome)**2`. If the sign flip were skipped for any move made by player `-1`, that entry would be measured against the wrong side of the "who's winning" scale and the squared-error loss would be nonsensical for that term. So yes — this includes making sure the *last* move's estimate (the one right before the game ends) lines up correctly with `outcome` too, but it's really a per-move requirement that happens to apply uniformly across the whole trajectory.
- **Why the extra constant (`p log p`)?** Mathematically it doesn't affect gradients (it doesn't depend on `π_θ`) — it's purely there so that when `π_θ` perfectly matches `p`, the loss reads exactly `0`, giving you a clean, interpretable "how far from perfect am I" metric. This matters in self-play settings, where there's no separate labeled validation set — the loss curve is your main signal that learning is actually happening.

### Why self-play "just works"

There's something almost bootstrapping about this loop: the network guides the search, and the search (which is more accurate than the raw network, because it looks ahead) produces better training targets that the network is then trained to match. Repeated over hundreds of episodes, the network gradually "absorbs" the benefit of search into its instant, one-shot evaluation.

## 7. Observing Training Results

After ~400 episodes (roughly a minute or two for Tic-Tac-Toe):

- **Loss** trends downward — but this is a rough proxy signal, not a precise accuracy metric, since the model is bootstrapping against its own evolving search rather than fixed ground truth.
- **Outcomes** shift from fairly random win/loss/draw distributions toward mostly **draws** (the game-theoretic optimal result for perfect Tic-Tac-Toe play) and increased **player 1 wins** — consistent with first-move advantage in Tic-Tac-Toe when the opponent isn't perfect.

Playing against the trained agent typically demonstrates:
- **Correct blocking behavior** — it stops opponent three-in-a-rows.
- **Recognizing the center as the strongest opening move.**
- **Exploiting opponent mistakes** — if you play a suboptimal move, the trained agent capitalizes on it and can force a win.
- Some **residual gaps in deeper strategy** may remain with limited training (e.g., subtler positional traps), which further training episodes or a larger/better `Policy` network can improve.

---

# Part 2 — Under the Hood: How the Code Works

This part opens up [`ConnectN.py`](./ConnectN.py), [`Play.py`](./Play.py), and [`MCTS.py`](./MCTS.py) to explain how the notebook-level black boxes from Part 1 are actually implemented — useful if you want to modify, extend, or debug them.

<a id="generalized-connect-n"></a>
## 8. [`ConnectN.py`](./ConnectN.py) Internals

### How scoring works (`get_score`)

Rather than scanning the whole board after every move (slow), `ConnectN` only checks lines **through the last move played** — since a win can't happen anywhere else on the turn it happens:

```python
def get_score(self):
    if self.n_moves < 2*self.N-1:
        return None            # not enough moves yet for anyone to have won

    i, j = self.last_move
    hor, ver, diag_right, diag_left = get_lines(self.state, (i, j))

    for line in [ver, hor, diag_right, diag_left]:
        if in_a_row(line, self.N, self.player):
            return self.player  # current player just won

    if np.all(self.state != 0):
        return 0                # board full, no winner -> draw

    return None                 # game continues
```

`get_lines()` is a piece of NumPy indexing gymnastics that, given a `(row, col)` location, extracts the full horizontal, vertical, and both diagonal lines passing through it — even on non-square generalized boards. This is the least "beginner friendly" part of the file, but conceptually all it does is: *"grab the 4 lines that pass through the last move, and check if any of them has N-in-a-row."*

`get_winning_loc()` uses the same line-extraction logic, but on an index grid instead of the state itself, so that it can return the *coordinates* of the winning line for rendering (used by `Play.draw_winner`, see §9).

### The Pie Rule (optional, advanced)

`ConnectN` optionally supports the **pie rule**, a fairness mechanic sometimes used in strong first-move games: the first move is placed as `0.5` instead of `1`, and the second player can choose to either play elsewhere or "swap" and take over that first move. This isn't used in the basic Tic-Tac-Toe notebook, but it's there if you want to experiment with fairness for stronger games like Connect-4.

### Fast copying (`__copy__`)

MCTS needs to simulate *many* hypothetical future games without touching the real one. To make that efficient, `ConnectN` implements a custom `__copy__` that copies just the necessary fields (state array, player, move count, etc.) rather than doing a full deep copy of everything — this matters a lot when you're doing thousands of simulations per move.

## 9. [`Play.py`](./Play.py) Internals

- On construction, `Play.__init__` immediately calls `self.play()`, which sets up a `matplotlib` figure (grid lines, aspect ratio, hidden spines for a clean board look).
- **If both players are AI**, it uses `matplotlib.animation.FuncAnimation` driven by a Python generator (`move_generator`) that yields one move at a time — this animates AI vs AI games automatically.
- **If at least one player is human**, it instead binds a mouse click handler (`button_press_event` → `self.click`). Clicking near a grid cell rounds the click coordinates to the nearest cell and attempts a move there.
- **`draw_move`** places a colored marker (salmon for player 1, sky blue for player 2) at the location just played.
- **`draw_winner`** checks whether the game has ended, and if so, overlays star markers on the winning line (using `ConnectN.get_winning_loc()`) and disconnects the click handler so no more moves can be made.

You don't need to modify `Play.py` to complete the notebook exercises, but understanding it helps if you want to customize colors, marker sizes, or animation speed.

## 10. [`MCTS.py`](./MCTS.py) Internals — Deep Dive

This is the heart of AlphaZero — it's what turns a so-so neural network into a much stronger player by *searching* several moves ahead before committing to one. This is what `Policy_Player_MCTS` (§5) and the training loop (§6) were actually calling.

<a id="board-symmetry-augmentation"></a>
### 10.1 Board symmetries (`t0`–`t7`, `process_policy`)

A Tic-Tac-Toe board (or any square board) looks "the same" strategically under 8 symmetries: 4 rotations × reflection. `MCTS.py` defines these as lambda transforms (`t0`...`t7`) plus their inverses.

```python
def process_policy(policy, game):
    if game.size[0] == game.size[1]:
        t, tinv = random.choice(transformation_list)      # all 8 symmetries
    else:
        t, tinv = random.choice(transformation_list_half)  # only reflections, for non-square boards

    frame = torch.tensor(t(game.state * game.player), dtype=torch.float)
    input = frame.unsqueeze(0).unsqueeze(0)
    prob, v = policy(input)
    mask = torch.tensor(game.available_mask())
    return game.available_moves(), tinv(prob)[mask].view(-1), v.squeeze().squeeze()
```

Every time the network evaluates a position, a **random symmetry is applied first**, then undone on the output (`tinv`). This effectively gives the network free extra training data — it sees the "same" position from a different orientation each time — which speeds up learning.

Note also `game.state * game.player`: the board is always presented **from the current player's point of view** (so the network only ever has to learn "am I winning?" rather than separately learning both "am I player +1 winning" and "am I player -1 winning").

### 10.2 The `Node` class

Each `Node` wraps:
- `self.game` — a (copied) game state at this point in the tree.
- `self.child` — dictionary mapping `action → Node`, filled in lazily once explored.
- `self.U` — the "upper confidence bound" score used to pick which branch to explore next.
- `self.V` — the current running estimate of value at this node (from MCTS backpropagation).
- `self.nn_v`, `self.prob` — the raw neural network value/probability for this node (kept as `torch.tensor` so gradients can flow through it later during training).
- `self.N` — visit count.
- `self.outcome` — `None` until the game is decided at this node; if the game *is* already over here, `V`/`U` are hard-set (`inf` for a win, computed directly from `game.score`).

### 10.3 Expanding the tree — `create_child`

```python
def create_child(self, actions, probs):
    games = [copy(self.game) for a in actions]
    for action, game in zip(actions, games):
        game.move(action)
    child = {tuple(a): Node(g, self, p) for a, g, p in zip(actions, games, probs)}
    self.child = child
```

For every legal action, it clones the current game, plays that single move, and wraps the result as a new child `Node`. If a resulting child position is already a terminal win/loss, its `Node.__init__` marks its `U` as `±inf` immediately — a key optimization explained next.

<a id="puct-search"></a>
### 10.4 Exploration — `explore(policy)`

This is the classic 4-step MCTS cycle: **select → expand → evaluate → backpropagate.**

**Step 1 — Select**, walking down the tree by always choosing the child with the highest `U`:

```python
while current.child and current.outcome is None:
    child = current.child
    max_U = max(c.U for c in child.values())
    actions = [a for a, c in child.items() if c.U == max_U]
    action = random.choice(actions)   # break ties randomly

    if max_U == -float("inf"):
        current.U = float("inf");  current.V = 1.0;  break
    elif max_U == float("inf"):
        current.U = -float("inf"); current.V = -1.0; break

    current = child[action]
```

**Why the infinity flip?** If *every* child of a node is a certain loss (`U = -inf`) for whoever moves next, that means the *player who just moved* made a winning move — so from one level up, this node itself is a guaranteed win (`U = +inf`). This propagates "solved" positions upward without wasting further search effort re-confirming them — a nice speed optimization once the tree has found forced wins/losses.

**Step 2 — Expand & evaluate**, once we reach a leaf that hasn't been expanded yet:

```python
if not current.child and current.outcome is None:
    next_actions, probs, v = process_policy(policy, current.game)
    current.nn_v = -v
    current.create_child(next_actions, probs)
    current.V = -float(v)
```

Note the **sign flip on `v`**. The neural network evaluates a position from the perspective of "the player about to move there," but that position was arrived at *by the previous player's move*, so from the previous player's perspective the value is the *negation*.

**Step 3 — Backpropagate**, walking back up to the root, updating running averages and the `U` scores of every sibling along the way:

```python
current.N += 1
while current.mother:
    mother = current.mother
    mother.N += 1
    mother.V += (-current.V - mother.V) / mother.N   # incremental average, sign-flipped

    for sibling in mother.child.values():
        if sibling.U not in (float('inf'), -float('inf')):
            sibling.U = sibling.V + c * float(sibling.prob) * sqrt(mother.N) / (1 + sibling.N)

    current = current.mother
```

The `U` formula is the classic **PUCT** (Predictor + UCT) formula used by AlphaZero:

$$U = V + c \cdot P \cdot \frac{\sqrt{N_{mother}}}{1+N_{child}}$$

- `V` rewards moves that have historically led to good outcomes.
- The second term rewards moves the neural network is confident about (`P`) but that haven't been explored much yet (`N_child` small) — balancing **exploitation vs. exploration**.
- `c` is a tunable constant (set to `1.0` at the top of the file) controlling how much weight exploration gets.

<a id="temperature-sampling"></a>
### 10.5 Picking a move — `next(temperature)`

After running `explore()` many times (50 times per move in this notebook), `next()` converts visit counts into a move choice:

```python
def next(self, temperature=1.0):
    child = self.child
    max_U = max(c.U for c in child.values())

    if max_U == float("inf"):
        # a guaranteed winning move exists — always take it
        prob = torch.tensor([1.0 if c.U == float("inf") else 0 for c in child.values()])
    else:
        maxN = max(node.N for node in child.values()) + 1
        prob = torch.tensor([(node.N / maxN) ** (1 / temperature) for node in child.values()])

    prob /= torch.sum(prob)
    nextstate = random.choices(list(child.values()), weights=prob)[0]
    return nextstate, (-self.V, -self.nn_v, prob, nn_prob)
```

**The temperature parameter** controls how "sharp" this move selection is:

$$p_a = \frac{N_a^{1/T}}{\sum_a N_a^{1/T}}$$

- **High T (e.g. `T=1`, used during self-play training)** → moves are chosen roughly proportional to how often they were visited, encouraging exploration/diversity in the training data.
- **Low T (e.g. `T=0.1`, used when actually playing)** → the move with the largest visit count is picked almost deterministically — you want the *strongest* move, not a random one, when actually competing.

`detach_mother()` simply severs the link back up the tree after a real move is made, so old branches (and their memory) can be garbage collected — you don't need the "parent" of the move you actually just committed to.

---

# Synthesis

## 11. Multi-Agent RL (MARL) View: Tic-Tac-Toe as Two Agents

It's useful to step back and view this whole system through the lens of **multi-agent reinforcement learning (MARL)** — a setting where more than one learning agent acts in a shared environment. Tic-Tac-Toe is one of the simplest possible MARL examples: exactly **two agents**, taking turns, in a **zero-sum, fully competitive** game.

### The two agents

- **Agent A** = player `+1`
- **Agent B** = player `-1`

Both "agents" are, in fact, driven by the **same neural network** (`policy`). This is a very common MARL trick called **self-play with a shared/parameter-tied policy**: instead of training two separate networks, one network plays *both* sides, always evaluating the board from "the current mover's point of view" (recall `game.state * game.player` in `process_policy`, §10.1). This is why a single `Policy` object and a single `MCTS.Node` tree can represent an entire two-player match.

### Actor and Critic roles

`Policy` is an **actor-critic** network — both roles live in the same model, just with two separate output heads:

| Role | Output | Question it answers | Code location |
|---|---|---|---|
| **Actor** | `prob` (a 3×3 move distribution) | *"Given this board, which move should I try next?"* | `fc_action1`, `fc_action2` in `Policy.forward` |
| **Critic** | `value` (a single scalar in `[-1, 1]`) | *"Given this board, how good is my position right now?"* | `fc_value1`, `fc_value2`, `tanh_value` in `Policy.forward` |

MCTS uses **both** outputs together every time it expands a node (`process_policy` in `MCTS.py`):
- The **actor's** `prob` seeds the `P` term in the PUCT formula (`U = V + c·P·√N/(1+N_child)`), biasing search toward moves the network already thinks are promising.
- The **critic's** `value` is used to *evaluate a leaf node instantly*, instead of having to simulate the game randomly all the way to the end (as classic MCTS without a critic would). This is what makes AlphaZero-style search so much faster than plain Monte Carlo rollout methods.

So in MARL terms: the actor proposes actions, the critic estimates state value, and MCTS combines both with explicit look-ahead search to make a stronger decision than either component could make alone.

**One important caveat:** the critic's value estimate is *not* used to train the actor via a policy-gradient/advantage rule the way a classical actor-critic RL algorithm (e.g. A2C) would. Instead, the actor is trained by imitating MCTS's search-refined move distribution `p` (cross-entropy, §6), and the critic is trained by regression toward the final game outcome `z` — both are supervised targets rather than TD-bootstrapped ones. So "actor-critic" here is an accurate *architectural* description (two heads, two roles), but the *training* procedure is closer to search-guided distillation than to textbook actor-critic reinforcement learning.

### Competition, not cooperation

Tic-Tac-Toe is **purely competitive (zero-sum)**: one player's win (`+1`) is exactly the other player's loss (`-1`), and a draw (`0`) is neutral for both. There's no shared objective to cooperate toward — this is *adversarial* self-play, not cooperative MARL.

You can see the competitive, zero-sum structure baked directly into the code's sign-flipping:

- In `MCTS.Node.explore`, `mother.V += (-current.V - mother.V) / mother.N` — a value that's good for the child is bad for the mother, because the mother and child represent opposing players.
- In `Policy_Player_MCTS` / `process_policy`, `nn_v = -v` for the same reason — a leaf's value from the mover's perspective must be negated to be meaningful for the player who moved into that position.
- In training, `vterm.append(nn_v * current_player)` re-expresses everyone's value estimate on a single shared scale (relative to player `+1`), so the eventual game outcome `z` can be compared consistently across both agents' moves.

(Cooperative MARL — where multiple agents share a *joint* reward and must coordinate — would look very different: shared or team rewards, and no negation between agents. Tic-Tac-Toe's structure is the opposite case, which is exactly why the code negates values every time perspective switches between players.)

### The MARL training loop, end to end

Putting it together, one training episode is a full multi-agent self-play loop:

1. **Reset** the environment — a fresh `ConnectN` board, `player = +1` to move first.
2. **Both agents share one policy.** At each turn, the *mover* uses MCTS (guided by the shared actor-critic network) to search ahead and pick a move — effectively, the network is "playing itself."
3. **Turns alternate** — after each move, `ConnectN` flips `self.player`, handing control to the other agent.
4. **Environment provides the outcome signal** only at the end (`game.score`) — this terminal `+1` / `-1` / `0` is the zero-sum reward both agents' value estimates are trained against.
5. **A single shared update** (`loss.backward()`) improves the one network that plays both sides — so the *next* self-play episode, both agents are simultaneously a little stronger and a little better at anticipating the opponent's best response.

This is what makes AlphaZero-style self-play elegant as a MARL example: rather than needing an explicit opponent model or separate agents to coordinate/compete, a single shared actor-critic bootstraps its own strength by repeatedly competing against itself and learning from every game it just played.

## 12. Key Takeaways / Concepts Recap

| Concept | Where it lives | Why it matters |
|---|---|---|
| Game abstraction (state, player, score) | `ConnectN.py` | Decouples "rules of the game" from "how to play well" |
| Fast, last-move-only score checking | `ConnectN.get_score` | Makes thousands of MCTS simulations per move computationally feasible |
| Availability-masked softmax | `Policy.forward` | Neural net never proposes illegal moves |
| Board symmetry augmentation | `MCTS.process_policy` | Free extra "training signal" from a single game state |
| PUCT formula (`U = V + c·P·√N_mother/(1+N_child)`) | `MCTS.Node.explore` | Balances exploiting known-good moves vs. exploring uncertain ones |
| Temperature-controlled move sampling | `MCTS.Node.next` | High T for diverse self-play data, low T for strong actual play |
| Self-play + distillation loss | Notebook training loop | Lets the network "learn from its own lookahead search," improving without any external labeled data |
| Shared actor-critic playing both sides (MARL) | `Policy` + `MCTS.Node.explore` | One network plays both competing agents; actor proposes moves, critic evaluates positions, sign-flips encode the zero-sum competition |

## 13. Suggested Exercises

1. **Fill in the `Policy` network** (`__init__` and `forward`) yourself before peeking at the reference implementation.
2. **Derive and implement the AlphaZero loss function** in the training loop, following the LaTeX formula given in the notebook.
3. Try adjusting:
   - Number of MCTS simulations per move (`50` → try `20` or `200`) and observe playing strength vs. speed trade-offs.
   - The exploration constant `c` in `MCTS.py`.
   - Temperature during training vs. play.
4. Once comfortable, try scaling `ConnectN` up (e.g. `size=(4,4), N=3` or larger Connect-4-style boards) to see how training time and required network capacity change.
