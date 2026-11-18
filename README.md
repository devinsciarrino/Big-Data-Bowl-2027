# Big-Data-Bowl-2027

Entry for the [NFL Big Data Bowl 2027](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027):
linking 10 Hz NFL Combine tracking data to regular-season NFL performance.

## Setup

```bash
uv sync                       # Python env + dependencies
./scripts/download_data.sh    # needs a Kaggle API token in ~/.kaggle/ and accepted competition rules
uv run jupyter lab
```

```python
from bdb27.data import load, game_tracking
players = load("players")
ct = load("combine_tracking", lazy=True)
```

Competition data is CC BY-NC 4.0 and is never committed (`data/` is gitignored).

See [PLAN.md](PLAN.md) for the timeline, rules and candidate ideas.
