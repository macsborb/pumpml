# Contributing to PumpML

Thanks for your interest! PumpML is a research project: the most valuable contributions are **honest experiments**.

## Ground rules

- **Paper trading only.** Pull requests that send real transactions or handle private keys will not be merged.
- **Never commit secrets.** API keys live in `.env` (see `.env.example`), which is git-ignored. Same for `data/`, `models/` and `logs/`.
- **No look-ahead.** A feature may only use what was known at decision time (+3 s latency). Add a test in `tests/test_core.py` for any feature that depends on history (other coins' outcomes, memories, annotations).
- **Narrative only in the weights.** Price, holders, trade flow and time of day are deliberately excluded from `training.feature_groups`; they can be used for exclusion *rules*, or as a clearly separated experiment.

## Reporting an experiment

Follow the protocol in [`.claude/skills/pumpml-experience/SKILL.md`](.claude/skills/pumpml-experience/SKILL.md):

1. Change **one** thing, keep the same data, validation hours and number of search trials on both sides.
2. Report the PnL in SOL on the **validation hours** (never seen in training), the number of buys, and compare with "buy everything" and the current champion.
3. Under ~30 validation buys, say "inconclusive".
4. Add a line to `rapports/experiences/JOURNAL.md`.

## Before opening a pull request

```bash
python -c "import pumpml.collector, pumpml.paper, pumpml.dashboard, pumpml.model, pumpml.amm, pumpml.fame, pumpml.search"
pytest -q tests
node --check web/ai.js   # if you touched the dashboard
```

The code and comments are in French; English pull requests and issues are welcome.
