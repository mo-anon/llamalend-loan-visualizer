# llamalend-loan-visualizer

Generate time-series chart videos of [Curve LLAMMA](https://docs.curve.fi/lending/overview/) soft-liquidation loans. Each frame shows the loan's band composition (collateral vs crvUSD), oracle price, health factor, and key events (borrows, repayments, liquidations) — stitched together into an MP4 video.

## How it works

1. Fetches on-chain loan state at each sampled block via RPC multicall
2. Renders a chart frame per block with matplotlib
3. Encodes frames into an MP4 with ffmpeg

Loan events (borrow, repay, liquidation, etc.) are auto-fetched from the [Curve Prices API](https://prices.curve.fi) and marked on the chart. For markets not covered by the API, events can be specified manually.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- [ffmpeg](https://ffmpeg.org/) installed and on PATH
- An Ethereum RPC endpoint (e.g. [Alchemy](https://www.alchemy.com/))

## Setup

```bash
git clone https://github.com/mo-anon/llamalend-loan-visualizer.git
cd llamalend-loan-visualizer
```

Create a `.env` file with your RPC endpoint(s) and optionally an Etherscan API key (needed if you specify block ranges as datetime strings):

```
RPC_URL_1=https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY
RPC_URL_2=https://eth-mainnet.g.alchemy.com/v2/BACKUP_KEY
ETHERSCAN_API_KEY=YOUR_KEY
```

Multiple `RPC_URL_*` entries enable automatic round-robin failover on rate limits.

## Usage

### Web UI

```bash
uv run python app.py
```

Open http://localhost:8000, fill in the controller and user addresses, configure the block range, and click Generate. Logs stream in real time and the video plays on completion.

### CLI

Edit the constants at the top of `visualize_loan.py` (addresses, block range, step size), then:

```bash
uv run python visualize_loan.py
```

### Block range options

The block range can be specified three ways:

| START_BLOCK | END_BLOCK | DURATION | Result |
|---|---|---|---|
| 24378003 | 24378883 | — | Explicit range |
| — | 24378883 | 880 | End block, go backwards |
| 24378883 | — | 100 | Start block, go forwards |

`START_BLOCK` and `END_BLOCK` accept block numbers (int), UTC datetime strings (e.g. `"2026-01-01 12:00"`), or `None` for latest.

`BLOCK_STEP` controls sampling density — lower values produce smoother videos with more frames.

### Parallel data fetching

Data fetching is parallelized across your RPC endpoints. Each `RPC_URL_*` gets its own Web3 connection with up to 2 concurrent workers, so adding more RPC keys directly increases throughput.

`MAX_WORKERS` (CLI) controls the total number of parallel fetchers. It caps at 2× the number of RPC URLs. Set to `1` to disable parallelism (useful if your RPC keys are rate-limited):

```python
MAX_WORKERS = 1   # sequential fetching
MAX_WORKERS = 6   # default — 2 workers per RPC
```

The web UI uses the default (2 per RPC). Workers retry with exponential backoff on rate-limit errors.

### Events

By default, loan events are auto-fetched from the Curve Prices API. To disable this and provide events manually:

**Web UI:** Uncheck "Auto-fetch events" to reveal the manual events form. Add events with a block number, action type, and optional amount.

**CLI:** Set `AUTO_FETCH_EVENTS = False` and populate the `EVENTS` list in `visualize_loan.py`:

```python
AUTO_FETCH_EVENTS = False
EVENTS = [
    {"blocknumber": 18242236, "action": "borrow", "amount": 1000000},
    {"blocknumber": 18277713, "action": "repay_debt", "amount": 200000},
    {"blocknumber": 19405972, "action": "closed", "amount": 3982068},
]
```

Supported action types: `borrow`, `repay_debt`, `add_collateral`, `remove_collateral`, `closed`, `liquidated`.

## Output

Each run produces a directory under `output/`:

```
output/{date}_{collateral}-{base}_{user_short}/
├── video.mp4          # Final video (H.264, 50fps)
└── block_data.json    # Cached RPC data (re-render without re-fetching)
```
