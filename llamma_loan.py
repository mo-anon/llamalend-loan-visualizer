"""
Reusable data-fetching module for Curve LLAMMA loans.

Provides RPCManager for round-robin RPC failover, CurveLoanGetter for
querying on-chain loan state, and supporting utilities.
"""

import json
import os
import random
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

from dotenv import load_dotenv
from web3 import Web3
from web3.exceptions import BlockNotFound
from w3multicall.multicall import W3Multicall

load_dotenv()

PRECISION = 10**18

BASE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# ABI loading
# ---------------------------------------------------------------------------

def load_abi(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


CONTROLLER_ABI = load_abi(BASE_DIR / "abis" / "CONTROLLER.abi")
ERC20_ABI = load_abi(BASE_DIR / "abis" / "ERC20.abi")
LLAMMA_ABI = load_abi(BASE_DIR / "abis" / "LLAMMA.abi")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def checksum_address(address: str) -> Optional[str]:
    """Converts an address to its checksummed version."""
    try:
        return Web3.to_checksum_address(address)
    except (ValueError, TypeError):
        return None


def timestamp_to_block(target_ts: int, etherscan_api_key: str | None = None) -> int:
    """Look up the block number closest to *target_ts* via the Etherscan V2 API.

    Requires an Etherscan API key — passed explicitly or via
    ``ETHERSCAN_API_KEY`` in the environment (free at etherscan.io).
    """
    api_key = etherscan_api_key or os.environ.get("ETHERSCAN_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "ETHERSCAN_API_KEY is required for datetime→block resolution. "
            "Get a free key at https://etherscan.io/myapikey and add it to .env"
        )
    url = (
        f"https://api.etherscan.io/v2/api?chainid=1&module=block&action=getblocknobytime"
        f"&timestamp={target_ts}&closest=before&apikey={api_key}"
    )
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read())
    if data.get("status") != "1":
        raise RuntimeError(f"Etherscan API error: {data.get('message')} — {data.get('result')}")
    return int(data["result"])


def resolve_block(value: Union[int, str, None], w3: Web3, log_fn=print,
                   etherscan_api_key: str | None = None) -> int:
    """Resolve a block specifier to a concrete block number.

    *value* can be:
    - ``int``   — returned as-is
    - ``None``  — resolves to the latest block
    - ``str``   — parsed as a UTC datetime (``"2026-01-01"`` or
      ``"2026-01-01 12:00"`` etc.) and resolved via the Etherscan API
    """
    if value is None:
        block = w3.eth.block_number
        log_fn(f"  END_BLOCK=None → latest block {block}")
        return block
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        # Try DD/MM/YYYY or DD/MM/YYYY HH:MM:SS first, then ISO format
        for fmt in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y"):
            try:
                dt = datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue
        else:
            dt = datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
        target_ts = int(dt.timestamp())
        block = timestamp_to_block(target_ts, etherscan_api_key=etherscan_api_key)
        log_fn(f"  Resolving \"{value}\" → block {block}")
        return block
    raise TypeError(f"Expected int, str, or None — got {type(value).__name__}")


def is_rate_limit_error(error) -> bool:
    """Checks if an error is a rate limit or quota exceeded error."""
    error_str = str(error).lower()
    rate_limit_indicators = [
        '429',
        '503',
        'rate limit',
        'quota',
        'exceeded',
        'too many requests',
        'service unavailable',
        'enhance your calm',
    ]
    return any(indicator in error_str for indicator in rate_limit_indicators)


# ---------------------------------------------------------------------------
# RPCManager
# ---------------------------------------------------------------------------

class RPCManager:
    """Round-robin RPC failover.

    Reads ``RPC_URL_1``, ``RPC_URL_2``, ... from the environment (loaded via
    *python-dotenv*).  Exposes a ``.w3`` property for the current Web3
    instance and a ``.switch()`` method that advances to the next URL.
    """

    def __init__(self, log_fn=print, rpc_url: str | None = None):
        self.log_fn = log_fn
        self._lock = threading.Lock()
        self.urls: list[str] = []
        if rpc_url:
            self.urls.append(rpc_url)
        i = 1
        while True:
            url = os.environ.get(f"RPC_URL_{i}")
            if url is None:
                break
            if url not in self.urls:
                self.urls.append(url)
            i += 1
        if not self.urls:
            raise RuntimeError("No RPC URLs provided and no RPC_URL_* environment variables found.")
        self._index = 0
        self._w3 = Web3(Web3.HTTPProvider(self.urls[self._index]))
        if not self._w3.is_connected():
            raise ConnectionError(f"Failed to connect to RPC URL 1: {self.urls[0][:50]}...")
        self.log_fn(f"Connected to RPC 1/{len(self.urls)}")

    @property
    def w3(self) -> Web3:
        return self._w3

    def switch(self) -> Web3:
        """Advance to the next RPC URL and return the new Web3 instance.

        Thread-safe: protected by an internal lock so concurrent callers
        cannot interleave index bumps.
        """
        with self._lock:
            if len(self.urls) < 2:
                raise RuntimeError("Only one RPC URL configured; cannot switch.")
            self._index = (self._index + 1) % len(self.urls)
            url = self.urls[self._index]
            self.log_fn(f"\n⚠️  Switching to RPC URL {self._index + 1}/{len(self.urls)}: {url[:50]}...")
            self._w3 = Web3(Web3.HTTPProvider(url))
            if not self._w3.is_connected():
                raise ConnectionError(f"Failed to connect to RPC URL {self._index + 1}")
            return self._w3


# ---------------------------------------------------------------------------
# call_with_retry
# ---------------------------------------------------------------------------

def call_with_retry(func, *args, rpc_manager: Optional[RPCManager] = None,
                    loan_getter=None, max_retries=3, log_fn=print, **kwargs):
    """Call *func* with retry logic, switching RPC on rate-limit errors.

    Parameters
    ----------
    rpc_manager : RPCManager, optional
        Used for switching RPC URLs on rate-limit errors.
    loan_getter : CurveLoanGetter, optional
        If provided, its Web3 instance and contracts are refreshed after a
        switch.
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except BlockNotFound:
            raise
        except Exception as e:
            last_error = e
            if is_rate_limit_error(e) and rpc_manager is not None and len(rpc_manager.urls) > 1:
                log_fn(f"  ⚠️  Rate limit error on attempt {attempt + 1}: {str(e)[:100]}")
                if attempt < max_retries - 1:
                    w3 = rpc_manager.switch()
                    if loan_getter is not None:
                        loan_getter.update_web3(w3)
                    time.sleep(1)
                    continue
            if attempt == max_retries - 1:
                raise last_error
            time.sleep(2 ** attempt)

    raise last_error


# ---------------------------------------------------------------------------
# CurveLoanGetter
# ---------------------------------------------------------------------------

class CurveLoanGetter:
    """Query a Curve LLAMMA pool for loan state at a specific block."""

    def __init__(self, w3_instance: Web3, block_number: int,
                 controller_address: str, user_address: str,
                 rpc_manager: Optional[RPCManager] = None, log_fn=print):
        self.w3 = w3_instance
        self.block = block_number
        self.controller_address = controller_address
        self.user_address = user_address
        self.rpc_manager = rpc_manager
        self.log_fn = log_fn
        self.controller = self.w3.eth.contract(
            address=checksum_address(self.controller_address), abi=CONTROLLER_ABI)
        # Derive AMM (LLAMMA) address from the controller
        self.llamma_address = self.controller.functions.amm().call(
            block_identifier=self.block)
        self.llamma = self.w3.eth.contract(
            address=checksum_address(self.llamma_address), abi=LLAMMA_ABI)
        self.get_initial_parameters()
        self.log_fn(f"AMM: {self.llamma_address}")
        self.log_fn("Initialization complete.")

    def update_web3(self, w3: Web3) -> None:
        """Refresh Web3 instance and recreate contract objects."""
        self.w3 = w3
        self.controller = w3.eth.contract(
            address=checksum_address(self.controller_address), abi=CONTROLLER_ABI)
        self.llamma = w3.eth.contract(
            address=checksum_address(self.llamma_address), abi=LLAMMA_ABI)

    def get_initial_parameters(self) -> None:
        """Retrieves initial parameters of the pool using a single multicall."""
        self.A = self.llamma.functions.A().call(block_identifier=self.block)
        self._get_coins()
        self.get_block_vars(self.block)

    def get_block_vars(self, block_number: int) -> None:
        """Retrieves the current block variables using multicall."""
        self.block = block_number

        def _get_block():
            return self.w3.eth.get_block(self.block)['timestamp']

        self.block_time = call_with_retry(_get_block, rpc_manager=self.rpc_manager, loan_getter=self)
        self.block_date = datetime.fromtimestamp(self.block_time, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

        user_address = self.user_address

        def _multicall():
            mc = W3Multicall(self.w3)

            llamma_calls = [
                ("price_oracle", "price_oracle()(uint256)"),
                ("get_base_price", "get_base_price()(uint256)"),
                ("active_band", "active_band()(int256)"),
                ("read_user_tick_numbers", "read_user_tick_numbers(address)(int128[2])", (user_address,)),
                ("band_balances", "get_xy(address)(uint256[][2])", (user_address,)),
            ]

            controller_calls = [
                ("health_full", "health(address,bool)(int256)", (user_address, True)),
                ("health_not_full", "health(address,bool)(int256)", (user_address, False)),
                ("debt", "debt(address)(uint256)", (user_address,)),
                ("user_state", "user_state(address)(uint256[4])", (user_address,)),
            ]

            for call in llamma_calls:
                if len(call) == 2:
                    _, sig = call
                    mc.add(W3Multicall.Call(self.llamma_address, sig, ()))
                else:
                    _, sig, args = call
                    mc.add(W3Multicall.Call(self.llamma_address, sig, args))

            for call in controller_calls:
                if len(call) == 2:
                    _, sig = call
                    mc.add(W3Multicall.Call(self.controller_address, sig, ()))
                else:
                    _, sig, args = call
                    mc.add(W3Multicall.Call(self.controller_address, sig, args))

            return mc.call(block_identifier=self.block), llamma_calls, controller_calls

        results, llamma_calls, controller_calls = call_with_retry(
            _multicall, rpc_manager=self.rpc_manager, loan_getter=self)

        for (attr, *_), value in zip(llamma_calls + controller_calls, results):
            setattr(self, attr, value)

    def _get_coins(self) -> None:
        """Retrieves token addresses, symbols, and decimals at the specified block."""
        self.coins = []
        for i in range(8):
            try:
                def _get_coin_address(idx=i):
                    return self.llamma.functions.coins(idx).call(block_identifier=self.block)

                address = call_with_retry(_get_coin_address, rpc_manager=self.rpc_manager, loan_getter=self)
                token_contract = self.w3.eth.contract(address=address, abi=ERC20_ABI)

                def _get_decimals():
                    return token_contract.functions.decimals().call(block_identifier=self.block)

                def _get_symbol():
                    return token_contract.functions.symbol().call(block_identifier=self.block)

                decimals = call_with_retry(_get_decimals, rpc_manager=self.rpc_manager, loan_getter=self)
                symbol = call_with_retry(_get_symbol, rpc_manager=self.rpc_manager, loan_getter=self)
                self.coins.append({"address": address, "symbol": symbol, "decimals": decimals})
            except Exception:
                break

    def fetch_block_data(self, block_number: int, w3: Web3 | None = None,
                         max_retries: int = 6) -> dict:
        """Fetch all loan data for a single block. Thread-safe.

        When *w3* is provided (parallel mode), uses it directly with
        retry + exponential backoff — no RPC switching.  When *w3* is
        ``None``, falls back to ``rpc_manager.w3``.
        """
        _w3 = w3 if w3 is not None else (
            self.rpc_manager.w3 if self.rpc_manager else self.w3)
        user_address = self.user_address
        last_error = None

        for attempt in range(max_retries):
            try:
                # 1. Block timestamp
                block_time = _w3.eth.get_block(block_number)['timestamp']
                block_date = datetime.fromtimestamp(
                    block_time, tz=timezone.utc
                ).strftime('%Y-%m-%d %H:%M:%S')

                # 2. Multicall for all loan state
                mc = W3Multicall(_w3)

                llamma_calls = [
                    ("price_oracle", "price_oracle()(uint256)"),
                    ("get_base_price", "get_base_price()(uint256)"),
                    ("active_band", "active_band()(int256)"),
                    ("read_user_tick_numbers",
                     "read_user_tick_numbers(address)(int128[2])",
                     (user_address,)),
                    ("band_balances",
                     "get_xy(address)(uint256[][2])", (user_address,)),
                ]
                controller_calls = [
                    ("health_full",
                     "health(address,bool)(int256)", (user_address, True)),
                    ("health_not_full",
                     "health(address,bool)(int256)", (user_address, False)),
                    ("debt", "debt(address)(uint256)", (user_address,)),
                    ("user_state",
                     "user_state(address)(uint256[4])", (user_address,)),
                ]

                for call in llamma_calls:
                    if len(call) == 2:
                        _, sig = call
                        mc.add(W3Multicall.Call(
                            self.llamma_address, sig, ()))
                    else:
                        _, sig, args = call
                        mc.add(W3Multicall.Call(
                            self.llamma_address, sig, args))
                for call in controller_calls:
                    if len(call) == 2:
                        _, sig = call
                        mc.add(W3Multicall.Call(
                            self.controller_address, sig, ()))
                    else:
                        _, sig, args = call
                        mc.add(W3Multicall.Call(
                            self.controller_address, sig, args))

                results = mc.call(block_identifier=block_number)

                # 3. Parse multicall results
                raw = {}
                for (attr, *_), value in zip(
                        llamma_calls + controller_calls, results):
                    raw[attr] = value

                # 4. Compute bands (pure computation)
                bands = self._compute_bands_static(
                    raw['read_user_tick_numbers'], raw['band_balances'],
                    raw['get_base_price'], self.A, self.coins)

                # 5. Build output dict
                price_oracle = raw['price_oracle'] / 1e18
                health = raw['health_full'] / PRECISION * 100
                debt = raw['debt'] / PRECISION
                collateral_native = 0
                collateral_crvusd = 0

                user_state = raw.get('user_state')
                if user_state:
                    try:
                        collateral_native = (user_state[0]
                                             / (10 ** self.coins[1]['decimals']))
                        collateral_crvusd = user_state[1] / 1e18
                        debt = user_state[2] / 1e18
                    except Exception:
                        pass

                total_collateral_value = (collateral_native * price_oracle
                                          + collateral_crvusd)
                if total_collateral_value == 0 and bands:
                    band_collateral_value = sum(
                        b.get('collateral_as_crvusd', 0) for b in bands)
                    band_crvusd_value = sum(
                        b.get('crvusd_amount', 0) for b in bands)
                    total_collateral_value = (band_collateral_value
                                              + band_crvusd_value)
                    collateral_crvusd = band_crvusd_value

                return {
                    'block': block_number,
                    'block_date': block_date,
                    'price_oracle': price_oracle,
                    'health': health,
                    'get_base_price': raw['get_base_price'],
                    'A': self.A,
                    'bands': bands,
                    'debt': debt,
                    'total_collateral_value': total_collateral_value,
                    'collateral_native': collateral_native,
                    'collateral_crvusd': collateral_crvusd,
                    'num_bands_state': len(bands),
                }

            except BlockNotFound:
                raise
            except Exception as e:
                last_error = e
                if attempt == max_retries - 1:
                    raise
                if is_rate_limit_error(e):
                    backoff = (2 ** attempt) + random.uniform(0, 1)
                else:
                    backoff = 2 ** attempt
                time.sleep(backoff)

        raise last_error  # unreachable but satisfies type checkers

    @staticmethod
    def _compute_bands_static(read_user_tick_numbers, band_balances,
                              get_base_price, A, coins) -> List[dict]:
        """Compute band data from raw multicall results. Pure function."""
        assert coins[0]['symbol'] == 'crvUSD', "Expected first coin to be crvUSD"

        first_user_band, last_user_band = read_user_tick_numbers
        base_price = get_base_price / float(PRECISION)
        k = (A - 1) / A
        bands = []

        for i, band_num in enumerate(
                range(first_user_band, last_user_band + 1)):
            max_price = base_price * (k ** band_num)
            min_price = base_price * (k ** (band_num + 1))
            band_width = max_price - min_price
            avg_price = (min_price + max_price) / 2
            crvusd_amount = band_balances[0][i] / PRECISION
            collateral_amount = (band_balances[1][i]
                                 / (10 ** coins[1]['decimals']))
            collateral_value = collateral_amount * avg_price
            total_value_for_bars = collateral_value + crvusd_amount

            bands.append({
                "band_num": band_num,
                "min_price": min_price,
                "max_price": max_price,
                "avg_price": avg_price,
                "band_width": band_width,
                "crvusd_amount": crvusd_amount,
                "collateral_amount": collateral_amount,
                "collateral_as_crvusd": collateral_value,
                "total_collateral_value": collateral_value,
                "band_total_value": total_value_for_bars,
            })

        return bands

    def band_max_price(self, n: int) -> float:
        """Calculates the top of the range of the nth band."""
        A = self.A
        base_price = self.get_base_price / float(PRECISION)
        k = (A - 1) / A
        return base_price * (k ** n)

    def band_min_price(self, n: int) -> float:
        """Calculates the bottom of the range of the nth band."""
        return self.band_max_price(n + 1)

    def find_user_bands(self) -> List[dict]:
        """Finds the bands that the user has liquidity in."""
        assert self.coins[0]['symbol'] == 'crvUSD', "Expected first coin to be crvUSD"

        first_user_band, last_user_band = self.read_user_tick_numbers
        bands = []
        i = 0

        for band_num in range(first_user_band, last_user_band + 1):
            min_price = self.band_min_price(band_num)
            max_price = self.band_max_price(band_num)
            band_width = max_price - min_price
            avg_price = (min_price + max_price) / 2
            crvusd_amount = self.band_balances[0][i] / PRECISION
            collateral_amount = self.band_balances[1][i] / (10 ** self.coins[1]['decimals'])
            collateral_value = collateral_amount * avg_price
            total_value_for_bars = collateral_value + crvusd_amount
            bars = {'collateral': {}, 'crvusd': {}}
            if collateral_value > 0 and crvusd_amount > 0:
                collateral_percent = collateral_value / total_value_for_bars
                crvusd_percent = crvusd_amount / total_value_for_bars
                collateral_end = min_price + band_width * collateral_percent
                collateral_mid = (min_price + collateral_end) / 2
                collateral_width = band_width * collateral_percent
                crvusd_start = collateral_end
                crvusd_mid = (crvusd_start + max_price) / 2
                crvusd_width = band_width * crvusd_percent
                bars['collateral']['position'] = collateral_mid
                bars['collateral']['width'] = collateral_width
                bars['crvusd']['position'] = crvusd_mid
                bars['crvusd']['width'] = crvusd_width
                collateral_value = collateral_amount * collateral_mid
            elif collateral_value > 0 and crvusd_amount == 0:
                bars['collateral']['position'] = avg_price
                bars['collateral']['width'] = band_width
                bars['crvusd']['position'] = 0
                bars['crvusd']['width'] = 0
            elif collateral_value == 0 and crvusd_amount > 0:
                bars['collateral']['position'] = 0
                bars['collateral']['width'] = 0
                bars['crvusd']['position'] = avg_price
                bars['crvusd']['width'] = band_width

            bands.append({
                "band_num": band_num,
                "min_price": min_price,
                "max_price": max_price,
                "avg_price": avg_price,
                "band_width": band_width,
                "crvusd_amount": crvusd_amount,
                "collateral_amount": collateral_amount,
                "collateral_as_crvusd": collateral_value,
                "total_collateral_value": collateral_value,
                "band_total_value": total_value_for_bars,
                "bars": bars
            })
            i += 1
        return bands


# ---------------------------------------------------------------------------
# Event fetching from Curve Prices API
# ---------------------------------------------------------------------------

def _classify_api_event(api_event: dict) -> tuple[str, float]:
    """Map a Curve Prices API event to (action, amount).

    Returns
    -------
    (action, amount) where action is one of: liquidated, closed,
    repay_debt, remove_collateral, borrow, add_collateral.
    """
    event_type = api_event.get("type", "")
    liquidation = api_event.get("liquidation")
    is_closed = api_event.get("is_position_closed", False)
    collateral_change = float(api_event.get("collateral_change") or 0)
    loan_change = float(api_event.get("loan_change") or 0)

    if event_type == "Repay":
        if liquidation is not None:
            liq_user = liquidation.get("user", "").lower()
            liq_liquidator = liquidation.get("liquidator", "").lower()
            if liq_user == liq_liquidator:
                return "closed", abs(float(liquidation.get("debt", 0)))
            return "liquidated", abs(float(liquidation.get("debt", 0)))
        if is_closed:
            return "closed", abs(loan_change)
        return "repay_debt", abs(loan_change)

    if event_type == "RemoveCollateral":
        return "remove_collateral", abs(collateral_change)

    if event_type == "Borrow":
        if collateral_change < 0:
            return "remove_collateral", abs(collateral_change)
        if loan_change > 0:
            return "borrow", loan_change
        if collateral_change > 0 and loan_change == 0:
            return "add_collateral", collateral_change

    return event_type.lower(), 0.0


def fetch_events(controller_address: str, user_address: str, w3: Web3,
                 rpc_manager: Optional[RPCManager] = None, log_fn=print) -> list[dict]:
    """Fetch loan events from the Curve Prices API and resolve block numbers.

    Automatically detects whether *controller_address* belongs to a crvUSD
    mint market or a lending market, then fetches collateral events for the
    given user.

    Returns a list of ``{"blocknumber": int, "action": str, "amount": float}``
    sorted by blocknumber.
    """
    base_url = "https://prices.curve.finance/v1"
    controller_lower = controller_address.lower()

    def _api_get(url: str) -> dict:
        req = urllib.request.Request(url, headers={"User-Agent": "llamma-video/1.0"})
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    # 1. Detect market type (crvUSD mint vs lending)
    log_fn("  Detecting market type via Curve Prices API...")
    markets_url = f"{base_url}/crvusd/markets"
    markets_data = _api_get(markets_url)

    crvusd_controllers: set[str] = set()
    eth_markets = (markets_data
                   .get("chains", {})
                   .get("ethereum", {})
                   .get("data", []))
    for market in eth_markets:
        addr = market.get("address", "")
        if addr:
            crvusd_controllers.add(addr.lower())

    is_crvusd = controller_lower in crvusd_controllers
    market_type = "crvusd" if is_crvusd else "lending"
    log_fn(f"  Market type: {market_type}")

    # 2. Fetch events
    events_url = (
        f"{base_url}/{market_type}/collateral_events/"
        f"ethereum/{controller_address}/{user_address}"
    )
    log_fn(f"  Fetching events from {events_url}")
    events_data = _api_get(events_url)

    api_events = events_data.get("data", [])
    if not api_events:
        log_fn("  No events found.")
        return []

    log_fn(f"  Found {len(api_events)} events, resolving block numbers...")

    # 3. Classify and resolve block numbers
    results = []
    for api_event in api_events:
        action, amount = _classify_api_event(api_event)
        tx_hash = api_event.get("transaction_hash", "")
        if not tx_hash:
            continue

        def _get_block(h=tx_hash):
            return w3.eth.get_transaction(h)["blockNumber"]

        block_number = call_with_retry(_get_block, rpc_manager=rpc_manager, log_fn=log_fn)
        results.append({
            "blocknumber": block_number,
            "action": action,
            "amount": round(amount, 2),
        })
        log_fn(f"    {action:<20} block {block_number}  amount {amount:,.2f}")

    results.sort(key=lambda e: e["blocknumber"])

    # Detect loan creation: a "borrow" is a creation if it's the first event
    # or if the previous event closed/liquidated the loan.
    for i, event in enumerate(results):
        if event["action"] != "borrow":
            continue
        if i == 0 or results[i - 1]["action"] in ("closed", "liquidated"):
            event["action"] = "loan_created"
            log_fn(f"    ^ Marking block {event['blocknumber']} as loan creation")

    log_fn(f"  Resolved {len(results)} events.")
    return results
